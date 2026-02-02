import argparse
import base64
import io
import logging
import math
import os
import re
import sys
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset
from openai import OpenAI
from rich.logging import RichHandler
from tqdm import tqdm

from evaluation.build_query import create_query_data
from utilities import read_json, save_json


# -----------------------------
# Helpers (validation / parsing)
# -----------------------------
def verify_response(response: Any) -> bool:
    """
    Decide whether an existing saved response is "valid" enough to skip reruns.
    """
    if response is None:
        return False
    if not isinstance(response, str):
        try:
            response = str(response)
        except Exception:
            return False

    resp = response.strip()
    if resp == "":
        return False
    if "Response Error" in resp:
        return False
    if resp.lower() in {"unsure", "unable to determine", "cannot determine", "unknown"}:
        return False
    return True


_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_python_code(text: str) -> str:
    """
    Extract python code from a model response.
    Accepts either raw code or a ```python ... ``` fenced block.
    """
    if not isinstance(text, str):
        return ""
    m = _CODE_FENCE_RE.search(text)
    if m:
        return (m.group(1) or "").strip()
    return text.strip()


_NUM_RE = re.compile(r"[-+]?(?:\d+\.\d+|\d+)(?:[eE][-+]?\d+)?")


def _try_parse_float(s: str) -> Optional[float]:
    try:
        return float(s)
    except Exception:
        return None


def normalize_for_compare(s: str) -> str:
    # Remove spaces and common punctuation that often varies.
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", "", s)
    s = s.replace(",", "")
    return s


def extract_answer_from_text(raw: str, problem: Dict[str, Any]) -> str:
    """
    Extract a clean answer string from potentially verbose model output.
    Preference order:
      1) Exact choice text if multi_choice
      2) A/B/C/... mapping if multi_choice
      3) Last number if numeric
      4) First non-empty line
    """
    if not isinstance(raw, str):
        try:
            raw = str(raw)
        except Exception:
            return ""
    raw_str = raw.strip()
    if raw_str == "":
        return ""

    choices = problem.get("choices", None)
    if choices:
        # 1) Exact choice substring match
        for c in choices:
            if isinstance(c, str) and c.strip() != "" and c in raw_str:
                return c.strip()

        # 2) A/B/C/D mapping
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        # Search for single letter answer like "A" or "(B)" or "Answer: C"
        m = re.search(r"(?:^|[\s:（(])([A-Z])(?:$|[\s\)）\.!,])", raw_str.upper())
        if m:
            letter = m.group(1)
            idx = letters.find(letter)
            if 0 <= idx < len(choices):
                c = choices[idx]
                return c.strip() if isinstance(c, str) else str(c)

        # 3) fallback: first line
        return raw_str.splitlines()[0].strip()

    # Numeric-ish answer types
    if problem.get("answer_type") in {"float", "integer", "number"} or problem.get("precision") is not None:
        nums = _NUM_RE.findall(raw_str)
        if nums:
            return nums[-1].strip()

    # Otherwise: first non-empty line
    for line in raw_str.splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def answers_agree(a: str, b: str, problem: Dict[str, Any]) -> bool:
    """
    Soft equality check used before invoking verifier.
    """
    a = (a or "").strip()
    b = (b or "").strip()
    if a == "" or b == "":
        return False

    # If choices exist, compare normalized strings.
    if problem.get("choices"):
        return normalize_for_compare(a) == normalize_for_compare(b)

    # Try numeric compare
    fa = _try_parse_float(a)
    fb = _try_parse_float(b)
    if fa is not None and fb is not None:
        # Precision-aware tolerance
        prec = problem.get("precision", None)
        if isinstance(prec, (int, float)) and prec > 0:
            tol = 0.5 * float(prec)
        else:
            tol = 1e-6
        return abs(fa - fb) <= tol

    return normalize_for_compare(a) == normalize_for_compare(b)


# -----------------------------
# Safe-ish code execution
# -----------------------------
_BANNED_CODE_PATTERNS = [
    r"(?m)^\s*(import|from)\b",
    r"\bimport\s+os\b",
    r"\bimport\s+sys\b",
    r"\bimport\s+subprocess\b",
    r"\bimport\s+socket\b",
    r"\bimport\s+requests\b",
    r"\bimport\s+pathlib\b",
    r"\bimport\s+shutil\b",
    r"\bfrom\s+os\b",
    r"\bfrom\s+sys\b",
    r"\bsubprocess\.",
    r"\bsocket\.",
    r"\bopen\(",
    r"\b__import__\b",
    r"\beval\(",
    r"\bexec\(",
    r"\bcompile\(",
    r"\binput\(",
    r"\bglobals\(",
    r"\blocals\(",
    r"\b__\w+__\b",
]


def _is_code_safe_enough(code: str) -> Tuple[bool, str]:
    if not code.strip():
        return False, "Empty code"
    for pat in _BANNED_CODE_PATTERNS:
        if re.search(pat, code):
            return False, f"Blocked by safety pattern: {pat}"
    return True, ""


# Allow-listed import rewriting (since sandbox disables __import__)
_ALLOWED_IMPORT_MODULES = {"math", "fractions", "decimal"}
_IMPORT_LINE_RE = re.compile(r"^\s*import\s+(.+?)\s*$")
_FROM_IMPORT_LINE_RE = re.compile(r"^\s*from\s+([a-zA-Z_]\w*)\s+import\s+(.+?)\s*$")

def rewrite_safe_imports(code: str) -> Tuple[str, Optional[str]]:
    """Rewrite a small subset of safe imports to assignments.

    Why: the sandbox removes the __import__ builtin, so `import math` raises:
          ImportError: __import__ not found

    Supported patterns:
      - import math
      - import math as m
      - import math, decimal
      - from math import sqrt
      - from math import sqrt as s, sin

    Everything else is rejected.
    """
    if not isinstance(code, str):
        try:
            code = str(code)
        except Exception:
            return "", "Code is not a string"

    out_lines: List[str] = []
    for line in code.splitlines():
        m = _IMPORT_LINE_RE.match(line)
        if m:
            rest = m.group(1)
            parts = [p.strip() for p in rest.split(",")]
            for part in parts:
                if not part:
                    continue
                m2 = re.match(r"^([a-zA-Z_]\w*)(?:\s+as\s+([a-zA-Z_]\w*))?$", part)
                if not m2:
                    return "", f"Disallowed import syntax: {line.strip()}"
                mod = m2.group(1)
                alias = m2.group(2)
                if mod not in _ALLOWED_IMPORT_MODULES:
                    return "", f"Disallowed import: {mod}"
                # Module object is already in globals; just bind alias if needed.
                if alias:
                    out_lines.append(f"{alias} = {mod}")
            continue

        m = _FROM_IMPORT_LINE_RE.match(line)
        if m:
            mod = m.group(1)
            items = (m.group(2) or "").strip()
            if mod not in _ALLOWED_IMPORT_MODULES:
                return "", f"Disallowed import: {mod}"
            if items == "*" or "*" in items:
                return "", f"Disallowed star import from {mod}"
            parts = [p.strip() for p in items.split(",") if p.strip()]
            if not parts:
                return "", f"Disallowed import syntax: {line.strip()}"
            for part in parts:
                m2 = re.match(r"^([a-zA-Z_]\w*)(?:\s+as\s+([a-zA-Z_]\w*))?$", part)
                if not m2:
                    return "", f"Disallowed import syntax: {line.strip()}"
                name = m2.group(1)
                alias = m2.group(2) or name
                out_lines.append(f"{alias} = {mod}.{name}")
            continue

        out_lines.append(line)

    return "\n".join(out_lines), None


def _sandbox_worker(code: str, q) -> None:
    """
    Run code in a separate process, capture stdout, return (stdout, error_str).
    """
    import io as _io
    import sys as _sys
    import math as _math
    import fractions as _fractions
    import decimal as _decimal

    # Restrictive builtins
    safe_builtins = {
        "abs": abs,
        "min": min,
        "max": max,
        "sum": sum,
        "range": range,
        "len": len,
        "round": round,
        "pow": pow,
        "print": print,
        "int": int,
        "float": float,
        "str": str,
        "enumerate": enumerate,
        "zip": zip,
        "map": map,
        "list": list,
        "dict": dict,
        "set": set,
        "tuple": tuple,
    }

    safe_globals = {
        "__builtins__": safe_builtins,
        "math": _math,
        "fractions": _fractions,
        "decimal": _decimal,
    }

    old_stdout = _sys.stdout
    new_stdout = _io.StringIO()
    _sys.stdout = new_stdout

    err = None
    try:
        exec(code, safe_globals, {})
    except Exception as e:
        err = "".join(traceback.format_exception(type(e), e, e.__traceback__))

    _sys.stdout = old_stdout
    out = new_stdout.getvalue().strip()

    q.put((out, err))


def evaluate_code_sandbox(code_string: str, timeout_s: int = 5) -> Tuple[str, Optional[str]]:
    """
    Execute python code in a subprocess with basic safety checks and a timeout.
    Returns: (stdout, error_string_or_None)
    """
    code = code_string or ""

    # Rewrite allow-listed imports to avoid __import__ in the sandbox
    code, import_err = rewrite_safe_imports(code)
    if import_err:
        return "", import_err

    ok, reason = _is_code_safe_enough(code)
    if not ok:
        return "", reason

    try:
        import multiprocessing as mp
        ctx = mp.get_context("fork") if hasattr(mp, "get_context") else mp
        q: Any = ctx.Queue()
        p = ctx.Process(target=_sandbox_worker, args=(code, q))
        p.daemon = True
        p.start()
        p.join(timeout_s)

        if p.is_alive():
            p.terminate()
            p.join(1)
            return "", f"Timeout after {timeout_s}s"

        if q.empty():
            return "", "No output from sandbox process"
        out, err = q.get_nowait()
        return out, err
    except Exception as e:
        return "", f"Sandbox failure: {e}"


def last_nonempty_line(s: str) -> str:
    if not isinstance(s, str):
        return ""
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


# -----------------------------
# Image helper for OpenAI
# -----------------------------
def pil_image_to_data_url(img, fmt: str = "PNG") -> str:
    """
    Convert PIL.Image -> data URL for OpenAI multimodal input.
    """
    try:
        if hasattr(img, "mode") and img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
    except Exception:
        pass

    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


class OpenAIChatModel:
    """
    Wrapper for OpenAI chat.completions with optional image input.
    """

    def __init__(
        self,
        client: OpenAI,
        model: str = "gpt-4o",
        temperature: float = 0.2,
        max_tokens: int = 512,
    ):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _attach_image_to_last_user(self, messages: List[Dict[str, Any]], decoded_image: Optional[object]) -> List[Dict[str, Any]]:
        if decoded_image is None:
            return messages

        # Find last user message
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                content = messages[i].get("content", "")
                if isinstance(content, list):
                    parts = content
                else:
                    parts = [{"type": "text", "text": str(content)}]

                try:
                    data_url = pil_image_to_data_url(decoded_image, fmt="PNG")
                    parts.append({"type": "image_url", "image_url": {"url": data_url}})
                except Exception as e:
                    logging.warning(f"Failed to attach image; fallback to text-only. Error: {e}")

                messages[i]["content"] = parts
                break
        return messages

    def chat(
        self,
        messages: List[Dict[str, Any]],
        decoded_image: Optional[object] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        messages = [dict(m) for m in messages]  # shallow copy
        messages = self._attach_image_to_last_user(messages, decoded_image)

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=self.max_tokens if max_tokens is None else max_tokens,
        )
        msg = resp.choices[0].message.content
        return msg if msg is not None else ""

    def get_response(self, user_prompt: str, decoded_image: Optional[object] = None) -> str:
        return self.chat([{"role": "user", "content": [{"type": "text", "text": user_prompt}]}], decoded_image=decoded_image)


# -----------------------------
# Agent prompts
# -----------------------------
def build_problem_context(problem: Dict[str, Any], extra_context: Optional[str] = None) -> str:
    q = (problem.get("question") or "").strip()
    choices = problem.get("choices")
    unit = problem.get("unit")
    precision = problem.get("precision")
    qtype = problem.get("question_type")
    atype = problem.get("answer_type")

    parts = [
        "PROBLEM",
        f"question_type: {qtype}",
        f"answer_type: {atype}",
        f"precision: {precision}",
        f"unit: {unit}",
        "",
        "Question:",
        q,
    ]
    if choices:
        parts += ["", "Choices:"]
        for i, c in enumerate(choices):
            letter = chr(ord("A") + i)
            parts.append(f"{letter}. {c}")
    if extra_context:
        parts += ["", "Additional context (caption/OCR/few-shot/etc):", extra_context.strip()]
    return "\n".join(parts).strip()


TASK_INFERENCE_SYSTEM = """You are Module-1 (Task Inference).
Goal: Write a SHORT Python program to solve the problem.

Rules:
- Output ONLY a python code block fenced with ```python ...```. No prose.
- The code MUST end by printing ONLY the final answer (single line) via print(...).
- Do NOT use any import/from statements. The modules math, fractions, decimal are already available as variables.
- No file I/O, no network, no subprocess, no OS/system access.
- You may use: math, fractions, decimal.
- If choices are provided, you MUST print EXACTLY one of the provided choices (copy-paste the text).
- If you truly cannot solve from the given information, print UNSURE.

Think silently; only output code."""
# NOTE: we intentionally keep it image-free. The direct solver will use the image.


DIRECT_SOLVER_SYSTEM = """You are Module-3 (Direct Solver).
Solve the problem using the provided image (if any) and text.

Output format rules:
- Output ONLY the final answer (no explanation).
- If choices are provided, output EXACTLY one of the provided choices (copy-paste).
- If numeric, output only the number (and unit only if explicitly requested by the question).
- If you cannot determine, output UNSURE."""


VERIFIER_SYSTEM = """You are Module-4 (Verifier).
You will be given the original problem and two candidate answers:

- Answer_A: produced by executing code from Module-2 (may have runtime errors).
- Answer_B: produced by the direct solver (Module-3).

Task:
- If both answers are the same (semantically), output that answer.
- If one answer is UNSURE/empty/error and the other is not, choose the non-UNSURE answer.
- If both differ and both look plausible, choose the one most consistent with the problem, choices, units, and precision.

Output ONLY the final answer (no explanation)."""


@dataclass
class ModuleLog:
    prompt: str = ""
    raw: str = ""
    extracted: str = ""
    error: str = ""


def run_agent_on_problem(
    model: OpenAIChatModel,
    problem: Dict[str, Any],
    problem_decoded_image: Optional[object],
    query: str,
    executor_timeout: int,
    max_tokens: int,
) -> Tuple[str, Dict[str, Any]]:
    """
    Runs:
      M1 task inference -> python code
      M2 executor -> answer_A
      M3 direct solver -> answer_B
      M4 verifier -> final
    Returns (final_answer, agent_trace_dict)
    """
    trace: Dict[str, Any] = {"modules": {}}

    # -------- Module 1: Task inference (code gen) --------
    m1 = ModuleLog()
    m1.prompt = build_problem_context(problem, extra_context=query)
    try:
        raw = model.chat(
            messages=[
                {"role": "system", "content": TASK_INFERENCE_SYSTEM},
                {"role": "user", "content": m1.prompt},
            ],
            decoded_image=None,
            temperature=0.2,
            max_tokens=max_tokens,
        )
        m1.raw = raw
        m1.extracted = extract_python_code(raw)
    except Exception as e:
        m1.error = f"{e}"
    trace["modules"]["task_inference"] = m1.__dict__

    # -------- Module 2: Code executor --------
    m2: Dict[str, Any] = {"stdout": "", "answer": "", "error": ""}
    try:
        if m1.extracted.strip():
            stdout, err = evaluate_code_sandbox(m1.extracted, timeout_s=executor_timeout)
            m2["stdout"] = stdout
            m2["error"] = "" if err is None else str(err)
            m2["answer"] = extract_answer_from_text(last_nonempty_line(stdout), problem)
        else:
            m2["error"] = "No code from Module-1"
            m2["answer"] = "UNSURE"
    except Exception as e:
        m2["error"] = f"{e}"
        m2["answer"] = "UNSURE"
    trace["modules"]["code_executor"] = m2

    ans_a = (m2.get("answer") or "").strip()
    if not ans_a:
        ans_a = "UNSURE"

    # -------- Module 3: Direct solver (image + text) --------
    m3 = ModuleLog()
    m3.prompt = build_problem_context(problem, extra_context=query)
    try:
        raw = model.chat(
            messages=[
                {"role": "system", "content": DIRECT_SOLVER_SYSTEM},
                {"role": "user", "content": m3.prompt},
            ],
            decoded_image=problem_decoded_image,
            temperature=0.2,
            max_tokens=max_tokens,
        )
        m3.raw = raw
        m3.extracted = extract_answer_from_text(raw, problem)
    except Exception as e:
        m3.error = f"{e}"
        m3.extracted = "UNSURE"
    trace["modules"]["direct_solver"] = m3.__dict__

    ans_b = (m3.extracted or "").strip()
    if not ans_b:
        ans_b = "UNSURE"

    # -------- Module 4: Verifier --------
    m4 = ModuleLog()
    if answers_agree(ans_a, ans_b, problem) and ans_a.lower() != "unsure":
        final_answer = ans_a
        m4.prompt = "AGREED (skipped LLM verifier)"
        m4.raw = ""
        m4.extracted = final_answer
        m4.error = ""
        trace["modules"]["verifier"] = m4.__dict__
        trace["final_answer"] = final_answer
        trace["answer_a"] = ans_a
        trace["answer_b"] = ans_b
        return final_answer, trace

    verifier_user = "\n".join(
        [
            build_problem_context(problem, extra_context=query),
            "",
            f"Answer_A (from code executor): {ans_a}",
            f"Answer_A_error: {m2.get('error','')}",
            "",
            f"Answer_B (from direct solver): {ans_b}",
        ]
    ).strip()

    m4.prompt = verifier_user
    try:
        raw = model.chat(
            messages=[
                {"role": "system", "content": VERIFIER_SYSTEM},
                {"role": "user", "content": verifier_user},
            ],
            decoded_image=None,
            temperature=0.0,
            max_tokens=128,
        )
        m4.raw = raw
        m4.extracted = extract_answer_from_text(raw, problem)
    except Exception as e:
        m4.error = f"{e}"
        # Fallback heuristic
        if ans_b.lower() != "unsure":
            m4.extracted = ans_b
        elif ans_a.lower() != "unsure":
            m4.extracted = ans_a
        else:
            m4.extracted = "UNSURE"

    final_answer = (m4.extracted or "").strip() or "UNSURE"
    trace["modules"]["verifier"] = m4.__dict__
    trace["final_answer"] = final_answer
    trace["answer_a"] = ans_a
    trace["answer_b"] = ans_b
    return final_answer, trace


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    # input
    parser.add_argument('--dataset_name', type=str, default='AI4Math/MathVista')
    parser.add_argument('--test_split_name', type=str, default='testmini')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--input_file', type=str, default='testmini.json')
    # output
    parser.add_argument('--output_dir', type=str, default='./results/bard')
    parser.add_argument('--output_file', type=str, default='output_bard.json')
    parser.add_argument('--log_file', type=str, default='agent_logs.json', help='Write detailed agent module logs/errors to this separate JSON file (in output_dir).')
    parser.add_argument('--max_num_problems', type=int, default=-1, help='The number of problems to run')
    parser.add_argument('--save_every', type=int, default=100, help='save every n problems')

    # Pipeline switch
    parser.add_argument(
        '--agent_pipeline',
        type=str,
        default='agent',
        choices=['direct', 'agent'],
        help="direct: single gpt-4o call (original). agent: M1->M2->M3->M4 pipeline.",
    )

    # Executor
    parser.add_argument('--executor_timeout', type=int, default=5, help='Timeout (seconds) for Module-2 sandbox.')

    # Local Model (not implemented)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)

    # Remote model (official OpenAI client)
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-4o',
        help='llm engine (e.g., gpt-4o)',
    )
    parser.add_argument('--key', type=str, default='', help='OpenAI API key (or set OPENAI_API_KEY env var)')

    # query
    parser.add_argument('--query_file', type=str, default=None)
    parser.add_argument('--caption_file', type=str, default='./data/texts/captions_bard.json')
    parser.add_argument('--ocr_file', type=str, default='./data/texts/ocrs_easyocr.json')
    parser.add_argument('--shot_type', type=str, default='solution', help='shot type', choices=['solution', 'code'])
    parser.add_argument('--shot_num', type=int, default=0, help='number of shot examples')
    parser.add_argument('--use_caption', action='store_true', help='use caption data')
    parser.add_argument('--use_ocr', action='store_true', help='use ocr data')

    # other settings
    parser.add_argument('--rerun', action='store_true', help='rerun answer extraction for all problems')
    parser.add_argument('--debug', action='store_true', help='debug mode')

    args = parser.parse_args()
    return args


def main():
    logging.info("MathVista: Generating Responses - Start")
    args = parse_args()

    # load data
    logging.info(f"Loading dataset {args.dataset_name}, split {args.test_split_name}...")
    data_list = load_dataset(args.dataset_name, split=args.test_split_name)
    data = {item['pid']: item for item in data_list}

    # load or create query data
    if args.query_file:
        query_file = os.path.join(args.data_dir, args.query_file)
        if os.path.exists(query_file):
            logging.info(f"Loading existing {query_file}...")
            query_data = read_json(query_file)
        else:
            raise FileNotFoundError(f"Query file not found: {query_file}")
    else:
        logging.info("Creating new query...")

        caption_data = {}
        if args.use_caption:
            caption_file = args.caption_file
            if os.path.exists(caption_file):
                logging.info(f"Reading {caption_file}...")
                try:
                    caption_data = read_json(caption_file)["texts"]
                    logging.info("Caption data loaded.")
                except Exception:
                    logging.info("Caption data not found!! Please Check.")

        ocr_data = {}
        if args.use_ocr:
            ocr_file = args.ocr_file
            if os.path.exists(ocr_file):
                logging.info(f"Reading {ocr_file}...")
                try:
                    ocr_data = read_json(ocr_file)["texts"]
                    logging.info("OCR data loaded.")
                except Exception:
                    logging.info("OCR data not found!! Please Check.")

        query_data = create_query_data(data, caption_data, ocr_data, args)

    # model init
    if args.model_path:
        logging.info(f"Loading model from {args.model_path}...")
        raise NotImplementedError("Local models are not yet supported.")
    else:
        model_name = args.model
        logging.info(f"Loading {model_name} via official OpenAI client...")

        key = args.key.strip() if args.key else os.getenv("OPENAI_API_KEY", "").strip()
        assert key != "", "OpenAI API key is missing. Set OPENAI_API_KEY env var or pass --key."

        client = OpenAI(api_key=key)
        model = OpenAIChatModel(
            client=client,
            model=model_name,
            temperature=args.temperature,
            max_tokens=args.max_new_tokens,
        )

    logging.info("Model loaded.")

    full_pids = list(data.keys())

    os.makedirs(args.output_dir, exist_ok=True)
    output_file_path = os.path.join(args.output_dir, args.output_file)
    log_file_path = os.path.join(args.output_dir, args.log_file)

    # load results
    if os.path.exists(output_file_path):
        logging.info("Results already exist.")
        logging.info(f"Reading {output_file_path}...")
        results = read_json(output_file_path)
    else:
        results = {}

    # load agent logs (kept separate from answers)
    if os.path.exists(log_file_path):
        try:
            logging.info(f"Reading existing agent logs: {log_file_path}...")
            agent_logs = read_json(log_file_path)
        except Exception:
            agent_logs = {}
    else:
        agent_logs = {}

    # skipping
    skip_pids = []
    if not args.rerun:
        for problem_id in full_pids:
            if problem_id in results and 'response' in results[problem_id]:
                response = results[problem_id]['response']
                if verify_response(response):
                    skip_pids.append(problem_id)

    if len(skip_pids) > 0:
        logging.info(
            f"Found existing results file with {len(skip_pids)} problems with valid responses. Skipping these problems..."
        )

    test_pids = [pid for pid in full_pids if pid not in skip_pids]

    if args.max_num_problems > 0:
        test_pids = test_pids[: min(args.max_num_problems, len(test_pids))]
        logging.warning(f'Limiting number of problems to {args.max_num_problems}.')

    logging.info(f"Number of test problems to run: {len(test_pids)}")

    for i, problem_id in enumerate(tqdm(test_pids)):
        problem: dict = data[problem_id].copy()

        # Remove decoded Image for JSON serialization
        problem_decoded_image = problem.get('decoded_image', None)
        if 'decoded_image' in problem:
            problem.pop('decoded_image')

        query = query_data[problem_id]

        logging.debug("--------------------------------------------------------------")
        logging.debug(f"Generating response for problem: {problem_id}...")

        results[problem_id] = problem
        results[problem_id]['query'] = query

        try:
            if args.agent_pipeline == "direct":
                response = model.get_response(user_prompt=query, decoded_image=problem_decoded_image)
                results[problem_id]['response'] = response
            else:
                final_answer, trace = run_agent_on_problem(
                    model=model,
                    problem=problem,
                    problem_decoded_image=problem_decoded_image,
                    query=query,
                    executor_timeout=args.executor_timeout,
                    max_tokens=args.max_new_tokens,
                )
                # For compatibility with MathVista scoring scripts, keep top-level 'response'
                results[problem_id]['response'] = final_answer

                # Save detailed module logs/errors to the separate log file (NOT mixed into output_file)
                errors = []
                for mname, mval in trace.get("modules", {}).items():
                    err = mval.get("error") if isinstance(mval, dict) else None
                    if err:
                        errors.append({"module": mname, "error": err})

                agent_logs[problem_id] = {
                    "pid": problem_id,
                    "final_answer": final_answer,
                    "answer_a": trace.get("answer_a", ""),
                    "answer_b": trace.get("answer_b", ""),
                    "modules": trace.get("modules", {}),
                    "errors": errors,
                    # Include minimal context for debugging (kept out of scoring file)
                    "question": problem.get("question", ""),
                    "image": problem.get("image", ""),
                    "choices": problem.get("choices", None),
                    "unit": problem.get("unit", None),
                    "precision": problem.get("precision", None),
                    "answer_type": problem.get("answer_type", None),
                    "question_type": problem.get("question_type", None),
                }

                logging.debug(f"[Agent] answer_a={trace.get('answer_a')} answer_b={trace.get('answer_b')} final={final_answer}")

        except Exception as e:
            logging.error(f"Error in extracting answer for {problem_id}")
            logging.error(e)
            results[problem_id]['error'] = str(e)
            results[problem_id]['response'] = "UNSURE"
            agent_logs[problem_id] = {
                "pid": problem_id,
                "final_answer": "UNSURE",
                "errors": [{"module": "pipeline", "error": str(e)}],
                "question": problem.get("question", ""),
                "image": problem.get("image", ""),
                "choices": problem.get("choices", None),
            }

        if (i % args.save_every == 0 and i > 0) or i == len(test_pids) - 1:
            try:
                save_json(results, output_file_path)
                logging.info(f"Saved results to {output_file_path}")
                if args.agent_pipeline == "agent":
                    save_json(agent_logs, log_file_path)
                    logging.info(f"Saved agent logs to {log_file_path}")
            except Exception as e:
                logging.info(f"Error in saving {output_file_path}")
                logging.info(e)

    logging.info("MathVista: Generating Responses - Finish")


if __name__ == '__main__':
    logging.basicConfig(
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        format="[%(name)s] %(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                rich_tracebacks=True,
                markup=False,
                show_path=False,
                omit_repeated_times=False,
            )
        ],
    )
    logger_blocklist = [
        "asyncio",
        "datasets",
        "httpx",
        "httpcore",
        "filelock",
        "fsspec",
        "openai",
        "PIL",
        "urllib3",
    ]
    for module in logger_blocklist:
        logging.getLogger(module).setLevel(logging.WARNING)

    main()
