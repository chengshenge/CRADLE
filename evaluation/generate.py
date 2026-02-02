import argparse
import base64
import io
import json
import logging
import os
import re
import signal
import sys
import traceback
import datetime
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from datasets import load_dataset
from openai import OpenAI
from rich.logging import RichHandler
from tqdm import tqdm

from evaluation.build_query import create_query_data
from utilities import read_json, save_json


# -----------------------------
# Utilities
# -----------------------------
def verify_response(response: Any) -> bool:
    if response is None:
        return False
    if isinstance(response, str):
        response = response.strip()
        if response == "":
            return False
        if "Response Error" in response:
            return False
    return True



def setup_file_logging(output_dir: str, log_filename: str = "agent_debug.log") -> str:
    """Attach a file handler to the root logger so you can inspect full traces after a run."""
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, log_filename)

    root = logging.getLogger()
    # Avoid duplicate handlers if main() is called multiple times
    for h in list(root.handlers):
        if isinstance(h, logging.FileHandler):
            try:
                if os.path.abspath(getattr(h, "baseFilename", "")) == os.path.abspath(log_path):
                    return log_path
            except Exception:
                pass

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    root.addHandler(fh)
    return log_path


def _append_jsonl(path: str, record: Dict[str, Any]) -> None:
    """Append one JSON record per line (safe for post-run grep/analysis)."""
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        # Never crash the run because logging failed
        logging.debug("Failed to write jsonl log.", exc_info=True)


def record_module_error(
    error_jsonl_path: Optional[str],
    pid: str,
    stage: str,
    err: Any,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Write a structured error record so you can see which module failed."""
    if not error_jsonl_path:
        return
    rec = {
        "ts": datetime.datetime.now().isoformat(timespec="seconds"),
        "pid": str(pid),
        "stage": stage,
        "error": repr(err),
    }
    if extra:
        rec["extra"] = extra
    # Best-effort traceback if it's an exception
    if isinstance(err, BaseException):
        try:
            rec["traceback"] = "".join(traceback.format_exception(type(err), err, err.__traceback__))
        except Exception:
            rec["traceback"] = traceback.format_exc()
    _append_jsonl(error_jsonl_path, rec)

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


def strip_code_fences(text: str) -> str:
    """
    Extract python code from common markdown fences.
    If no fence found, returns the original text.
    """
    if not isinstance(text, str):
        return ""
    s = text.strip()

    # ```python ... ```
    m = re.search(r"```(?:python)?\s*(.*?)\s*```", s, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # ``` ... ``` (no language)
    m = re.search(r"```\s*(.*?)\s*```", s, flags=re.DOTALL)
    if m:
        return m.group(1).strip()

    return s


def extract_json_object(text: str) -> Optional[dict]:
    """
    Best-effort: extract the first JSON object from arbitrary text.
    """
    if not isinstance(text, str):
        return None
    s = text.strip()
    # find first {...} block
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if not m:
        return None
    chunk = m.group(0)
    try:
        return json.loads(chunk)
    except Exception:
        return None


def normalize_answer_for_compare(ans: Any) -> str:
    """
    Loose normalization for agreement check.
    (Do NOT use this for grading; it's only for module agreement.)
    """
    if ans is None:
        return ""
    if not isinstance(ans, str):
        ans = str(ans)

    s = ans.strip()
    s = s.replace("答案：", "").replace("Answer:", "").strip()
    s = re.sub(r"\s+", " ", s)

    # normalize unicode degree sign variants
    s = s.replace("º", "°")
    # remove surrounding quotes
    s = s.strip("\"'“”‘’")
    return s


def extract_answer_from_exec_output(exec_out: str) -> str:
    """
    Prefer the last non-empty line as the produced answer.
    Also supports 'FINAL_ANSWER: ...' convention.
    """
    if not isinstance(exec_out, str):
        return ""
    lines = [ln.strip() for ln in exec_out.strip().splitlines() if ln.strip()]
    if not lines:
        return ""
    # Look for explicit tag
    for ln in reversed(lines):
        m = re.search(r"FINAL_ANSWER\s*[:=]\s*(.*)$", ln)
        if m:
            return m.group(1).strip()
    return lines[-1]


# -----------------------------
# Sandboxed code execution
# -----------------------------
class _Timeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _Timeout("Execution timed out")


def evaluate_code_sandboxed(code_string: str, timeout_sec: int = 3) -> Tuple[str, Optional[Exception]]:
    """
    Execute untrusted python code in a *restricted* sandbox.
    - No imports
    - No file/network access
    - Short timeout
    The code is expected to print the final answer (single line preferred).
    """
    code_string = strip_code_fences(code_string)

    # Redirect stdout
    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout

    # Timeout via signal (works on Linux/macOS; WSL is OK)
    old_handler = None
    try:
        old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(max(1, int(timeout_sec)))
    except Exception:
        old_handler = None  # fallback: no alarm

    # Restricted builtins: keep only safe essentials
    safe_builtins = {
        "abs": abs,
        "min": min,
        "max": max,
        "sum": sum,
        "len": len,
        "range": range,
        "enumerate": enumerate,
        "zip": zip,
        "map": map,
        "filter": filter,
        "sorted": sorted,
        "round": round,
        "int": int,
        "float": float,
        "str": str,
        "bool": bool,
        "list": list,
        "dict": dict,
        "set": set,
        "tuple": tuple,
        "print": print,
    }

    # Provide common math utilities without allowing imports
    import math
    from fractions import Fraction
    from decimal import Decimal, getcontext

    safe_globals = {
        "__builtins__": safe_builtins,
        "math": math,
        "Fraction": Fraction,
        "Decimal": Decimal,
        "getcontext": getcontext,
    }
    safe_locals: Dict[str, Any] = {}

    error: Optional[Exception] = None
    try:
        exec(code_string, safe_globals, safe_locals)
    except Exception as e:
        error = e
    finally:
        # Cancel alarm & restore handler
        try:
            signal.alarm(0)
        except Exception:
            pass
        try:
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)
        except Exception:
            pass
        sys.stdout = old_stdout

    captured_output = new_stdout.getvalue()
    if isinstance(captured_output, str):
        captured_output = captured_output.strip()

    return captured_output, error


# -----------------------------
# OpenAI wrapper
# -----------------------------
class OpenAIChatModel:
    """
    Wrapper that supports:
    - system prompt
    - text + optional image (data URL)
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

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        decoded_image: Optional[object] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        content = [{"type": "text", "text": user_prompt}]
        if decoded_image is not None:
            try:
                data_url = pil_image_to_data_url(decoded_image, fmt="PNG")
                content.append({"type": "image_url", "image_url": {"url": data_url}})
            except Exception as e:
                logging.warning(f"Failed to attach image; fallback to text-only. Error: {e}")

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=self.max_tokens if max_tokens is None else max_tokens,
        )
        msg = resp.choices[0].message.content
        return msg if msg is not None else ""


# -----------------------------
# 4-Module agent
# -----------------------------
TASK_INFERENCE_SYSTEM = """You are Module-1: Task Inference for multimodal math problems.
Goal: Read the problem, then write Python code to solve it.

Strict requirements:
- Output ONLY Python code. No markdown fences, no explanations.
- The code MUST print the final answer (single line). Prefer:
    FINAL_ANSWER = "..."
    print(FINAL_ANSWER)
- Do NOT import anything. You may use: math, Fraction, Decimal (already available).
- Do NOT read/write files, do NOT access network, do NOT use randomness.
- If the problem is multiple-choice, print the EXACT option string (e.g., "145°"), not A/B/C/D.
- If precision is given (e.g., 1.0), format that many decimals.
- If a unit is provided, include the unit in the printed answer exactly as required.
"""

DIRECT_SOLVE_SYSTEM = """You are Module-3: Direct Solver (gpt-4o).
Solve the given problem directly from the text + image.

Strict requirements:
- Output ONLY the final answer (single line). No reasoning.
- If multiple-choice, output the EXACT option string (not A/B/C/D).
- If precision is given, match it (e.g., 1 decimal place).
- If a unit is provided, include it exactly.
"""

VERIFIER_SYSTEM = """You are Module-4: Verifier & Arbiter.
You will be given:
- the original problem (text + meta)
- Module-2 output (executed-code answer + any error)
- Module-3 output (direct answer)

Your job:
1) Determine whether the two answers are the same (after light normalization).
2) If they match -> output that answer.
3) If they differ -> choose the more reasonable one. Prefer:
   - a non-empty answer over empty/error
   - an answer that matches the problem type (e.g., is one of the choices for multi-choice)
   - sensible units/precision if specified

Output STRICT JSON ONLY:
{
  "final_answer": "...",
  "agree": true/false,
  "chosen": "code" | "direct",
  "notes": "one short sentence"
}
No extra keys, no markdown.
"""


def build_problem_payload(problem: Dict[str, Any], query_text: str) -> Dict[str, Any]:
    # Keep only stable fields (avoid huge blobs)
    keys = [
        "pid",
        "question",
        "choices",
        "unit",
        "precision",
        "answer_type",
        "question_type",
        "metadata",
    ]
    payload = {k: problem.get(k, None) for k in keys if k in problem}
    payload["extra_context"] = query_text  # may include caption/OCR depending on your query builder
    return payload


@dataclass
class AgentOutputs:
    module1_code: str = ""
    module2_stdout: str = ""
    module2_error: str = ""
    module2_answer: str = ""
    module3_answer: str = ""
    module4_raw: str = ""
    module4_json: Dict[str, Any] = None
    final_answer: str = ""


class FourModuleAgent:
    def __init__(
        self,
        model: OpenAIChatModel,
        exec_timeout_sec: int = 3,
        ti_max_tokens: int = 900,
        direct_max_tokens: int = 256,
        verifier_max_tokens: int = 256,
        error_jsonl_path: Optional[str] = None,
    ):
        self.model = model
        self.exec_timeout_sec = exec_timeout_sec
        self.ti_max_tokens = ti_max_tokens
        self.direct_max_tokens = direct_max_tokens
        self.verifier_max_tokens = verifier_max_tokens
        self.error_jsonl_path = error_jsonl_path
        self.logger = logging.getLogger('agent')

    
    def run(self, problem: Dict[str, Any], query_text: str, decoded_image: Optional[object]) -> AgentOutputs:
        out = AgentOutputs(module4_json={})
        pid = str(problem.get("pid", ""))

        payload = build_problem_payload(problem, query_text)
        user_blob = json.dumps(payload, ensure_ascii=False)

        self.logger.debug(f"[pid={pid}] starting four-module agent")

        # -----------------
        # Module-1: task inference -> code
        # -----------------
        try:
            m1_user = f"PROBLEM_JSON:\n{user_blob}\n\nWrite Python code to solve it."
            self.logger.debug(f"[pid={pid}] [M1] start")
            out.module1_code = self.model.chat(
                system_prompt=TASK_INFERENCE_SYSTEM,
                user_prompt=m1_user,
                decoded_image=decoded_image,    # allow reading diagrams if useful
                temperature=0.0,
                max_tokens=self.ti_max_tokens,
            ).strip()
            self.logger.debug(f"[pid={pid}] [M1] done (code_len={len(out.module1_code)})")
        except Exception as e:
            self.logger.exception(f"[pid={pid}] [M1] failed")
            record_module_error(self.error_jsonl_path, pid, "M1", e, extra={"hint": "task_inference/chat"})
            out.module1_code = ""

        # -----------------
        # Module-2: execute code
        # -----------------
        try:
            self.logger.debug(f"[pid={pid}] [M2] start")
            if out.module1_code.strip():
                stdout, err = evaluate_code_sandboxed(out.module1_code, timeout_sec=self.exec_timeout_sec)
                out.module2_stdout = stdout
                out.module2_error = "" if err is None else repr(err)
                out.module2_answer = extract_answer_from_exec_output(stdout) if err is None else ""
                if err is not None:
                    record_module_error(
                        self.error_jsonl_path,
                        pid,
                        "M2",
                        err,
                        extra={"hint": "sandbox_exec", "stdout_tail": stdout[-500:] if stdout else ""},
                    )
                    self.logger.warning(f"[pid={pid}] [M2] exec error: {out.module2_error}")
                else:
                    self.logger.debug(f"[pid={pid}] [M2] done (answer={out.module2_answer!r})")
            else:
                out.module2_stdout = ""
                out.module2_error = "module1_empty_code"
                out.module2_answer = ""
                record_module_error(self.error_jsonl_path, pid, "M2", out.module2_error, extra={"hint": "module1 produced empty code"})
                self.logger.warning(f"[pid={pid}] [M2] skipped because Module-1 returned empty code")
        except Exception as e:
            self.logger.exception(f"[pid={pid}] [M2] failed")
            record_module_error(self.error_jsonl_path, pid, "M2", e, extra={"hint": "sandbox_exec_exception"})
            out.module2_stdout = out.module2_stdout or ""
            out.module2_error = repr(e)
            out.module2_answer = ""

        # -----------------
        # Module-3: direct solve
        # -----------------
        try:
            m3_user = f"PROBLEM_JSON:\n{user_blob}\n\nReturn the final answer only."
            self.logger.debug(f"[pid={pid}] [M3] start")
            out.module3_answer = self.model.chat(
                system_prompt=DIRECT_SOLVE_SYSTEM,
                user_prompt=m3_user,
                decoded_image=decoded_image,
                temperature=0.0,
                max_tokens=self.direct_max_tokens,
            ).strip()
            self.logger.debug(f"[pid={pid}] [M3] done (answer={out.module3_answer!r})")
        except Exception as e:
            self.logger.exception(f"[pid={pid}] [M3] failed")
            record_module_error(self.error_jsonl_path, pid, "M3", e, extra={"hint": "direct_solve/chat"})
            out.module3_answer = ""

        # quick agreement check (optional; still run verifier for tie-break and formatting sanity)
        code_norm = normalize_answer_for_compare(out.module2_answer)
        direct_norm = normalize_answer_for_compare(out.module3_answer)
        agree_quick = (code_norm != "" and code_norm == direct_norm)

        # -----------------
        # Module-4: verifier
        # -----------------
        try:
            m4_user = (
                f"PROBLEM_JSON:\n{user_blob}\n\n"
                f"MODULE2:\n"
                f"- stdout: {out.module2_stdout}\n"
                f"- error: {out.module2_error}\n"
                f"- extracted_answer: {out.module2_answer}\n\n"
                f"MODULE3:\n"
                f"- answer: {out.module3_answer}\n\n"
                f"QuickCheck agree={str(agree_quick).lower()}.\n"
                "Decide final_answer."
            )
            self.logger.debug(f"[pid={pid}] [M4] start")
            out.module4_raw = self.model.chat(
                system_prompt=VERIFIER_SYSTEM,
                user_prompt=m4_user,
                decoded_image=None,
                temperature=0.0,
                max_tokens=self.verifier_max_tokens,
            ).strip()

            m4_json = extract_json_object(out.module4_raw) or {}
            out.module4_json = m4_json if isinstance(m4_json, dict) else {}

            final_answer = out.module4_json.get("final_answer", "").strip() if isinstance(out.module4_json, dict) else ""
            self.logger.debug(f"[pid={pid}] [M4] done (final={final_answer!r})")
        except Exception as e:
            self.logger.exception(f"[pid={pid}] [M4] failed")
            record_module_error(self.error_jsonl_path, pid, "M4", e, extra={"hint": "verifier/chat_or_parse"})
            out.module4_raw = out.module4_raw or ""
            out.module4_json = {}
            final_answer = ""

        # -----------------
        # Final fallback logic (if verifier fails/malformed)
        # -----------------
        if not final_answer:
            if agree_quick and out.module3_answer.strip():
                final_answer = out.module3_answer.strip()
            elif out.module3_answer.strip():
                final_answer = out.module3_answer.strip()
            else:
                final_answer = out.module2_answer.strip()

        out.final_answer = final_answer
        return out


# -----------------------------
# CLI / Main
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    # input
    parser.add_argument('--dataset_name', type=str, default='AI4Math/MathVista')
    parser.add_argument('--test_split_name', type=str, default='testmini')
    parser.add_argument('--data_dir', type=str, default='../data')
    parser.add_argument('--input_file', type=str, default='testmini.json')
    # output
    parser.add_argument('--output_dir', type=str, default='../results/bard')
    parser.add_argument('--output_file', type=str, default='output_bard.json')
    # logging
    parser.add_argument('--log_file', type=str, default='agent_debug.log', help='debug log file written under output_dir')
    parser.add_argument('--error_log_file', type=str, default='agent_errors.jsonl', help='structured module error log (jsonl) under output_dir')

    parser.add_argument('--max_num_problems', type=int, default=-1, help='The number of problems to run')
    parser.add_argument('--save_every', type=int, default=100, help='save every n problems')

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
    parser.add_argument('--caption_file', type=str, default='../data/texts/captions_bard.json')
    parser.add_argument('--ocr_file', type=str, default='../data/texts/ocrs_easyocr.json')
    parser.add_argument('--shot_type', type=str, default='solution', help='legacy arg (ignored)', choices=['solution', 'code'])
    parser.add_argument('--shot_num', type=int, default=0, help='number of shot examples')
    parser.add_argument('--use_caption', action='store_true', help='use caption data')
    parser.add_argument('--use_ocr', action='store_true', help='use ocr data')

    # agent settings
    parser.add_argument('--agent_mode', type=str, default='four_module', choices=['direct', 'four_module'],
                        help='direct: only Module-3; four_module: Modules 1-4 (default)')
    parser.add_argument('--exec_timeout_sec', type=int, default=3, help='timeout for Module-2 code execution (seconds)')
    parser.add_argument('--ti_max_tokens', type=int, default=900, help='max tokens for Module-1 (code generation)')
    parser.add_argument('--direct_max_tokens', type=int, default=256, help='max tokens for Module-3 (direct)')
    parser.add_argument('--verifier_max_tokens', type=int, default=256, help='max tokens for Module-4 (verifier)')

    # other settings
    parser.add_argument('--rerun', action='store_true', help='rerun for all problems')
    parser.add_argument('--debug', action='store_true', help='debug mode')

    args = parser.parse_args()
    return args


def main():
    logging.info("MathVista: Generating Responses - Start")
    args = parse_args()
    # attach file logging under output_dir so you can inspect failures after the run
    os.makedirs(args.output_dir, exist_ok=True)
    debug_log_path = setup_file_logging(args.output_dir, args.log_file)
    error_jsonl_path = os.path.join(args.output_dir, args.error_log_file)
    logging.info(f"Debug log: {debug_log_path}")
    logging.info(f"Module error log (jsonl): {error_jsonl_path}")


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
        llm = OpenAIChatModel(
            client=client,
            model=model_name,
            temperature=args.temperature,
            max_tokens=args.max_new_tokens,
        )

    logging.info("Model loaded.")

    agent = FourModuleAgent(
        model=llm,
        exec_timeout_sec=args.exec_timeout_sec,
        ti_max_tokens=args.ti_max_tokens,
        direct_max_tokens=args.direct_max_tokens,
        verifier_max_tokens=args.verifier_max_tokens,
        error_jsonl_path=error_jsonl_path,
    )

    full_pids = list(data.keys())

    os.makedirs(args.output_dir, exist_ok=True)
    output_file_path = os.path.join(args.output_dir, args.output_file)

    # load results
    if os.path.exists(output_file_path):
        logging.info("Results already exist.")
        logging.info(f"Reading {output_file_path}...")
        results = read_json(output_file_path)
    else:
        results = {}

    # skip valid answers unless rerun
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

        query_text = query_data[problem_id]

        logging.debug("--------------------------------------------------------------")
        logging.debug(f"Generating response for problem: {problem_id}...")
        try:
            if args.agent_mode == "direct":
                # Module-3 only
                payload = build_problem_payload(problem, query_text)
                user_blob = json.dumps(payload, ensure_ascii=False)
                try:
                    response = llm.chat(
                        system_prompt=DIRECT_SOLVE_SYSTEM,
                        user_prompt=f"PROBLEM_JSON:\n{user_blob}\n\nReturn the final answer only.",
                        decoded_image=problem_decoded_image,
                        temperature=0.0,
                        max_tokens=args.direct_max_tokens,
                    ).strip()
                except Exception as e:
                    logging.exception(f"[pid={problem_id}] [M3] direct solve failed")
                    record_module_error(error_jsonl_path, str(problem_id), "M3", e, extra={"hint": "direct_mode/chat"})
                    response = ""

                results[problem_id] = problem
                results[problem_id]['query'] = query_text
                results[problem_id]['response'] = response
                results[problem_id]['agent'] = {"mode": "direct"}
            else:
                out = agent.run(problem=problem, query_text=query_text, decoded_image=problem_decoded_image)

                results[problem_id] = problem
                results[problem_id]['query'] = query_text
                # keep 'response' for downstream scripts (answer extraction / scoring)
                results[problem_id]['response'] = out.final_answer
                results[problem_id]['agent'] = {
                    "mode": "four_module",
                    "module1_code": out.module1_code,
                    "module2_stdout": out.module2_stdout,
                    "module2_error": out.module2_error,
                    "module2_answer": out.module2_answer,
                    "module3_answer": out.module3_answer,
                    "module4_raw": out.module4_raw,
                    "module4_json": out.module4_json,
                }

                logging.debug(f"Final Answer: {out.final_answer}")

            logging.debug(f"Query: \n{query_text}")
            logging.debug(f"Response: \n{results[problem_id]['response']}")
        except Exception as e:
            logging.error(f"Error in extracting answer for {problem_id}")
            logging.error(e)
            results[problem_id] = problem
            results[problem_id]['query'] = query_text
            results[problem_id]['error'] = str(e)

        if (i % args.save_every == 0 and i > 0) or i == len(test_pids) - 1:
            try:
                save_json(results, output_file_path)
                logging.info(f"Saved results to {output_file_path}")
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
