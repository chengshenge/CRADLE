#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Four-module agent wrapper for MathVista-style generation:
- Module 1: Task Inference (generate Python code)
- Module 2: Code Executor (sandbox execute generated code)
- Module 3: Direct Solver (normal 4o: image+text -> answer)
- Module 4: Verifier (compare M2 and M3, choose final)
Plus:
- agent_debug.log: readable module start/done/fail logs
- agent_errors.jsonl: structured per-error records (one JSON per line)
- agent_errors_pretty.log: human-readable multi-line error log with real newlines
"""

import argparse
import io
import json
import logging
import os
import re
import signal
import sys
import time
import traceback
from contextlib import redirect_stdout
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from datasets import load_dataset
from rich.logging import RichHandler
from tqdm import tqdm

# -----------------------------
# Utilities: json read/write
# -----------------------------
def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# -----------------------------
# Logging setup
# -----------------------------
def setup_console_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%H:%M:%S]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )

def setup_file_logging(output_dir: str, log_file: str) -> str:
    """
    Attach a FileHandler to root logger. File logs contain DEBUG+.
    Console stays INFO via RichHandler.
    """
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, log_file)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="[%H:%M:%S]",
    )
    fh.setFormatter(fmt)
    root.addHandler(fh)

    return log_path

# -----------------------------
# Error log writers
# -----------------------------
def _append_jsonl(path: str, record: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def _append_pretty_error_log(path: str, record: Dict[str, Any]) -> None:
    """
    Write a human-readable multi-line record with REAL newlines.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = []
    lines.append("=" * 80)
    lines.append(f"ts: {record.get('ts')}")
    lines.append(f"pid: {record.get('pid')}")
    lines.append(f"stage: {record.get('stage')}")
    lines.append(f"error_type: {record.get('error_type')}")
    lines.append(f"error: {record.get('error')}")
    extra = record.get("extra")
    if extra:
        try:
            lines.append("extra:")
            lines.append(json.dumps(extra, ensure_ascii=False, indent=2))
        except Exception:
            lines.append(f"extra: {extra!r}")
    tb = record.get("traceback")
    if tb:
        lines.append("traceback:")
        lines.append(tb)
    lines.append("")  # trailing newline
    blob = "\n".join(lines) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(blob)

def record_module_error(
    error_jsonl_path: str,
    error_pretty_path: str,
    pid: str,
    stage: str,
    exc: BaseException,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    rec = {
        "ts": time.time(),
        "pid": str(pid),
        "stage": stage,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "extra": extra or {},
        "traceback": traceback.format_exc(),
    }
    _append_jsonl(error_jsonl_path, rec)
    _append_pretty_error_log(error_pretty_path, rec)

# -----------------------------
# Prompt templates
# -----------------------------
TASK_INFERENCE_SYSTEM = """You are Module-1: Task Inference for MathVista-style problems.

You will receive a PROBLEM_JSON that includes:
- question (text)
- choices (optional list of strings)
- unit (optional)
- precision (optional number of decimal places)
- other metadata

Your job:
1) Solve the problem by WRITING PYTHON CODE (no markdown).
2) The code must compute the final answer deterministically.
3) The code MUST end by printing the final answer on a single line.

Hard rules:
- Output ONLY valid Python code. No explanations.
- Do NOT use import statements. (Imports are blocked in the sandbox.)
- Use only basic arithmetic and built-ins.
- If the problem is multiple choice, you MUST print the full choice string exactly as it appears in choices.
- If precision is provided (e.g., 1.0), format numeric output with that many decimal places.

Output format:
print(FINAL_ANSWER)
"""

DIRECT_SOLVE_SYSTEM = """You are Module-3: Direct Solver.
You will receive the question text plus an image (if provided).

Return ONLY the final answer in one line.
- If multiple choice, return the full choice string (not A/B/C/D).
- If numeric, keep appropriate units if the problem includes units, and respect required precision if stated.
No extra text.
"""

VERIFIER_SYSTEM = """You are Module-4: Verifier.
Input contains:
- PROBLEM_JSON (original)
- MODULE2 outputs (stdout, error, extracted_answer)
- MODULE3 answer

Decide final_answer:
- If MODULE2 and MODULE3 agree (after trivial normalization), choose that.
- If they disagree, choose the more reasonable one:
  Prefer an answer that:
  1) is non-empty
  2) matches one of the choices (if choices exist)
  3) respects precision/unit requirements
  4) comes from a module without execution/error flags

Return STRICT JSON only:
{"final_answer":"...","agree":true/false,"chosen":"code|direct","notes":"..."}
"""

# -----------------------------
# Small helpers: normalization / JSON extraction
# -----------------------------
def normalize_answer_for_compare(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = s.replace("°", " degrees")
    s = re.sub(r"\s+", " ", s)
    s = s.strip()
    return s

def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Try to extract a JSON object from a text blob.
    """
    if not text:
        return None
    text = text.strip()
    # If it's already valid JSON
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Try to find the first {...} block
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None

# -----------------------------
# OpenAI client wrapper (simple)
# -----------------------------
class SimpleOpenAIChat:
    def __init__(self, model: str, api_key: str = ""):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            raise ValueError("OpenAI API key is empty. Provide --key or set OPENAI_API_KEY.")

        try:
            from openai import OpenAI
        except Exception as e:
            raise ImportError("openai package not found. pip install openai") from e

        from openai import OpenAI  # type: ignore
        self.client = OpenAI(api_key=self.api_key)

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        decoded_image: Optional[Any] = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        """
        If decoded_image is provided, send as multimodal content.
        `decoded_image` here should be a PIL.Image or bytes; to keep this file standalone,
        we accept None by default.
        """
        # Build messages
        messages = [{"role": "system", "content": system_prompt}]

        if decoded_image is None:
            messages.append({"role": "user", "content": user_prompt})
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""

        # multimodal: image
        try:
            import base64
            from io import BytesIO
            from PIL import Image
        except Exception:
            # fallback to text-only if PIL not present
            messages.append({"role": "user", "content": user_prompt})
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""

        if hasattr(decoded_image, "save"):
            buf = BytesIO()
            decoded_image.save(buf, format="PNG")
            img_bytes = buf.getvalue()
        elif isinstance(decoded_image, (bytes, bytearray)):
            img_bytes = bytes(decoded_image)
        else:
            # unknown type
            messages.append({"role": "user", "content": user_prompt})
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""

        b64 = base64.b64encode(img_bytes).decode("utf-8")
        content = [
            {"type": "text", "text": user_prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]
        messages.append({"role": "user", "content": content})
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

# -----------------------------
# Module-2 sandbox executor
# -----------------------------
class TimeoutError(Exception):
    pass

def _timeout_handler(signum, frame):
    raise TimeoutError("Sandbox execution timed out")

def safe_exec(code: str, timeout_sec: int = 3) -> Tuple[str, Optional[str]]:
    """
    Execute python code in a restricted environment. Return (stdout, err_str).
    - imports are blocked by not providing __import__
    """
    if not isinstance(code, str):
        return "", "code is not a string"

    safe_builtins = {
        "abs": abs,
        "min": min,
        "max": max,
        "sum": sum,
        "len": len,
        "range": range,
        "round": round,
        "int": int,
        "float": float,
        "str": str,
        "print": print,
        "enumerate": enumerate,
        "zip": zip,
        "map": map,
        "list": list,
        "dict": dict,
        "set": set,
        "tuple": tuple,
    }

    env = {"__builtins__": safe_builtins}
    stdout_buf = io.StringIO()

    old = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(max(1, int(timeout_sec)))

    try:
        with redirect_stdout(stdout_buf):
            exec(code, env, env)
        out = stdout_buf.getvalue()
        signal.alarm(0)
        return out, None
    except Exception as e:
        signal.alarm(0)
        return stdout_buf.getvalue(), f"{type(e).__name__}: {e}"
    finally:
        signal.signal(signal.SIGALRM, old)

def extract_answer_from_stdout(stdout: str) -> str:
    """
    Heuristic: prefer line starting with 'FINAL_ANSWER:' else last non-empty line.
    """
    if not stdout:
        return ""
    lines = [ln.strip() for ln in stdout.splitlines() if ln.strip()]
    if not lines:
        return ""
    for ln in reversed(lines):
        if ln.lower().startswith("final_answer"):
            # formats: FINAL_ANSWER: xxx
            parts = ln.split(":", 1)
            if len(parts) == 2:
                return parts[1].strip()
    return lines[-1].strip()

# -----------------------------
# Agent dataclasses
# -----------------------------
@dataclass
class AgentOutputs:
    pid: str
    module1_code: str = ""
    module2_stdout: str = ""
    module2_error: str = ""
    module2_answer: str = ""
    module3_answer: str = ""
    module4_raw: str = ""
    module4_json: Dict[str, Any] = None
    final_answer: str = ""

# -----------------------------
# Four-module agent
# -----------------------------
class FourModuleAgent:
    def __init__(
        self,
        model: SimpleOpenAIChat,
        error_jsonl_path: str,
        error_pretty_path: str,
        logger: Optional[logging.Logger] = None,
        exec_timeout_sec: int = 3,
        ti_max_tokens: int = 900,
        direct_max_tokens: int = 256,
        verifier_max_tokens: int = 256,
    ):
        self.model = model
        self.error_jsonl_path = error_jsonl_path
        self.error_pretty_path = error_pretty_path
        self.exec_timeout_sec = exec_timeout_sec
        self.ti_max_tokens = ti_max_tokens
        self.direct_max_tokens = direct_max_tokens
        self.verifier_max_tokens = verifier_max_tokens
        self.logger = logger or logging.getLogger("FourModuleAgent")

    def run_one(self, pid: str, problem: Dict[str, Any], decoded_image: Optional[Any] = None) -> AgentOutputs:
        out = AgentOutputs(pid=str(pid), module4_json={})

        user_blob = json.dumps(problem, ensure_ascii=False)
        # -----------------
        # Module-1: task inference (generate code)
        # -----------------
        try:
            m1_user = f"PROBLEM_JSON:\n{user_blob}\n\nWrite Python code now."
            self.logger.debug(f"[pid={pid}] [M1] start")
            out.module1_code = self.model.chat(
                system_prompt=TASK_INFERENCE_SYSTEM,
                user_prompt=m1_user,
                decoded_image=decoded_image,
                temperature=0.0,
                max_tokens=self.ti_max_tokens,
            ).strip()
            # Some models may wrap in ```python ...```
            out.module1_code = strip_code_fences(out.module1_code)
            self.logger.debug(f"[pid={pid}] [M1] done (len={len(out.module1_code)})")
        except Exception as e:
            self.logger.exception(f"[pid={pid}] [M1] failed")
            record_module_error(self.error_jsonl_path, self.error_pretty_path, pid, "M1", e, extra={"hint": "task_inference/chat"})
            out.module1_code = ""

        # -----------------
        # Module-2: execute code
        # -----------------
        try:
            self.logger.debug(f"[pid={pid}] [M2] start")
            if out.module1_code.strip():
                stdout, err = safe_exec(out.module1_code, timeout_sec=self.exec_timeout_sec)
            else:
                stdout, err = "", "empty code from M1"
            out.module2_stdout = stdout or ""
            out.module2_error = err or ""
            out.module2_answer = extract_answer_from_stdout(out.module2_stdout)

            if out.module2_error:
                # record M2 errors even if no exception thrown
                record_module_error(
                    self.error_jsonl_path,
                    self.error_pretty_path,
                    pid,
                    "M2",
                    RuntimeError(out.module2_error),
                    extra={
                        "hint": "sandbox_exec",
                        "stdout_tail": (out.module2_stdout[-500:] if out.module2_stdout else ""),
                    },
                )
            self.logger.debug(f"[pid={pid}] [M2] done (ans={out.module2_answer!r}, err={bool(out.module2_error)})")
        except Exception as e:
            self.logger.exception(f"[pid={pid}] [M2] failed")
            record_module_error(self.error_jsonl_path, self.error_pretty_path, pid, "M2", e, extra={"hint": "sandbox_exec/exception"})
            out.module2_stdout = out.module2_stdout or ""
            out.module2_error = out.module2_error or str(e)
            out.module2_answer = out.module2_answer or ""

        # -----------------
        # Module-3: direct solve
        # -----------------
        try:
            m3_user = problem.get("question", "")
            # add choices into prompt for clarity
            if problem.get("choices"):
                m3_user += "\n\nChoices:\n" + "\n".join([str(c) for c in problem["choices"]])
            self.logger.debug(f"[pid={pid}] [M3] start")
            out.module3_answer = self.model.chat(
                system_prompt=DIRECT_SOLVE_SYSTEM,
                user_prompt=m3_user,
                decoded_image=decoded_image,
                temperature=0.0,
                max_tokens=self.direct_max_tokens,
            ).strip()
            out.module3_answer = out.module3_answer.strip()
            self.logger.debug(f"[pid={pid}] [M3] done (ans={out.module3_answer!r})")
        except Exception as e:
            self.logger.exception(f"[pid={pid}] [M3] failed")
            record_module_error(self.error_jsonl_path, self.error_pretty_path, pid, "M3", e, extra={"hint": "direct_solve/chat"})
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
            record_module_error(self.error_jsonl_path, self.error_pretty_path, pid, "M4", e, extra={"hint": "verifier/chat_or_parse"})
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

def strip_code_fences(s: str) -> str:
    if not s:
        return s
    s = s.strip()
    # remove ```python ... ```
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_+-]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()

# -----------------------------
# Main pipeline (MathVista-like)
# -----------------------------
def create_query(problem: Dict[str, Any], caption_data: Optional[Dict[str, Any]], ocr_data: Optional[Dict[str, Any]], use_caption: bool, use_ocr: bool) -> Dict[str, Any]:
    """
    Create a PROBLEM_JSON bundle for the agent. Keep it simple.
    """
    q = {
        "pid": problem.get("pid"),
        "question": problem.get("question", ""),
        "choices": problem.get("choices"),
        "unit": problem.get("unit"),
        "precision": problem.get("precision"),
        "answer_type": problem.get("answer_type"),
        "question_type": problem.get("question_type"),
        "metadata": problem.get("metadata", {}),
        "image": problem.get("image"),
    }
    if use_caption and caption_data is not None:
        q["caption"] = caption_data.get(str(problem.get("pid")), "")
    if use_ocr and ocr_data is not None:
        q["ocr"] = ocr_data.get(str(problem.get("pid")), "")
    return q

def load_image_if_available(img_path: str) -> Optional[Any]:
    if not img_path:
        return None
    if not os.path.exists(img_path):
        return None
    try:
        from PIL import Image
        return Image.open(img_path).convert("RGB")
    except Exception:
        return None

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
    parser.add_argument('--pretty_error_log_file', type=str, default='agent_errors_pretty.log',
                        help='human-readable multi-line module error log (pretty) under output_dir')

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
    pretty_name = getattr(args, 'pretty_error_log_file', 'agent_errors_pretty.log')
    error_pretty_path = os.path.join(args.output_dir, pretty_name)
    logging.info(f"Debug log: {debug_log_path}")
    logging.info(f"Module error log (jsonl): {error_jsonl_path}")
    logging.info(f"Module error log (pretty): {error_pretty_path}")

    # Ensure DEBUG logs go to the file (FileHandler is DEBUG), but keep console readable
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for h in root.handlers:
        try:
            if isinstance(h, RichHandler):
                h.setLevel(logging.INFO)
        except Exception:
            pass

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
        caption_data = None
        ocr_data = None
        if args.use_caption and os.path.exists(args.caption_file):
            caption_data = read_json(args.caption_file)
        if args.use_ocr and os.path.exists(args.ocr_file):
            ocr_data = read_json(args.ocr_file)

        query_data = {}
        for pid, item in data.items():
            query_data[str(pid)] = create_query(item, caption_data, ocr_data, args.use_caption, args.use_ocr)

    # init model + agent
    model = SimpleOpenAIChat(model=args.model, api_key=(args.key or ""))
    agent_logger = logging.getLogger("agent")
    agent = FourModuleAgent(
        model=model,
        error_jsonl_path=error_jsonl_path,
        error_pretty_path=error_pretty_path,
        logger=agent_logger,
        exec_timeout_sec=args.exec_timeout_sec,
        ti_max_tokens=args.ti_max_tokens,
        direct_max_tokens=args.direct_max_tokens,
        verifier_max_tokens=args.verifier_max_tokens,
    )

    # load output if exists
    out_path = os.path.join(args.output_dir, args.output_file)
    if os.path.exists(out_path) and not args.rerun:
        logging.info(f"Loading existing output: {out_path}")
        results = read_json(out_path)
    else:
        results = {}

    # run
    pids = list(query_data.keys())
    if args.max_num_problems and args.max_num_problems > 0:
        pids = pids[: args.max_num_problems]

    for i, pid in enumerate(tqdm(pids)):
        if (not args.rerun) and (pid in results):
            continue

        problem = query_data[pid]
        # resolve image absolute path
        img_rel = problem.get("image")
        decoded_image = None
        if img_rel:
            # in MathVista HF dataset, image path often like "images/xxx.jpg"
            img_path = img_rel
            if args.data_dir and not os.path.isabs(img_path):
                img_path = os.path.join(args.data_dir, img_path)
            decoded_image = load_image_if_available(img_path)

        try:
            if args.agent_mode == "direct":
                # Module-3 only
                m3_user = problem.get("question", "")
                if problem.get("choices"):
                    m3_user += "\n\nChoices:\n" + "\n".join([str(c) for c in problem["choices"]])
                ans = model.chat(
                    system_prompt=DIRECT_SOLVE_SYSTEM,
                    user_prompt=m3_user,
                    decoded_image=decoded_image,
                    temperature=0.0,
                    max_tokens=args.direct_max_tokens,
                ).strip()
                results[pid] = {
                    "pid": pid,
                    "question": problem.get("question", ""),
                    "response": ans,
                    "agent": {
                        "mode": "direct",
                        "module3_answer": ans,
                    },
                }
            else:
                # four-module
                outs = agent.run_one(pid=pid, problem=problem, decoded_image=decoded_image)
                results[pid] = {
                    "pid": pid,
                    "question": problem.get("question", ""),
                    "response": outs.final_answer,
                    "agent": {
                        "mode": "four_module",
                        "module1_code": outs.module1_code,
                        "module2_stdout": outs.module2_stdout,
                        "module2_error": outs.module2_error,
                        "module2_answer": outs.module2_answer,
                        "module3_answer": outs.module3_answer,
                        "module4_raw": outs.module4_raw,
                        "module4_json": outs.module4_json,
                        "final_answer": outs.final_answer,
                    },
                }

        except Exception as e:
            logging.exception(f"[pid={pid}] main-loop failed")
            record_module_error(error_jsonl_path, error_pretty_path, pid, "MAIN", e, extra={"hint": "outer_loop"})
            # still save something
            results[pid] = {
                "pid": pid,
                "question": problem.get("question", ""),
                "response": "",
                "agent": {
                    "mode": args.agent_mode,
                    "error": f"{type(e).__name__}: {e}",
                },
            }

        # periodic save
        if args.save_every > 0 and (i + 1) % args.save_every == 0:
            save_json(out_path, results)

    # final save
    save_json(out_path, results)
    logging.info(f"Saved results to: {out_path}")


if __name__ == "__main__":
    setup_console_logging()
    main()
