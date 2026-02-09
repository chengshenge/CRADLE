import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

try:
    from PIL import Image
except Exception:
    Image = None


def ensure_image_png(task_dir: str, src_img_path: str) -> str:
    """VisualSketchpad expects task_dir/image.png to exist.

    Create it from src_img_path if missing.
    """
    os.makedirs(task_dir, exist_ok=True)
    dst = os.path.join(task_dir, "image.png")
    if os.path.exists(dst):
        return dst
    if Image is None:
        raise RuntimeError("Pillow is required to create image.png")
    Image.open(src_img_path).convert("RGB").save(dst)
    return dst


def _safe_text(x):
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return str(x)


def _normalize_prompt(text: str) -> str:
    """Make the prompt robust for downstream runners/LLMs.

    1) Normalize CRLF -> LF
    2) Convert literal '\\n' into real newlines '\n'
       (This commonly happens when upstream code accidentally double-escapes newlines.)
    """
    s = _safe_text(text)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # Turn literal backslash-n into real newlines (but keep other backslashes, e.g., LaTeX \\mathrm)
    s = s.replace("\\n", "\n")
    return s


def _content_to_text(content) -> str:
    # autogen multimodal content may be str or list[{"type":"text","text":...}, ...]
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict):
                if p.get("type") == "text" and "text" in p:
                    parts.append(_safe_text(p["text"]))
                elif "text" in p:
                    parts.append(_safe_text(p["text"]))
                elif "content" in p:
                    parts.append(_safe_text(p["content"]))
            else:
                parts.append(_safe_text(p))
        return "\n".join([t for t in parts if t.strip()])
    return _safe_text(content)


def _deep_find_last_text(obj) -> str:
    texts = []

    def visit(x):
        if isinstance(x, dict):
            if "content" in x:
                texts.append(_content_to_text(x["content"]))
            if "text" in x and isinstance(x["text"], str):
                texts.append(x["text"])
            for v in x.values():
                visit(v)
        elif isinstance(x, list):
            for y in x:
                visit(y)

    visit(obj)
    return texts[-1] if texts else ""


def _extract_last_assistant_text(messages_obj) -> str:
    """VisualSketchpad saves autogen chat_messages (usually list[dict]).
    Try robustly to extract the last assistant/planner text content.
    """
    if isinstance(messages_obj, dict) and "error" in messages_obj:
        return f"Response Error: {messages_obj['error']}"

    if isinstance(messages_obj, list):
        # Prefer the last msg with role=assistant OR name=planner
        for msg in reversed(messages_obj):
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "")
            name = msg.get("name", "")
            if role == "assistant" or name == "planner":
                content = msg.get("content", "")
                return _content_to_text(content).strip()

        # Fallback: last dict that has content
        for msg in reversed(messages_obj):
            if isinstance(msg, dict) and "content" in msg:
                return _content_to_text(msg["content"]).strip()

    # Fallback: deep search for last text/content
    return _deep_find_last_text(messages_obj).strip()


def build_mathvista_query_for_vsk(problem: dict) -> str:
    """Build a tool-friendly prompt for VisualSketchpad.

    MathVista provides two text fields:
    - question: sometimes short/incomplete (may omit unit / answer-format constraints)
    - hint: often includes the required answer format and unit, and may repeat the question

    Fix:
    - Prefer `hint` when present,
    - but always ensure the raw `question` is included,
    - and use REAL newlines ("\n"), not literal "\\n".
    """
    q = (problem.get("question") or "").strip()
    hint = (problem.get("hint") or "").strip()

    qtype = problem.get("question_type", "")
    choices = problem.get("choices", None)

    # Base prompt: use hint first because it usually contains format constraints.
    if hint:
        prompt = hint
        # If hint doesn't explicitly include the question, append it.
        if "Question:" not in prompt and q:
            prompt = f"{prompt}\nQuestion: {q}"
        elif q and q not in prompt:
            # Hint includes 'Question:' but might not include the exact question string.
            prompt = f"{prompt}\n{q}"
    else:
        prompt = q

    # Append choices for multi-choice
    if qtype == "multi_choice" and isinstance(choices, list) and len(choices) > 0:
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        choice_lines = []
        for i, c in enumerate(choices):
            tag = letters[i] if i < len(letters) else str(i)
            choice_lines.append(f"({tag}) {c}")
        prompt = (
            f"{prompt}\n\nChoices:\n"
            + "\n".join(choice_lines)
            + "\n\nPlease answer with the option letter (A/B/C/...) or the exact choice text."
        )
    else:
        prompt = f"{prompt}\n\nPlease give the final answer at the end."

    return prompt


class VisualSketchpadAgent:
    """A CRADLE-compatible wrapper:

    - signature: get_response(user_prompt: str, decoded_image: PIL.Image | None) -> str
    - internally calls VisualSketchpad run_agent in a subprocess

    IMPORTANT:
    Some VisualSketchpad runners/variants read different keys from request.json
    (e.g., "query" vs "question" vs "prompt").
    To maximize compatibility, we write multiple aliases that all contain the same prompt.
    We also normalize accidental literal '\\n' into real newlines.
    """

    def __init__(
        self,
        output_dir: str,
        api_key: str,
        model: str = "gpt-4o",
        temperature: float = 0.0,
        max_reply: int = 10,
        keep_traces: bool = True,
        task_type: str = "vision",
        vsk_root: str | None = None,
        som_address: str = "http://localhost:8080/",
        gd_address: str = "http://localhost:8081/",
        da_address: str = "http://localhost:8082/",
    ):
        self.output_dir = os.path.abspath(output_dir)
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_reply = max_reply
        self.keep_traces = keep_traces
        self.task_type = task_type

        cradle_root = Path(__file__).resolve().parents[1]
        self.vsk_root = (
            str((cradle_root / "third_party" / "VisualSketchpad").resolve())
            if vsk_root is None
            else vsk_root
        )
        self.runner = str((cradle_root / "scripts" / "run_visual_sketchpad_once.py").resolve())

        self.som_address = som_address
        self.gd_address = gd_address
        self.da_address = da_address

        self.trace_root = os.path.join(self.output_dir, "vsketchpad_traces")
        os.makedirs(self.trace_root, exist_ok=True)

    def get_response(self, user_prompt: str, decoded_image=None) -> str:
        run_id = f"mv_{uuid.uuid4().hex[:12]}"

        # Normalize prompt for better compatibility (handle accidental "\\n" etc.)
        prompt = _normalize_prompt(user_prompt)

        with tempfile.TemporaryDirectory(prefix=f"vsk_{run_id}_") as tmp:
            task_input = os.path.join(tmp, f"input_{run_id}")
            os.makedirs(task_input, exist_ok=True)

            # 1) Write request.json (use multiple key aliases for robustness)
            req = {
                # common keys
                "query": prompt,
                "question": prompt,
                "prompt": prompt,
                "text": prompt,
                "instruction": prompt,
                # for image paths
                "images": [],
            }

            # Also write plain-text copies (some variants may read these files)
            try:
                with open(os.path.join(task_input, "query.txt"), "w", encoding="utf-8") as f:
                    f.write(prompt)
                with open(os.path.join(task_input, "question.txt"), "w", encoding="utf-8") as f:
                    f.write(prompt)
                with open(os.path.join(task_input, "prompt.txt"), "w", encoding="utf-8") as f:
                    f.write(prompt)
            except Exception:
                # Non-fatal
                pass

            # 2) Write image.png (if any)
            if decoded_image is not None:
                img_path = os.path.join(task_input, "image.png")
                self._save_image(decoded_image, img_path)
                req["images"] = ["image.png"]

            with open(os.path.join(task_input, "request.json"), "w", encoding="utf-8") as f:
                json.dump(req, f, ensure_ascii=False, indent=2)

            # 3) Run VisualSketchpad in a subprocess
            cmd = [
                sys.executable,
                self.runner,
                "--vsk_root",
                self.vsk_root,
                "--task_input",
                task_input,
                "--output_dir",
                self.trace_root,
                "--task_type",
                self.task_type,
                "--api_key",
                self.api_key,
                "--model",
                self.model,
                "--temperature",
                str(self.temperature),
                "--max_reply",
                str(self.max_reply),
                "--som_address",
                self.som_address,
                "--gd_address",
                self.gd_address,
                "--da_address",
                self.da_address,
            ]

            try:
                p = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=True,
                    cwd=task_input,  # critical: relative paths resolve to task_input
                )
            except subprocess.CalledProcessError as e:
                err = (e.stderr or "")[-2000:]
                out = (e.stdout or "")[-2000:]
                return (
                    "Response Error: VisualSketchpad failed.\n"
                    f"STDERR:\n{err}\nSTDOUT:\n{out}"
                )

            # 4) Read output.json (in trace_root/<basename(task_input)>/output.json)
            out_dir = os.path.join(self.trace_root, os.path.basename(task_input))
            out_json = os.path.join(out_dir, "output.json")

            if not os.path.exists(out_json):
                # Include stdout/stderr for debugging
                stdout = (p.stdout or "")[-2000:]
                stderr = (p.stderr or "")[-2000:]
                return (
                    f"Response Error: output.json not found at {out_json}\n"
                    f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}"
                )

            with open(out_json, "r", encoding="utf-8") as f:
                messages_obj = json.load(f)

            text = _extract_last_assistant_text(messages_obj)
            text = re.sub(r"\bTERMINATE\b", "", text).strip()

            # 5) If not keeping traces, delete this run's output directory
            if not self.keep_traces:
                try:
                    shutil.rmtree(out_dir, ignore_errors=True)
                except Exception:
                    pass

            return text

    def _save_image(self, decoded_image, img_path: str):
        if Image is not None and isinstance(decoded_image, Image.Image):
            img = decoded_image
        elif Image is not None:
            # numpy array, path-like, etc.
            try:
                import numpy as np

                if isinstance(decoded_image, np.ndarray):
                    img = Image.fromarray(decoded_image)
                else:
                    img = Image.open(decoded_image)
            except Exception:
                raise ValueError("decoded_image is not a PIL.Image and cannot be converted.")
        else:
            raise RuntimeError("Pillow is required for VisualSketchpadAgent.")

        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(img_path)
