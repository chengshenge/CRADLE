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


def _safe_text(x):
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return str(x)


def _extract_last_assistant_text(messages_obj) -> str:
    """
    VisualSketchpad 保存的是 autogen 的 chat_messages（通常是 list[dict]）。
    这里尽量鲁棒地抽取最后一条 assistant/planner 的文本 content。
    """
    if isinstance(messages_obj, dict) and "error" in messages_obj:
        return f"Response Error: {messages_obj['error']}"

    if isinstance(messages_obj, list):
        # 优先找 role=assistant 或 name=planner 的最后一条
        for msg in reversed(messages_obj):
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "")
            name = msg.get("name", "")
            if role == "assistant" or name == "planner":
                content = msg.get("content", "")
                return _content_to_text(content).strip()

        # fallback：拿最后一个 dict 的 content
        for msg in reversed(messages_obj):
            if isinstance(msg, dict) and "content" in msg:
                return _content_to_text(msg["content"]).strip()

    # fallback：递归搜 text/content
    return _deep_find_last_text(messages_obj).strip()


def _content_to_text(content) -> str:
    # autogen multimodal content 可能是 str 或 list[{"type":"text","text":...}, ...]
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


def build_mathvista_query_for_vsk(problem: dict) -> str:
    q = problem.get("question", "")
    qtype = problem.get("question_type", "")
    choices = problem.get("choices", None)

    if qtype == "multi_choice" and isinstance(choices, list) and len(choices) > 0:
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        choice_lines = []
        for i, c in enumerate(choices):
            tag = letters[i] if i < len(letters) else str(i)
            choice_lines.append(f"({tag}) {c}")
        q = (
            f"{q}\n\nChoices:\n" +
            "\n".join(choice_lines) +
            "\n\nPlease answer with the option letter (A/B/C/...) or the exact choice text."
        )
    else:
        q = f"{q}\n\nPlease answer concisely."

    return q


class VisualSketchpadAgent:
    """
    A CRADLE-compatible wrapper:
    - signature: get_response(user_prompt: str, decoded_image: PIL.Image | None) -> str
    - internally calls VisualSketchpad run_agent in a subprocess
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
        self.output_dir = output_dir
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_reply = max_reply
        self.keep_traces = keep_traces
        self.task_type = task_type

        cradle_root = Path(__file__).resolve().parents[1]
        self.vsk_root = str((cradle_root / "third_party" / "VisualSketchpad").resolve()) if vsk_root is None else vsk_root
        self.runner = str((cradle_root / "scripts" / "run_visual_sketchpad_once.py").resolve())

        self.som_address = som_address
        self.gd_address = gd_address
        self.da_address = da_address

        self.trace_root = os.path.join(self.output_dir, "vsketchpad_traces")
        os.makedirs(self.trace_root, exist_ok=True)

    def get_response(self, user_prompt: str, decoded_image=None) -> str:
        run_id = f"mv_{uuid.uuid4().hex[:12]}"

        with tempfile.TemporaryDirectory(prefix=f"vsk_{run_id}_") as tmp:
            task_input = os.path.join(tmp, f"input_{run_id}")
            os.makedirs(task_input, exist_ok=True)

            # 1) 写 request.json
            req = {
                "query": user_prompt,
                "images": []
            }

            # 2) 写 image.png（如果有图）
            if decoded_image is not None:
                img_path = os.path.join(task_input, "image.png")
                self._save_image(decoded_image, img_path)
                req["images"] = ["image.png"]

            with open(os.path.join(task_input, "request.json"), "w", encoding="utf-8") as f:
                json.dump(req, f, ensure_ascii=False, indent=2)

            # 3) 子进程跑 VisualSketchpad
            cmd = [
                sys.executable, self.runner,
                "--vsk_root", self.vsk_root,
                "--task_input", task_input,
                "--output_dir", self.trace_root,
                "--task_type", self.task_type,
                "--api_key", self.api_key,
                "--model", self.model,
                "--temperature", str(self.temperature),
                "--max_reply", str(self.max_reply),
                "--som_address", self.som_address,
                "--gd_address", self.gd_address,
                "--da_address", self.da_address,
            ]

            try:
                p = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=True,
                )
            except subprocess.CalledProcessError as e:
                err = (e.stderr or "")[-2000:]
                out = (e.stdout or "")[-2000:]
                return f"Response Error: VisualSketchpad failed.\nSTDERR:\n{err}\nSTDOUT:\n{out}"

            # 4) 读取 output.json（在 trace_root/<basename(task_input)>/output.json）
            out_dir = os.path.join(self.trace_root, os.path.basename(task_input))
            out_json = os.path.join(out_dir, "output.json")

            if not os.path.exists(out_json):
                # 把 stdout/stderr 也带上方便 debug
                return f"Response Error: output.json not found at {out_json}"

            with open(out_json, "r", encoding="utf-8") as f:
                messages_obj = json.load(f)

            text = _extract_last_assistant_text(messages_obj)
            text = re.sub(r"\bTERMINATE\b", "", text).strip()

            # 5) 如果不保留 trace，就删掉这一题的输出目录
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
            # numpy array etc
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
