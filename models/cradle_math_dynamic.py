import ast
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# Utilities
# ============================================================

def _now_ts() -> float:
    return time.time()


def _j(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _strip_markdown_fences(s: str) -> str:
    """Remove common markdown code fences (``` / ```json).

    This is a deterministic sanitation step (no model re-ask).
    It supports outputs like:
      ```json\n{...}\n```
    and also cases where the closing fence is missing.
    """
    if not isinstance(s, str):
        return s
    t = s.strip()

    # Prefer extracting the first fenced block if it exists.
    m = re.search(r"```(?:json)?\s*(.*?)```", t, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # If only an opening fence exists, drop the first fence line.
    if t.startswith("```"):
        parts = t.splitlines()
        if len(parts) >= 2:
            t = "\n".join(parts[1:]).strip()
        else:
            t = ""

    # If there's a trailing fence without a matched opener, strip it.
    if t.endswith("```"):
        t = t[:-3].strip()

    return t


def _extract_first_balanced_json_object(s: str) -> Optional[str]:
    """Return the first balanced {...} object substring, or None."""
    if not isinstance(s, str):
        return None
    start = s.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[start : i + 1]
    return None


def _escape_newlines_inside_strings(s: str) -> str:
    """JSON requires control characters in strings to be escaped.

    LLMs often insert real newlines inside quoted strings for readability, which
    makes the JSON invalid. This function deterministically escapes those.
    """
    if not isinstance(s, str):
        return s
    out = []
    in_str = False
    esc = False
    for ch in s:
        if in_str:
            if esc:
                out.append(ch)
                esc = False
                continue
            if ch == "\\":
                out.append(ch)
                esc = True
                continue
            if ch == '"':
                out.append(ch)
                in_str = False
                continue
            if ch == "\n":
                out.append("\\n")
                continue
            if ch == "\r":
                out.append("\\r")
                continue
            if ch == "\t":
                out.append("\\t")
                continue
            # Other control chars (<0x20) are illegal in JSON strings.
            if ord(ch) < 0x20:
                out.append(" ")
                continue
            out.append(ch)
        else:
            if ch == '"':
                out.append(ch)
                in_str = True
            else:
                out.append(ch)
    return "".join(out)


def _extract_json_obj(text: str) -> Dict[str, Any]:
    """Parse a JSON object from model output with light, deterministic sanitation.

    This is CRADLE-aligned in spirit: we do NOT ask the model to "repair" its own
    output by feeding the previous output back. We only strip common wrappers
    (markdown fences) and sanitize illegal newline characters inside JSON strings.
    """
    if not isinstance(text, str):
        raise ValueError("Model output is not a string.")

    s = _strip_markdown_fences(text)

    # First try direct parse
    try:
        return json.loads(s)
    except Exception:
        pass

    # Try extracting the first balanced JSON object
    candidate = _extract_first_balanced_json_object(s)
    if candidate is None:
        raise ValueError(f"Cannot find JSON object in: {s[:200]}...")

    # Sanitize newlines inside strings (common LLM formatting artifact)
    candidate2 = _escape_newlines_inside_strings(candidate)

    return json.loads(candidate2)


# ============================================================
# Observation (CRADLE alignment: single source of truth)
# ============================================================


@dataclass
class Observation:
    """Single source of truth passed to every LLM call."""

    text: str
    image: Any
    meta: Dict[str, Any]


# ============================================================
# Security checks for generated skill code
# ============================================================
DENY_PATTERNS = [
    r"\bimport\b",
    r"\bopen\s*\(",
    r"\bos\.",
    r"\bsys\.",
    r"\bsubprocess\b",
    r"\bsocket\b",
    r"\bshutil\b",
    r"\bpathlib\b",
    r"__",
    r"\beval\s*\(",
    r"\bexec\s*\(",
    r"\bglobals\s*\(",
    r"\blocals\s*\(",
]


ALLOWED_AST_NODES = {
    # module/function structure
    ast.Module,
    ast.FunctionDef,
    ast.arguments,
    ast.arg,
    ast.Return,
    ast.Expr,
    ast.Assign,
    ast.AnnAssign,
    # constants/ops
    ast.Constant,
    ast.Name,
    ast.Load,
    ast.Store,
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Compare,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.USub,
    ast.UAdd,
    ast.And,
    ast.Or,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    # containers
    ast.List,
    ast.Tuple,
    ast.Dict,
    ast.Set,
    ast.Subscript,
    ast.Slice,
    # control
    ast.If,
    ast.For,
    ast.While,
    ast.Break,
    ast.Continue,
    ast.Pass,
    # calls
    ast.Call,
    ast.keyword,
    # f-strings
    ast.JoinedStr,
    ast.FormattedValue,
}


def check_code_static_safety(code: str) -> Optional[str]:
    if not isinstance(code, str) or not code.strip():
        return "Empty code."
    for pat in DENY_PATTERNS:
        if re.search(pat, code):
            return f"Blocked by deny pattern: {pat}"

    try:
        tree = ast.parse(code)
    except Exception as e:
        return f"AST parse error: {e}"

    for node in ast.walk(tree):
        if type(node) not in ALLOWED_AST_NODES:
            return f"Disallowed AST node: {type(node).__name__}"

        # Forbid attribute calls (Attribute node is not allowed)
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in ("__import__",):
                    return "Disallowed call: __import__"
            else:
                return "Disallowed call target (non-Name)."

    # Ensure only function defs at top-level (optional but useful)
    for stmt in tree.body:
        if not isinstance(stmt, (ast.FunctionDef, ast.Expr)):
            return f"Top-level statement not allowed: {type(stmt).__name__}"
        if isinstance(stmt, ast.Expr) and not isinstance(stmt.value, ast.Constant):
            return "Only docstring allowed as top-level Expr."

    return None


# ============================================================
# Safe execution context for python tool steps + skills
# ============================================================

def make_safe_context() -> Dict[str, Any]:
    import math

    safe_builtins = {
        "abs": abs,
        "min": min,
        "max": max,
        "sum": sum,
        "len": len,
        "round": round,
        "range": range,
        "int": int,
        "float": float,
        "str": str,
        "print": print,
    }
    ctx = {
        "__builtins__": safe_builtins,
        "math": math,
    }
    try:
        import sympy as sp  # type: ignore

        ctx["sp"] = sp
    except Exception:
        pass
    return ctx


def safe_exec(code: str, g: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    for pat in DENY_PATTERNS:
        if re.search(pat, code):
            return "", f"Blocked unsafe pattern: {pat}"

    import io

    old_stdout = sys.stdout  # type: ignore
    buf = io.StringIO()
    sys.stdout = buf  # type: ignore
    err = None
    try:
        exec(code, g, g)
    except Exception as e:
        err = str(e)
    sys.stdout = old_stdout  # type: ignore
    return buf.getvalue().strip(), err


# (sys is used in safe_exec; keep import local to avoid global denylist concerns)
import sys  # noqa: E402


# ============================================================
# Skill Manager
# ============================================================


@dataclass
class SkillRecord:
    name: str
    code: str
    signature: str
    created_at: float
    updated_at: float
    tags: List[str]


class SkillManager:
    def __init__(
        self,
        base_dir: str,
        subdir: str = "skills",
        freeze: bool = False,
        reset: bool = False,
        max_new_skills: int = 3,
        debug: bool = False,
    ):
        self.base_dir = base_dir
        self.subdir = subdir
        self.freeze = freeze
        self.reset = reset
        self.max_new_skills = max_new_skills
        self.debug = debug

        self.skills_dir = os.path.join(base_dir, subdir)
        os.makedirs(self.skills_dir, exist_ok=True)

        self.registry_path = os.path.join(self.skills_dir, "registry.jsonl")
        self._loaded: Dict[str, SkillRecord] = {}

        self.ctx = make_safe_context()

        if self.reset:
            self._reset_registry()

        self.load_registry()

    def _reset_registry(self):
        if os.path.exists(self.registry_path):
            os.remove(self.registry_path)
        self._loaded = {}
        self.ctx = make_safe_context()

    def load_registry(self):
        if not os.path.exists(self.registry_path):
            return
        with open(self.registry_path, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    obj = json.loads(ln)
                except Exception:
                    continue
                rec = SkillRecord(
                    name=obj["name"],
                    code=obj["code"],
                    signature=obj.get("signature", ""),
                    created_at=obj.get("created_at", _now_ts()),
                    updated_at=obj.get("updated_at", _now_ts()),
                    tags=obj.get("tags", []),
                )
                err = check_code_static_safety(rec.code)
                if err is None:
                    _, run_err = safe_exec(rec.code, self.ctx)
                    if run_err is None:
                        self._loaded[rec.name] = rec

    def catalog(self) -> List[Dict[str, Any]]:
        cat = []
        for name, rec in self._loaded.items():
            cat.append(
                {
                    "name": name,
                    "signature": rec.signature,
                    "tags": rec.tags,
                    "desc": "learned skill",
                }
            )
        cat.extend(
            [
                {
                    "name": "calc_numeric",
                    "signature": "(expr:str)->float",
                    "tags": ["builtin"],
                    "desc": "Compute numeric expression in python.",
                },
                {
                    "name": "sympy_solve",
                    "signature": "(eqs, vars)->solution",
                    "tags": ["builtin"],
                    "desc": "Solve equations using sp in python if available.",
                },
                {
                    "name": "format_final",
                    "signature": "(ans)->str",
                    "tags": ["builtin"],
                    "desc": "Return exactly one line: Answer: ...",
                },
            ]
        )
        return cat

    def add_or_update_skill(
        self,
        name: str,
        code: str,
        signature: str,
        tags: Optional[List[str]] = None,
        tests: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[bool, str]:
        if self.freeze:
            return False, "Skills are frozen."

        static_err = check_code_static_safety(code)
        if static_err:
            return False, static_err

        temp_ctx = dict(self.ctx)
        _, err = safe_exec(code, temp_ctx)
        if err:
            return False, f"Skill code exec error: {err}"

        if name not in temp_ctx or not callable(temp_ctx[name]):
            return False, "Defined function not found or not callable."

        if tests:
            for t in tests:
                tcode = t.get("code", "")
                expect = t.get("expect_contains", "")
                tout, terr = safe_exec(tcode, temp_ctx)
                if terr:
                    return False, f"Test error: {terr}"
                if expect and expect not in str(tout):
                    return False, f"Test failed: expected output contains '{expect}', got '{tout}'"

        _, err2 = safe_exec(code, self.ctx)
        if err2:
            return False, f"Commit exec error: {err2}"

        now = _now_ts()
        rec = SkillRecord(
            name=name,
            code=code,
            signature=signature,
            created_at=self._loaded[name].created_at if name in self._loaded else now,
            updated_at=now,
            tags=tags or [],
        )
        self._loaded[name] = rec

        with open(self.registry_path, "a", encoding="utf-8") as f:
            f.write(
                _j(
                    {
                        "name": rec.name,
                        "code": rec.code,
                        "signature": rec.signature,
                        "created_at": rec.created_at,
                        "updated_at": rec.updated_at,
                        "tags": rec.tags,
                    }
                )
                + "\n"
            )

        return True, "ok"


# ============================================================
# Dynamic CRADLE-style Agent (CRADLE-aligned image injection)
# ============================================================

@dataclass
class CradleMathDynamicAgent:
    backbone: Any
    output_dir: str
    skills_subdir: str = "skills"
    max_steps: int = 10
    freeze_skills: bool = False
    reset_skills: bool = False
    max_new_skills: int = 3
    debug: bool = False

    always_attach_image: bool = True
    image_steps: Optional[List[str]] = None
    json_max_retries: int = 1

    _JSON_RULES = (
        "\n\n[STRICT_OUTPUT_RULES]\n"
        "- Output RAW JSON only.\n"
        "- Do NOT wrap in ``` or ```json.\n"
        "- Do NOT include any extra commentary.\n"
        "- ALL string values must be single-line (no literal newlines).\n"
    )

    IG_PROMPT = """You are the Information Gathering module for MathVista.
You receive:
- user_prompt (may contain few-shot examples); focus ONLY on the LAST target question.
- an image (the target figure/table/diagram).

Task: Extract facts only. Do NOT solve.

Return STRICT JSON:
{
  "problem_summary": "...",
  "knowns": [{"name": "...", "value": "...", "unit": "...", "source":"text|image", "note":"..."}],
  "asks": ["..."],
  "constraints": ["..."],
  "visual_facts": ["..."],
  "question_type_guess": "arithmetic|algebra|geometry|probability|chart|table|physics|logic|other",
  "ambiguity": ["..."]
}
""" + _JSON_RULES

    TI_PROMPT = """You are the Task Inference module for MathVista.
Input is extracted info JSON for the LAST target question.

Decide solution route and tool/skill needs. Do NOT output final answer.

Return STRICT JSON:
{
  "task_goal": "...",
  "subproblems": ["..."],
  "required_skills": ["..."],
  "tool_needs": ["python"|"sympy"|"none"],
  "risks": ["..."],
  "plan_style": "direct|tool-assisted|verify-heavy"
}
""" + _JSON_RULES

    SC_PROMPT = """You are the Skill Curation module for MathVista.
Input:
- task inference JSON
- current skill catalog (including learned skills)

Select skills to use.
If missing reusable capability, propose new_skill_specs.

Return STRICT JSON:
{
  "selected_skills": [{"name":"...", "purpose":"...", "how_to_use":"..."}],
  "new_skill_specs": [
     {"name":"...", "signature":"(args)->ret", "purpose":"...", "inputs":["..."], "outputs":"...", "edge_cases":["..."], "tests":[{"input":"...", "expect":"..."}], "tags":["..."]}
  ],
  "skill_priority": ["skill1","skill2"]
}
""" + _JSON_RULES

    SKILL_WRITER_PROMPT = """You are the Skill Writer module.
Input is new_skill_specs (reusable). You must produce SAFE python code for each skill.

Constraints:
- Code must define EXACTLY one function with the given name.
- No imports, no file/network access, no open(), no eval/exec, no attribute calls like x.y().
- Use only basic arithmetic, control flow, and safe builtins: abs,min,max,sum,len,round,range,int,float,str
- If you need math functions, use provided global `math` (already available).

Return STRICT JSON:
{
  "skills": [
    {
      "name": "...",
      "signature": "...",
      "code": "def name(...): ...",
      "tests": [
        {"code": "print(name(...))", "expect_contains": "..."}
      ],
      "tags": ["..."]
    }
  ]
}
""" + _JSON_RULES

    AP_PROMPT = """You are the Action Planning module for MathVista.
Input:
- info JSON
- task inference JSON
- selected skills
- skill catalog (learned)
- memory snippets (optional)

Output an executable plan. NO final answer yet.

Allowed actions: "reason", "python", "normalize"
Return STRICT JSON:
{
  "steps": [
    {"intention":"...", "action":"reason", "note":"..."},
    {"intention":"...", "action":"python", "code":"..."},
    {"intention":"...", "action":"normalize", "text":"..."}
  ],
  "final_format": "Answer: {ans}"
}

Rules:
- steps length <= MAX_STEPS
- python code must be short and deterministic
- If a learned skill exists, call it instead of rewriting logic.
- final_format must force a clean extraction.
""" + _JSON_RULES

    SR_PROMPT = """You are the Self-Reflection module for MathVista.
Input:
- info JSON
- tool execution logs
- draft answer (optional)

Check mistakes: units, rounding, choice mapping, axis reading, etc.
Fix formatting to be extractor-friendly.

Return STRICT JSON:
{
  "issues_found": ["..."],
  "fixes": ["..."],
  "final_answer": "Answer: ...",
  "confidence": 0.0
}

Final answer MUST be a single line starting with "Answer:".
""" + _JSON_RULES

    MEM_PROMPT = """You are the Memory module for MathVista.
Input:
- info JSON
- final answer
- reflection JSON

Write reusable tips/pitfalls/format rules only (short).
Return STRICT JSON:
{
  "memory_writes": [
    {"type":"skill_tip|pitfall|format_rule|verification_rule", "key":"...", "content":"...", "tags":["..."]}
  ]
}
""" + _JSON_RULES

    def _memory_path(self) -> str:
        os.makedirs(self.output_dir, exist_ok=True)
        return os.path.join(self.output_dir, "cradle_memory.jsonl")

    def _append_memory(self, writes: List[Dict[str, Any]]) -> None:
        if not writes:
            return
        with open(self._memory_path(), "a", encoding="utf-8") as f:
            for w in writes:
                f.write(_j(w) + "\n")

    def _load_memory_snippets(self, tags: List[str], k: int = 6) -> List[Dict[str, Any]]:
        path = self._memory_path()
        if not os.path.exists(path):
            return []
        res = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()[-300:]
            for ln in reversed(lines):
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    obj = json.loads(ln)
                except Exception:
                    continue
                itags = set(obj.get("tags", []))
                if itags.intersection(set(tags)):
                    res.append(obj)
                if len(res) >= k:
                    break
        except Exception:
            return []
        return res

    def _should_attach_image(self, step: str, need_image: bool) -> bool:
        if self.always_attach_image:
            return True
        if need_image:
            return True
        if self.image_steps is None:
            return step in {"IG", "AP", "SR"}
        return step in set(self.image_steps)

    def _wrap_prompt(self, step: str, prompt: str, obs: Observation, attach_image: bool) -> str:
        header = (
            f"[CRADLE_STEP]{step}\n"
            f"[OBS_IMAGE_ATTACHED]{str(bool(attach_image and obs.image is not None))}\n"
        )
        if obs.meta:
            safe_meta = {k: obs.meta[k] for k in list(obs.meta.keys())[:8]}
            header += f"[OBS_META]{_j(safe_meta)}\n"
        return header + prompt

    def _llm_call(self, step: str, prompt: str, obs: Observation, need_image: bool) -> str:
        attach = self._should_attach_image(step, need_image)
        wrapped = self._wrap_prompt(step, prompt, obs, attach)
        img = obs.image if attach else None
        return self.backbone.get_response(user_prompt=wrapped, decoded_image=img)

    def _llm_call_json(self, step: str, prompt: str, obs: Observation, need_image: bool) -> Dict[str, Any]:
        last = ""
        for attempt in range(self.json_max_retries + 1):
            p = prompt
            if attempt > 0:
                p = prompt + "\n\n[REMINDER] Output RAW JSON only. No code fences. Strings must be single-line."
            last = self._llm_call(step, p, obs, need_image)
            try:
                return _extract_json_obj(last)
            except Exception:
                continue
        raise ValueError(f"[{step}] JSON parse failed. Raw head: {last[:220]}")

    def get_response(self, user_prompt: str, decoded_image=None) -> str:
        obs = Observation(
            text=user_prompt,
            image=decoded_image,
            meta={"source": "MathVista", "ts": _now_ts()},
        )

        if not hasattr(self, "_skill_mgr"):
            self._skill_mgr = SkillManager(
                base_dir=self.output_dir,
                subdir=self.skills_subdir,
                freeze=self.freeze_skills,
                reset=self.reset_skills,
                max_new_skills=self.max_new_skills,
                debug=self.debug,
            )
        skill_mgr: SkillManager = self._skill_mgr

        ig = self._llm_call_json(
            "IG",
            self.IG_PROMPT + "\n\n[USER_PROMPT]\n" + obs.text,
            obs,
            need_image=True,
        )

        ti = self._llm_call_json(
            "TI",
            self.TI_PROMPT + "\n\n[INFO_JSON]\n" + _j(ig),
            obs,
            need_image=False,
        )

        qtype = ig.get("question_type_guess", "other")
        mem_snips = self._load_memory_snippets([qtype, "mathvista"], k=6)

        sc_input = (
            self.SC_PROMPT
            + "\n\n[TASK_INFERENCE_JSON]\n"
            + _j(ti)
            + "\n\n[SKILL_CATALOG]\n"
            + _j(skill_mgr.catalog())
        )
        sc = self._llm_call_json("SC", sc_input, obs, need_image=False)

        new_specs = sc.get("new_skill_specs", [])
        if isinstance(new_specs, list) and new_specs and (not self.freeze_skills):
            sw_input = self.SKILL_WRITER_PROMPT + "\n\n[NEW_SKILL_SPECS]\n" + _j(new_specs)
            sw = self._llm_call_json("SkillWriter", sw_input, obs, need_image=False)
            accepted = 0
            for sk in sw.get("skills", []):
                if accepted >= self.max_new_skills:
                    break
                name = sk.get("name", "")
                code = sk.get("code", "")
                signature = sk.get("signature", "")
                tests = sk.get("tests", [])
                tags = sk.get("tags", [])
                ok, _ = skill_mgr.add_or_update_skill(
                    name=name,
                    code=code,
                    signature=signature,
                    tags=tags,
                    tests=tests,
                )
                if ok:
                    accepted += 1

        ap_prompt = self.AP_PROMPT.replace("MAX_STEPS", str(self.max_steps))
        ap_input = (
            ap_prompt
            + "\n\n[INFO_JSON]\n"
            + _j(ig)
            + "\n\n[TASK_INFERENCE_JSON]\n"
            + _j(ti)
            + "\n\n[SELECTED_SKILLS]\n"
            + _j(sc.get("selected_skills", []))
            + "\n\n[LEARNED_SKILL_CATALOG]\n"
            + _j(skill_mgr.catalog())
            + "\n\n[MEMORY_SNIPPETS]\n"
            + _j(mem_snips)
        )
        ap = self._llm_call_json("AP", ap_input, obs, need_image=True)

        tool_logs = []
        steps = ap.get("steps", [])
        if isinstance(steps, list):
            for idx, step in enumerate(steps[: self.max_steps]):
                action = step.get("action")
                intention = step.get("intention", "")
                if action == "python":
                    code = step.get("code", "")
                    out, err = safe_exec(code, skill_mgr.ctx)
                    tool_logs.append(
                        {
                            "i": idx,
                            "action": "python",
                            "intention": intention,
                            "code": code,
                            "stdout": out,
                            "error": err,
                        }
                    )
                elif action == "reason":
                    tool_logs.append(
                        {
                            "i": idx,
                            "action": "reason",
                            "intention": intention,
                            "note": step.get("note", ""),
                        }
                    )
                elif action == "normalize":
                    tool_logs.append(
                        {
                            "i": idx,
                            "action": "normalize",
                            "intention": intention,
                            "text": step.get("text", ""),
                        }
                    )
                else:
                    tool_logs.append({"i": idx, "action": "unknown", "intention": intention, "raw": step})

        sr_input = (
            self.SR_PROMPT
            + "\n\n[INFO_JSON]\n"
            + _j(ig)
            + "\n\n[TOOL_LOGS]\n"
            + _j(tool_logs)
            + "\n\n[DRAFT]\n"
        )
        sr = self._llm_call_json("SR", sr_input, obs, need_image=True)

        final_answer = str(sr.get("final_answer", "")).strip()
        if not final_answer.startswith("Answer:"):
            final_answer = "Answer: " + final_answer

        mem_input = (
            self.MEM_PROMPT
            + "\n\n[INFO_JSON]\n"
            + _j(ig)
            + "\n\n[FINAL_ANSWER]\n"
            + final_answer
            + "\n\n[REFLECTION_JSON]\n"
            + _j(sr)
        )
        try:
            mem = self._llm_call_json("MEM", mem_input, obs, need_image=False)
            self._append_memory(mem.get("memory_writes", []))
        except Exception:
            pass

        return final_answer
