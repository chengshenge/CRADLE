import ast
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import logging
import uuid
import traceback
from pathlib import Path
import hashlib


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


def _sha1_text(s: str) -> str:
    try:
        return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()
    except Exception:
        return "0" * 40


def _sha1_bytes(b: bytes) -> str:
    try:
        return hashlib.sha1(b).hexdigest()
    except Exception:
        return "0" * 40


def _truncate(s: str, n: int = 240) -> str:
    if s is None:
        return ""
    s = str(s)
    if len(s) <= n:
        return s
    return s[: n // 2] + " ... " + s[-n // 2 :]


def _atomic_write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)



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

def _safe_eval_expr(expr: str, ctx: Dict[str, Any]) -> Any:
    """Safely evaluate a numeric/math expression.

    Allowed:
    - literals, arithmetic ops, comparisons/booleans
    - names present in ctx (e.g., math, sp, symbols) and safe builtins
    - attribute access only on whitelisted module names: math, sp
    - function calls only if the callee is a Name in ctx OR attribute on math/sp
    """
    tree = ast.parse(expr, mode="eval")

    ALLOWED = (
        ast.Expression, ast.Constant, ast.Name, ast.Load,
        ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
        ast.USub, ast.UAdd, ast.And, ast.Or,
        ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
        ast.Call, ast.keyword, ast.Attribute,
        ast.Tuple, ast.List, ast.Dict, ast.Set, ast.Subscript, ast.Slice,
    )

    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED):
            raise ValueError(f"Disallowed expression node: {type(node).__name__}")

        if isinstance(node, ast.Attribute):
            # only allow math.xxx or sp.xxx
            if not isinstance(node.value, ast.Name) or node.value.id not in ("math", "sp"):
                raise ValueError("Attribute access only allowed on math or sp")

        if isinstance(node, ast.Name):
            if node.id not in ctx and node.id not in ("True", "False", "None"):
                raise ValueError(f"Unknown name in expression: {node.id}")

        if isinstance(node, ast.Call):
            # allow f(...) where f is a Name in ctx OR math.xxx(...) / sp.xxx(...)
            if isinstance(node.func, ast.Name):
                if node.func.id not in ctx:
                    raise ValueError(f"Call to unknown function: {node.func.id}")
            elif isinstance(node.func, ast.Attribute):
                if not (isinstance(node.func.value, ast.Name) and node.func.value.id in ("math", "sp")):
                    raise ValueError("Call target must be a Name or math/sp attribute")
            else:
                raise ValueError("Disallowed call target")

    return eval(compile(tree, "<expr>", "eval"), ctx, ctx)


def _install_builtin_skills(ctx: Dict[str, Any]) -> None:
    """Install builtin helper skills into the safe execution context."""
    import math as _math
    ctx.setdefault("math", _math)

    # Ensure sympy module alias if available
    sp = ctx.get("sp", None)

    def calc_numeric(expr: str) -> float:
        val = _safe_eval_expr(expr, ctx)
        try:
            if sp is not None and isinstance(val, sp.Basic):
                val = float(sp.N(val))
        except Exception:
            pass
        return float(val)

    def sympy_solve(eqs, vars_):
        """Solve equations with sympy if available.

        Args:
          eqs: string like 'a=b' or list of such strings, or sympy expressions.
          vars_: variable name string like 'd' or 'd,x' or list of sympy symbols.
        Returns:
          sympy.solve(..., dict=True) output (list of dicts) or a best-effort numeric fallback.
        """
        if sp is None:
            raise RuntimeError("sympy (sp) is not available in this environment.")

        def _symbols_from_text(s: str):
            names = [x.strip() for x in s.split(",") if x.strip()]
            return [sp.Symbol(n) for n in names]

        def _parse_eq(e):
            if isinstance(e, str):
                if "=" in e:
                    lhs, rhs = e.split("=", 1)
                    return sp.Eq(sp.sympify(lhs, locals=ctx), sp.sympify(rhs, locals=ctx))
                return sp.sympify(e, locals=ctx)
            return e  # assume sympy expr/Eq

        eq_list = eqs
        if isinstance(eq_list, str):
            eq_list = [eq_list]
        parsed = [_parse_eq(e) for e in eq_list]

        if isinstance(vars_, str):
            syms = _symbols_from_text(vars_)
        else:
            syms = vars_

        try:
            return sp.solve(parsed, syms, dict=True)
        except Exception as e:
            # Fallback: try non-dict solve
            return sp.solve(parsed, syms)

    def format_final(ans, unit: Optional[str] = None, precision: Optional[int] = None) -> str:
        """Format as a single line 'Answer: ...'."""
        try:
            # unwrap common containers
            if isinstance(ans, (list, tuple)) and ans:
                ans = ans[0]
            if sp is not None and isinstance(ans, sp.Basic):
                ans = sp.N(ans)
                # try float
                try:
                    ans = float(ans)
                except Exception:
                    ans = str(ans)
        except Exception:
            pass

        if precision is not None:
            try:
                ans = round(float(ans), int(precision))
            except Exception:
                pass

        if unit:
            return f"Answer: {ans} {unit}".strip()
        return f"Answer: {ans}"

    # Install/overwrite
    ctx["calc_numeric"] = calc_numeric
    ctx["sympy_solve"] = sympy_solve
    ctx["format_final"] = format_final


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
    ctx: Dict[str, Any] = {
        "__builtins__": safe_builtins,
        "math": math,
    }
    try:
        import sympy as sp  # type: ignore
        ctx["sp"] = sp
    except Exception:
        pass

    # Critical: builtin skills used by AP must exist in ctx.
    _install_builtin_skills(ctx)
    return ctx
def safe_exec(code: str, g: Dict[str, Any]) -> Tuple[str, Optional[str], Any]:
    """Execute or evaluate generated python tool code in a constrained context.

    Returns:
      stdout_text, error_text_or_None, value
    Where `value` is the evaluated value for expression-only snippets (best-effort),
    otherwise None.
    """
    for pat in DENY_PATTERNS:
        if re.search(pat, code):
            return "", f"Blocked unsafe pattern: {pat}", None

    import io

    old_stdout = sys.stdout  # type: ignore
    buf = io.StringIO()
    sys.stdout = buf  # type: ignore
    err: Optional[str] = None
    val: Any = None

    try:
        tree = ast.parse(code, mode="exec")
        # If the snippet is a single expression, eval it to capture the return value.
        if len(tree.body) == 1 and isinstance(tree.body[0], ast.Expr):
            expr_node = tree.body[0].value
            compiled = compile(ast.Expression(expr_node), "<expr>", "eval")
            val = eval(compiled, g, g)
        else:
            exec(code, g, g)
    except Exception as e:
        err = str(e)
    finally:
        sys.stdout = old_stdout  # type: ignore

    out = buf.getvalue().strip()

    # If nothing was printed but we evaluated a value, expose it in stdout for downstream draft extraction.
    if (not out) and (err is None) and (val is not None):
        try:
            out = str(val)
        except Exception:
            out = repr(val)

    return out, err, val
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
                    _, run_err, _ = safe_exec(rec.code, self.ctx)
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
        _, err, _ = safe_exec(code, temp_ctx)
        if err:
            return False, f"Skill code exec error: {err}"

        if name not in temp_ctx or not callable(temp_ctx[name]):
            return False, "Defined function not found or not callable."

        if tests:
            for t in tests:
                tcode = t.get("code", "")
                expect = t.get("expect_contains", "")
                tout, terr, _ = safe_exec(tcode, temp_ctx)
                if terr:
                    return False, f"Test error: {terr}"
                if expect and expect not in str(tout):
                    return False, f"Test failed: expected output contains '{expect}', got '{tout}'"

        _, err2, _ = safe_exec(code, self.ctx)
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
    IG_PROMPT = """You are the Information Gathering (IG) module for MathVista-style multimodal math problems.
You receive:
- user_prompt (may contain few-shot examples); focus ONLY on the LAST target question.
- an image (the target figure/table/diagram/screenshot).

Your job: extract ALL information needed to solve. Do NOT solve the problem.

You may conceptually use OCR/region detection to read the image.
First decide ONE most relevant region to detect/zoom for this problem.

Region choice rules:
1) If the problem requires reading words/numbers from the image but no specific region is mentioned, choose "text".
2) If the problem involves a plot/graph (axes/legend/curve/bars), choose "chart".
3) If the problem involves a table (rows/columns/cells), choose "table".
4) If the problem involves geometry (shapes/angles/length labels), choose "diagram".
5) If the problem is purely textual and all necessary info is already in user_prompt, choose "null".
6) If the image is missing/blank/unreadable, choose "null".

Return STRICT JSON with the schema below.
IMPORTANT:
- Do NOT solve.
- Output RAW JSON only (no markdown fences).
- ALL string values must be single-line (no literal newlines).

{
  "region_to_detect": "text|chart|table|diagram|null",
  "reasoning_of_region": ["..."],
  "problem_understanding": "...",
  "math_elements_extracted": {
    "known": [{"name": "...", "value": "...", "unit": "...", "source": "text|image", "note": "..."}],
    "unknown": ["..."],
    "relations": ["..."],
    "constraints": ["..."],
    "options": [{"label": "A", "text": "..."}, {"label": "B", "text": "..."}]
  },
  "question_type_guess": "arithmetic|algebra|geometry|probability|statistics|calculus|chart|table|logic|other",
  "need_ocr": "yes|no",
  "noun_and_verb": "X nouns Y verbs",
  "task_horizon": 0,
  "reasoning_of_task": ["..."],

  "problem_summary": "...",
  "knowns": [{"name": "...", "value": "...", "unit": "...", "source":"text|image", "note":"..."}],
  "asks": ["..."],
  "constraints": ["..."],
  "visual_facts": ["..."],
  "ambiguity": ["..."]
}
""" + _JSON_RULES
    GTI_PROMPT = """You are the Gather Text Information (GTI) module for MathVista-style multimodal math problems.
Input:
- INFO_JSON for the LAST target question (produced by IG or IG-Refine). It includes:
  - region_to_detect (text|chart|table|diagram|null)
  - math_elements_extracted (known/unknown/relations/constraints/options)
- the image

Goal: extract ALL readable text/values/labels from the image, focusing on region_to_detect.
Do NOT solve the problem.

How to use inputs explicitly:
1) Use INFO_JSON.region_to_detect to decide what to read:
   - text: read all visible text lines/numbers.
   - chart: read title, axes labels, tick values, units, legend entries, and key plotted values if readable.
   - table: read headers and every relevant cell; preserve row/col structure.
   - diagram: read point labels, length/angle labels, and any marked relationships.
2) Use INFO_JSON.math_elements_extracted to target missing/uncertain items:
   - If IG listed an unknown value/relationship but did not provide the actual number/label, attempt to locate it in the image and record it.
   - If multiple-choice options exist, extract options exactly as shown.

Return STRICT JSON (RAW JSON only; no markdown fences). Strings must be single-line.
Schema:
{
  "region_used": "text|chart|table|diagram|null",
  "information": ["..."],

  "structured": {
    "text_lines": ["..."],

    "chart": {
      "title": "...",
      "x_axis": {"label":"...", "unit":"...", "ticks":["..."]},
      "y_axis": {"label":"...", "unit":"...", "ticks":["..."]},
      "legend": ["..."],
      "key_points": [{"label":"...", "x":"...", "y":"...", "note":"..."}]
    },

    "table": {
      "headers": ["..."],
      "rows": [
        {"row_label":"...", "cells":[{"col":"...", "text":"..."}]}
      ]
    },

    "diagram": {
      "labels": ["..."],
      "marked_lengths": [{"segment":"...", "value":"...", "unit":"..."}],
      "marked_angles": [{"angle":"...", "value":"...", "unit":"deg"}],
      "marked_relations": ["..."]
    }
  },

  "targets_from_math_elements": {
    "known_additions": [{"name":"...", "value":"...", "unit":"...", "source":"image", "note":"..."}],
    "relation_additions": ["..."],
    "constraint_additions": ["..."],
    "options_extracted": [{"label":"A","text":"..."},{"label":"B","text":"..."}],
    "unresolved_targets": ["..."]
  },

  "readability_issues": ["..."],
  "reasoning": ["..."]
}
""" + _JSON_RULES


    IGR_PROMPT = """You are the IG-Refine module for MathVista.
Input:
- INFO_JSON from IG (may be incomplete)
- GTI_JSON from Gather Text Information (more faithful raw readings from image)

Goal: merge and refine INFO_JSON using GTI_JSON to produce a cleaner, more complete INFO_JSON.
Do NOT solve the problem.

Explicit usage rules:
1) Keep INFO_JSON.region_to_detect as the primary modality, but you may adjust it if GTI proves a different modality is dominant.
2) Update INFO_JSON.math_elements_extracted by incorporating GTI_JSON.targets_from_math_elements:
   - append known_additions into known
   - append relation_additions into relations
   - append constraint_additions into constraints
   - if options were extracted, set options to those extracted (prefer exact on-image text)
3) Update INFO_JSON.visual_facts with key GTI readings (axis ticks, table cells, diagram labels) as short factual strings.
4) Keep all existing top-level keys used downstream:
   region_to_detect, reasoning_of_region, problem_understanding, math_elements_extracted,
   question_type_guess, need_ocr, noun_and_verb, task_horizon, reasoning_of_task,
   problem_summary, knowns, asks, constraints, visual_facts, ambiguity.

Return STRICT JSON using the SAME schema as INFO_JSON expected by downstream modules.
RAW JSON only; no markdown fences. Strings must be single-line.
""" + _JSON_RULES


    TI_PROMPT = """You are the Task Inference (TI) module for MathVista-style multimodal math problems.
You will be given:
- USER_PROMPT: the original question text (may include choices).
- INFO_JSON: output of IG-Refine (contains region_to_detect + math_elements_extracted and other fields).
- GTI_JSON: output of Gather Text Information (may contain extracted numbers/labels and unresolved targets).

Your job: produce a clean, decision-ready summary of the problem state and the next reasoning objectives.
Do NOT fully solve the problem. Do NOT invent visual facts.

You MUST explicitly use:
- INFO_JSON.region_to_detect to reason about what modality-specific checks are needed.
- INFO_JSON.math_elements_extracted (and GTI_JSON additions) as the ground truth for known/unknown/relations/options.

Return STRICT JSON (raw JSON only; no markdown fences). All strings must be single-line.

Required behavior:
1) Summarize the problem as a short story of the math situation: what is shown, what is asked, what must be computed.
2) List the entities/variables and the operations required (e.g., read chart -> compute slope -> map to option).
3) Identify any missing/uncertain visual targets from GTI_JSON.unresolved_targets and mark them as blockers.
4) Add modality-specific risk checks based on region_to_detect:
   - chart: axis scale, tick spacing, legend mapping, interpolation, units
   - table: row/column alignment, header meaning, units, totals vs per-item
   - diagram: label-to-segment mapping, angle markers, parallel/perpendicular indicators
   - text: currency/percent, rounding, multi-line conditions, hidden footnotes
5) If multiple-choice exists, ensure options are present and remind that final answer must map to one option label/value.

Schema:
{
  "problem_state_summary": "...",              // >= 6 sentences
  "entities": ["..."],                        // variables/objects/series names
  "unknowns": ["..."],                        // what must be found
  "required_operations": ["..."],             // ordered, high-level steps
  "blockers": ["..."],                        // unresolved visual targets or missing info
  "risk_checks": ["..."],                     // modality-specific checks
  "multiple_choice": {
    "is_mcq": true,
    "options": [{"label":"A","text":"..."}],
    "answer_format": "Return final as option label or exact value per instruction"
  },
  "grounding": {
    "region_to_detect": "text|chart|table|diagram|null",
    "must_use_fields": ["math_elements_extracted.known","math_elements_extracted.unknown","math_elements_extracted.relations","math_elements_extracted.constraints","math_elements_extracted.options"]
  },
  "tags": ["..."]
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
    AP_PROMPT = """You are the Action Planning (AP) module for MathVista-style multimodal math problems.

You are given:
- USER_PROMPT: the original question text (may include multiple-choice options).
- INFO_JSON: refined information extraction (contains region_to_detect + math_elements_extracted + constraints).
- GTI_JSON: direct readings from the image (numbers/labels/axes/cells) + unresolved_targets if any.
- TASK_INFERENCE_JSON: decision-ready summary, blockers, and risk checks.
- SELECTED_SKILLS + LEARNED_SKILL_CATALOG: available helper functions you MAY call from Python.
- MEMORY_SNIPPETS: prior similar patterns (optional).

Your job: output an executable reasoning plan that produces a SINGLE-LINE candidate answer string.
Do NOT output the final answer directly in normal text; instead produce steps that compute it and then a normalize step.

CRITICAL: You MUST explicitly use:
1) INFO_JSON.region_to_detect to decide which modality-specific facts matter (chart/table/diagram/text).
2) INFO_JSON.math_elements_extracted as the canonical state of the problem (known/unknown/relations/constraints/options).
3) GTI_JSON as the only source for image-read values; DO NOT invent visual numbers or labels.

Allowed step actions (MUST match executor):
- "reason": describe what to do next (no code).
- "python": provide short deterministic Python code to compute needed quantities; you may call functions from the skill library in this code.
- "normalize": produce a single-line candidate answer text (e.g., "Answer: B" or "Answer: 12.5").

Planning rules (must follow):
A) If TASK_INFERENCE_JSON.blockers is non-empty or GTI_JSON.unresolved_targets is non-empty:
   - Add an early "reason" step that lists the blockers.
   - Continue only with what is grounded; never fabricate missing values.
B) If region_to_detect == "chart":
   - Explicitly verify axis scale and units using GTI_JSON (ticks/labels).
   - If reading a point/slope, state which two points/intervals are used and ensure consistent units.
C) If region_to_detect == "table":
   - Explicitly identify the row/column headers used and confirm units.
D) If region_to_detect == "diagram":
   - Explicitly map labels to segments/angles; use constraints from math_elements_extracted.
E) If multiple-choice options exist:
   - Compute a value (if needed) THEN map to the closest/required option based on the problem instruction.
   - normalize MUST output the option label if the dataset expects a label.
F) Always include at least one final sanity check:
   - unit consistency, sign, range constraints, and whether the result satisfies relations/constraints.

Output STRICT JSON only (raw JSON; no markdown fences). All strings must be single-line.

Schema:
{
  "steps": [
    {"intention":"...", "action":"reason", "note":"..."},
    {"intention":"...", "action":"python", "code":"..."},
    {"intention":"...", "action":"normalize", "text":"Answer: ..."}
  ],
  "final_format": "Answer: {ans}"
}

Additional constraints:
- steps length <= MAX_STEPS
- python code should not import heavy libraries; use basic math only.
- If an equivalent helper exists in SELECTED_SKILLS/LEARNED_SKILL_CATALOG, call it instead of rewriting complex logic.
- The normalize text must be a single line and start with "Answer:".
""" + _JSON_RULES
    SR_PROMPT = """You are the Self-Reflection (SR) module for MathVista-style multimodal math problems.

You receive:
- INFO_JSON: the refined information extraction (must include region_to_detect and math_elements_extracted).
- GTI_JSON: raw visual readings (OCR/region reading results).
- TI_JSON: task interpretation / high-level solution plan.
- AP_JSON: action plan / step-by-step reasoning plan (may include tool-usage intentions).
- TOOL_LOGS: outputs/errors from tools (python/normalize/etc).
- DRAFT: a candidate final answer string (may be missing, wrong, or unformatted).

Your job:
1) Decide whether the DRAFT answer is correct and properly formatted.
2) If not, identify the single most probable failure cause and produce a corrected final answer.

You MUST explicitly use:
- INFO_JSON.region_to_detect to apply modality-specific checks:
  * chart: axis direction, tick spacing, legend mapping, interpolation, bar/line alignment
  * table: correct row/column, header mapping, units, totals vs per-row values
  * diagram: which segment/angle, label ownership, parallel/perpendicular marks, scale assumptions
  * text: units, percent vs decimal, wording constraints, "at least/at most", rounding rules
- INFO_JSON.math_elements_extracted (known/unknown/relations/constraints/options) and GTI_JSON visual readings to confirm all numbers/labels used are actually present (no hallucinated guaranteed values).
- Constraints (INFO_JSON.math_elements_extracted.constraints) to validate the final answer.
- If multiple-choice options exist, ensure the final answer matches the required format (usually option label), and that the value maps to the chosen option.

Reflection procedure (keep it concise but careful):
A) Restate what is being asked (from INFO_JSON.asks / problem_summary).
B) Verify key visual facts (compare INFO_JSON.math_elements_extracted vs GTI_JSON; note any mismatch).
C) Re-check the computation/logic quickly (you may rely on TOOL_LOGS if they are consistent; otherwise re-derive).
D) Validate formatting: single line starting with "Answer:"; include units if required; if MCQ, output the option label.

Return STRICT JSON (RAW JSON only; no markdown fences). All string values MUST be single-line (no literal newlines):

{
  "answer_correct": true,
  "issues_found": ["..."],
  "most_probable_cause": "visual_misread|wrong_relation|arithmetic_error|unit_error|option_mapping_error|rounding_error|format_error|other",
  "fixes": ["..."],
  "final_answer": "Answer: ...",
  "confidence": 0.0
}

Rules:
- If you are not fully sure, still output the best-justified final_answer and lower confidence.
- final_answer MUST begin with "Answer:" and be a single line.
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

    def _ensure_logging(self) -> None:
        """Initialize per-run JSON log + a human-readable debug log once."""
        if getattr(self, "_log_ready", False):
            return
        os.makedirs(self.output_dir, exist_ok=True)

        # Human-readable debug log (optional)
        self._debug_log_path = os.path.join(self.output_dir, "agent_debug.log")
        self._logger = logging.getLogger(f"cradle_mathvista_{id(self)}")
        self._logger.setLevel(logging.DEBUG if self.debug else logging.INFO)
        self._logger.propagate = False
        if not self._logger.handlers:
            fh = logging.FileHandler(self._debug_log_path, encoding="utf-8")
            fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
            fh.setFormatter(fmt)
            self._logger.addHandler(fh)

        # Default structured run log path (will mirror to log_output_<outputfile>.json if detectable)
        self._structured_log_default = os.path.join(self.output_dir, "log_output.json")

        self._log_ready = True

    def _guess_output_filename(self) -> Optional[str]:
        """Try to infer the output_*.json produced by the evaluator in this output_dir."""
        try:
            cands = []
            for fn in os.listdir(self.output_dir):
                if fn.startswith("output_") and fn.endswith(".json"):
                    p = os.path.join(self.output_dir, fn)
                    try:
                        cands.append((os.path.getmtime(p), fn))
                    except Exception:
                        cands.append((0.0, fn))
            if not cands:
                return None
            cands.sort(reverse=True)
            return cands[0][1]
        except Exception:
            return None

    def _structured_log_paths(self) -> List[str]:
        """Return a list of structured log paths to write to (default + mirrored name if possible)."""
        self._ensure_logging()
        paths = [self._structured_log_default]
        out_fn = self._guess_output_filename()
        if out_fn:
            mirror = os.path.join(self.output_dir, "log_" + out_fn)  # e.g., log_output_xxx.json
            if mirror not in paths:
                paths.append(mirror)
        return paths

    def _append_run_log(self, run_record: Dict[str, Any]) -> None:
        """Append a single run record into:
        - JSONL (recommended for reading / tail / grep)
        - legacy JSON array (best-effort, may be skipped for huge files)
        """
        self._ensure_logging()
        # Enrich with a compact status
        rr = dict(run_record)
        rr.setdefault("status", "error" if rr.get("errors") else "ok")

        paths = self._structured_log_paths()
        # 1) JSONL (append-only)
        for path in paths:
            jsonl_path = path[:-5] + ".jsonl" if path.endswith(".json") else path + ".jsonl"
            try:
                os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
                with open(jsonl_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rr, ensure_ascii=False) + "\n")
            except Exception as e:
                try:
                    self._logger.error(f"Failed appending JSONL run log to {jsonl_path}: {e}")
                except Exception:
                    pass

        # 2) Legacy JSON array (may become slow/huge; skip when too large)
        MAX_ARRAY_BYTES = 50 * 1024 * 1024  # 50MB
        for path in paths:
            try:
                if os.path.exists(path) and os.path.getsize(path) > MAX_ARRAY_BYTES:
                    # Rely on JSONL instead.
                    continue
                if os.path.exists(path):
                    with open(path, "r", encoding="utf-8") as f:
                        try:
                            arr = json.load(f)
                            if not isinstance(arr, list):
                                arr = []
                        except Exception:
                            arr = []
                else:
                    arr = []
                arr.append(rr)
                _atomic_write_json(path, arr)
            except Exception as e:
                try:
                    self._logger.error(f"Failed appending JSON array run log to {path}: {e}")
                except Exception:
                    pass

    def _write_artifacts(self, run_id: str, artifacts: Dict[str, Any]) -> str:
        """Write per-run artifacts under output_dir/artifacts/<run_id>/ and return the directory."""
        self._ensure_logging()
        adir = os.path.join(self.output_dir, "artifacts", run_id)
        os.makedirs(adir, exist_ok=True)

        for name, obj in (artifacts or {}).items():
            try:
                if name.endswith(".txt"):
                    with open(os.path.join(adir, name), "w", encoding="utf-8") as f:
                        f.write(str(obj))
                else:
                    _atomic_write_json(os.path.join(adir, f"{name}.json"), obj)
            except Exception as e:
                try:
                    self._logger.error(f"Failed writing artifact {name} for {run_id}: {e}")
                except Exception:
                    pass

        # Write a human-readable summary for quick debugging (tail-friendly)
        try:
            rr = artifacts.get("run_record", {}) if isinstance(artifacts, dict) else {}
            stage_times = rr.get("stage_times", {}) or {}
            errors = rr.get("errors", []) or []
            final_answer = rr.get("final_answer", "")
            tlogs = rr.get("tool_logs", []) or artifacts.get("tool_logs", []) or []
            py_errs = []
            for t in tlogs:
                if t.get("action") == "python" and t.get("error"):
                    py_errs.append(f"- step {t.get('i')}: {t.get('code')} -> {t.get('error')}")

            lines = []
            lines.append(f"run_id: {rr.get('run_id', run_id)}")
            lines.append(f"ts: {rr.get('ts')}")
            lines.append(f"prompt_hash: {rr.get('prompt_hash')}")
            lines.append(f"img_hash: {rr.get('img_hash')}")
            lines.append("")
            lines.append("stage_times (s):")
            for k in ["IG","GTI","IGR","TI","SC","AP","EXEC","SR","MEM","TOTAL"]:
                if k in stage_times:
                    lines.append(f"  {k}: {stage_times.get(k)}")
            lines.append("")
            lines.append(f"final_answer: {final_answer}")
            lines.append("")
            if errors:
                lines.append("errors:")
                for e in errors:
                    lines.append("  " + json.dumps(e, ensure_ascii=False))
            else:
                lines.append("errors: (none)")
            if py_errs:
                lines.append("")
                lines.append("python_step_errors:")
                lines.extend(py_errs)

            with open(os.path.join(adir, "summary.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
        except Exception:
            pass

        return adir

    def _should_attach_image(self, step: str, need_image: bool) -> bool:
        if self.always_attach_image:
            return True
        if need_image:
            return True
        if self.image_steps is None:
            return step in {"IG", "GTI", "AP", "SR"}
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
        self._ensure_logging()
        last = ""
        attach_image = self._should_attach_image(step, need_image)
        for attempt in range(self.json_max_retries + 1):
            p = prompt
            if attempt > 0:
                p = prompt + "\n\n[REMINDER] Output RAW JSON only. No code fences. Strings must be single-line."
            t0 = time.perf_counter()
            last = self._llm_call(step, p, obs, need_image)
            dt = time.perf_counter() - t0
            parse_ok = False
            obj: Optional[Dict[str, Any]] = None
            err = ""
            try:
                obj = _extract_json_obj(last)
                parse_ok = True
            except Exception as e:
                err = str(e)

            call_log = {
                "run_id": obs.meta.get("run_id"),
                "stage": step,
                "attempt": attempt,
                "need_image": bool(need_image),
                "attach_image": bool(attach_image),
                "prompt_len": len(p) if isinstance(p, str) else None,
                "latency_s": round(dt, 4),
                "parse_ok": parse_ok,
                "raw_head": _truncate(last, 240),
                "error": err,
            }
            llm_calls = obs.meta.get("llm_calls")
            if isinstance(llm_calls, list):
                llm_calls.append(call_log)
            try:
                if self.debug and not parse_ok:
                    self._logger.warning(f"[{step}] JSON parse failed (attempt {attempt}). head={call_log['raw_head']}")
            except Exception:
                pass

            if parse_ok and obj is not None:
                return obj

        raise ValueError(f"[{step}] JSON parse failed. Raw head: {last[:220]}")

    def get_response(self, user_prompt: str, decoded_image=None) -> str:
        obs = Observation(
            text=user_prompt,
            image=decoded_image,
            meta={"source": "MathVista", "ts": _now_ts()},
        )

        self._ensure_logging()

        # Per-problem run context
        prompt_hash = _sha1_text(user_prompt)[:10]
        run_id = f"{int(time.time()*1000)}_{prompt_hash}"
        llm_calls: List[Dict[str, Any]] = []
        stage_times: Dict[str, float] = {}
        errors: List[Dict[str, Any]] = []

        # Best-effort image hash (for detecting missing/duplicate images)
        img_hash = None
        try:
            if decoded_image is None:
                img_hash = None
            elif isinstance(decoded_image, (bytes, bytearray)):
                img_hash = _sha1_bytes(bytes(decoded_image))[:12]
            elif hasattr(decoded_image, "tobytes"):
                img_hash = _sha1_bytes(decoded_image.tobytes())[:12]
            else:
                img_hash = _sha1_text(str(type(decoded_image)))[:12]
        except Exception:
            img_hash = None

        obs.meta["run_id"] = run_id
        obs.meta["llm_calls"] = llm_calls
        obs.meta["img_hash"] = img_hash

        t_total0 = time.perf_counter()

        def _stage(name: str):
            class _StageCtx:
                def __enter__(self_inner):
                    self_inner.t0 = time.perf_counter()
                    return self_inner
                def __exit__(self_inner, exc_type, exc, tb):
                    stage_times[name] = round(time.perf_counter() - self_inner.t0, 6)
                    return False
            return _StageCtx()

        def _missing_keys(obj: Dict[str, Any], keys: List[str]) -> List[str]:
            miss = []
            for k in keys:
                if k not in obj:
                    miss.append(k)
            return miss

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

        with _stage("IG"):
            ig0 = self._llm_call_json(
            "IG",
            self.IG_PROMPT + "\n\n[USER_PROMPT]\n" + obs.text,
            obs,
            need_image=True,
            )
            _mk = _missing_keys(ig0, ["region_to_detect", "math_elements_extracted", "question_type_guess"])
            if _mk:
                errors.append({'stage':'IG','type':'missing_keys','missing':_mk})

        with _stage("GTI"):
            gti = self._llm_call_json(
            "GTI",
            self.GTI_PROMPT + "\n\n[INFO_JSON]\n" + _j(ig0),
            obs,
            need_image=True,
            )
            # Schema-flex validation: GTI uses region_used / structured.text_lines / targets_from_math_elements.unresolved_targets
            _mk = []
            if not (("region_to_detect" in gti) or ("region_used" in gti)):
                _mk.append("region_used")
            _text_lines = None
            if "text_lines" in gti:
                _text_lines = gti.get("text_lines")
            else:
                _text_lines = (gti.get("structured") or {}).get("text_lines")
            if _text_lines is None:
                _mk.append("text_lines")
            _unresolved = None
            if "unresolved_targets" in gti:
                _unresolved = gti.get("unresolved_targets")
            else:
                _unresolved = ((gti.get("targets_from_math_elements") or {}).get("unresolved_targets"))
            if _unresolved is None:
                _mk.append("unresolved_targets")
            if _mk:
                errors.append({'stage':'GTI','type':'missing_keys','missing':_mk})
        with _stage("IGR"):
            ig = self._llm_call_json(
            "IGR",
            self.IGR_PROMPT + "\n\n[INFO_JSON]\n" + _j(ig0) + "\n\n[GTI_JSON]\n" + _j(gti),
            obs,
            need_image=False,
            )
            _mk = _missing_keys(ig, ["region_to_detect", "math_elements_extracted"])
            if _mk:
                errors.append({'stage':'IGR','type':'missing_keys','missing':_mk})

        with _stage("TI"):
            ti = self._llm_call_json(
            "TI",
            self.TI_PROMPT + "\n\n[USER_PROMPT]\n" + obs.text + "\n\n[INFO_JSON]\n" + _j(ig) + "\n\n[GTI_JSON]\n" + _j(gti),
            obs,
            need_image=False,
            )
            _mk = _missing_keys(ti, ["problem_state_summary", "required_operations", "risk_checks", "grounding"])
            if _mk:
                errors.append({'stage':'TI','type':'missing_keys','missing':_mk})

        qtype = ig.get("question_type_guess", "other")
        mem_snips = self._load_memory_snippets([qtype, "mathvista"], k=6)

        sc_input = (
            self.SC_PROMPT
            + "\n\n[TASK_INFERENCE_JSON]\n"
            + _j(ti)
            + "\n\n[SKILL_CATALOG]\n"
            + _j(skill_mgr.catalog())
        )
        with _stage("SC"):
            sc = self._llm_call_json("SC", sc_input, obs, need_image=False)
            # SC may omit optional keys; only selected_skills is required.
            _mk = _missing_keys(sc, ["selected_skills"])
            if _mk:
                errors.append({'stage':'SC','type':'missing_keys','missing':_mk})
            # Treat missing tags as a warning only (log-friendly, no hard failure).
            if "tags" not in sc:
                errors.append({'stage':'SC','type':'warning_missing_optional','missing':['tags']})
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
            + "\n\n[USER_PROMPT]\n" + obs.text
            + "\n\n[INFO_JSON]\n"
            + _j(ig)
            + "\n\n[GTI_JSON]\n"
            + _j(gti)
            + "\n\n[TASK_INFERENCE_JSON]\n"
            + _j(ti)
            + "\n\n[SELECTED_SKILLS]\n"
            + _j(sc.get("selected_skills", []))
            + "\n\n[LEARNED_SKILL_CATALOG]\n"
            + _j(skill_mgr.catalog())
            + "\n\n[MEMORY_SNIPPETS]\n"
            + _j(mem_snips)
        )
        with _stage("AP"):
            ap = self._llm_call_json("AP", ap_input, obs, need_image=True)
            _mk = _missing_keys(ap, ["steps", "final_format"])
            if _mk:
                errors.append({'stage':'AP','type':'missing_keys','missing':_mk})

        tool_logs = []
        t_exec0 = time.perf_counter()
        steps = ap.get("steps", [])
        if isinstance(steps, list):
            for idx, step in enumerate(steps[: self.max_steps]):
                action = step.get("action")
                intention = step.get("intention", "")
                if action == "python":
                    code = step.get("code", "")
                    _t_py0 = time.perf_counter()
                    out, err, val = safe_exec(code, skill_mgr.ctx)
                    _t_py = time.perf_counter() - _t_py0
                    tool_logs.append(
                        {
                            "i": idx,
                            "action": "python",
                            "intention": intention,
                            "code": code,
                            "stdout": out,
                            "value_repr": _truncate(val, 240) if val is not None else "",
                            "latency_s": round(_t_py, 4),
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

        stage_times['EXEC'] = round(time.perf_counter() - t_exec0, 6)
        # Build a draft answer from executed plan logs (prefer last normalize text)
        draft = ""
        try:
            norm_texts = [t.get("text", "") for t in tool_logs if t.get("action") == "normalize" and t.get("text")]
            if norm_texts:
                draft = str(norm_texts[-1]).strip()
            else:
                py_outs = [t.get("stdout", "") for t in tool_logs if t.get("action") == "python" and t.get("stdout")]
                draft = str(py_outs[-1]).strip() if py_outs else ""
        except Exception:
            draft = ""

        sr_input = (
            self.SR_PROMPT
            + "\n\n[INFO_JSON]\n"
            + _j(ig)
            + "\n\n[GTI_JSON]\n"
            + _j(gti)
            + "\n\n[TI_JSON]\n"
            + _j(ti)
            + "\n\n[AP_JSON]\n"
            + _j(ap)
            + "\n\n[TOOL_LOGS]\n"
            + _j(tool_logs)
            + "\n\n[DRAFT]\n"
            + draft
        )
        with _stage("SR"):
            sr = self._llm_call_json("SR", sr_input, obs, need_image=True)
            _mk = _missing_keys(sr, ["final_answer", "issues_found", "confidence"])
            if _mk:
                errors.append({'stage':'SR','type':'missing_keys','missing':_mk})

        final_answer = str(sr.get("final_answer", "")).strip()
        if not final_answer.startswith("Answer:"):
            final_answer = "Answer: " + final_answer

        mem_input = (
            self.MEM_PROMPT
            + "\n\n[INFO_JSON]\n"
            + _j(ig)
            + "\n\n[GTI_JSON]\n"
            + _j(gti)
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

        # Persist debug artifacts + structured logs beside final results
        run_record = {
            'run_id': run_id,
            'ts': obs.meta.get('ts'),
            'prompt_hash': prompt_hash,
            'prompt_len': len(user_prompt) if isinstance(user_prompt, str) else None,
            'img_hash': img_hash,
            'stage_times': stage_times,
            'errors': errors,
            'llm_calls': llm_calls,
            'region_to_detect': (ig.get('region_to_detect') if isinstance(ig, dict) else None),
            'question_type_guess': (ig.get('question_type_guess') if isinstance(ig, dict) else None),
            'ig0': ig0,
            'gti': gti,
            'ig': ig,
            'ti': ti,
            'sc': sc,
            'ap': ap,
            'tool_logs': tool_logs,
            'draft': draft,
            'sr': sr,
            'final_answer': final_answer,
        }
        artifacts = {
            'run_record': run_record,
            'user_prompt.txt': _truncate(user_prompt, 4000),
            'ig0': ig0,
            'gti': gti,
            'ig': ig,
            'ti': ti,
            'sc': sc,
            'ap': ap,
            'tool_logs': tool_logs,
            'sr': sr,
            'final_answer': {'final_answer': final_answer},
        }
        try:
            artifacts_dir = self._write_artifacts(run_id, artifacts)
            run_record['artifacts_dir'] = artifacts_dir
        except Exception as e:
            errors.append({'stage':'ARTIFACTS','type':'write_failed','error':str(e)})
        stage_times['TOTAL'] = round(time.perf_counter() - t_total0, 6)
        try:
            self._append_run_log(run_record)
        except Exception:
            pass
        return final_answer
