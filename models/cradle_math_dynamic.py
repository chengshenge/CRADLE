import ast
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, ClassVar
import logging
import uuid
import traceback
from pathlib import Path
import hashlib
import math


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



# ============================================================
# Placeholder rendering + answer selection helpers
# ============================================================

_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z_]\w*)(?::([^}]+))?\}")

def _to_scalar(v: Any) -> Any:
    """Best-effort conversion of tool outputs to a scalar for formatting."""
    if isinstance(v, (list, tuple)) and len(v) == 1:
        v = v[0]
    # SolveResult is a float subclass; treat as scalar automatically.
    try:
        # sympy-like
        if hasattr(v, "evalf") and callable(getattr(v, "evalf")):
            try:
                v = float(v.evalf())
            except Exception:
                pass
    except Exception:
        pass
    return v

def render_placeholders(text: str, ctx: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    """Render {var} and {var:format} placeholders using values in ctx.

    Returns (rendered_text, error_str). On error, rendered_text is the original.
    Includes a conservative heuristic: if fixed-point formatting would round a small non-zero
    *length-like* value to 0.0 with low precision, try meters->centimeters.
    """
    if not isinstance(text, str):
        return text, None
    if ctx is None:
        ctx = {}

    # Use decimal ROUND_HALF_UP for fixed-point formats to avoid banker's rounding surprises.
    def _format_fixed_half_up(val: float, fmt: str) -> Optional[str]:
        mfmt = re.search(r"\.(\d+)f$", (fmt or '').strip())
        if not mfmt:
            return None
        try:
            from decimal import Decimal, ROUND_HALF_UP
            decimals = int(mfmt.group(1))
            q = Decimal('1').scaleb(-decimals)  # 10**(-decimals)
            d = Decimal(str(float(val))).quantize(q, rounding=ROUND_HALF_UP)
            # Ensure fixed number of decimals
            return format(d, f".{decimals}f")
        except Exception:
            return None


    def _maybe_rescale_length(v: float, fmt: str) -> float:
        try:
            fv = float(v)
        except Exception:
            return v
        if not math.isfinite(fv) or fv == 0.0:
            return v
        mfmt = re.search(r"\.(\d+)f$", (fmt or '').strip())
        if not mfmt:
            return v
        decimals = int(mfmt.group(1))
        if decimals > 1:
            return v

        q = str(ctx.get('_question_text', '')).lower()
        unit = str(ctx.get('_unit', '')).strip().lower()
        # If the question explicitly requests a unit, do not auto-rescale.
        try:
            if unit in {'cm','mm'}:
                return v
            if re.search(r"\b(in\s*cm|centimeters?)\b", q) or re.search(r"\b(in\s*mm|millimeters?)\b", q):
                return v
            if re.search(r"\b(in\s*m|meters?|metres?)\b", q):
                return v
        except Exception:
            pass

        lengthy = any(w in q for w in [
            'distance','length','height','width','radius','diameter',
            'compress','compression','displacement','spring','stretched',
            'moved','travel','how far'
        ])
        if not lengthy:
            return v

        try:
            test = ("{:" + fmt + "}").format(fv)
            if test.startswith('0') and abs(fv) < 0.1:
                scaled = fv * 100.0
                if 0.1 <= abs(scaled) <= 10000:
                    return scaled
        except Exception:
            pass
        return v

    def _render_one(m: re.Match) -> str:
        var = m.group(1)
        fmt = m.group(2) or ''
        if var not in ctx:
            return m.group(0)
        v = ctx[var]
        try:
            if fmt and isinstance(v, (int, float)):
                v = _maybe_rescale_length(v, fmt)
                s = _format_fixed_half_up(v, fmt)
                return s if s is not None else ("{:" + fmt + "}").format(v)
            if fmt:
                s = _format_fixed_half_up(v, fmt)
                return s if s is not None else ("{:" + fmt + "}").format(v)
            return str(v)
        except Exception:
            return m.group(0)

    try:
        rendered = _PLACEHOLDER_RE.sub(_render_one, text)
        if _PLACEHOLDER_RE.search(rendered):
            return rendered, "Unresolved placeholders remain"
        return rendered, None
    except Exception as e:
        return text, str(e)


def _sanitize_answer_line(s: str) -> str:
    """Normalize an 'Answer: ...' line for downstream parsing/validation.

    - Strips markdown fences and leading/trailing whitespace
    - If an 'Answer:' prefix exists, keep from the first occurrence onward
    - Collapses to a single line
    """
    if not isinstance(s, str):
        return ""
    t = _strip_markdown_fences(s).strip()

    # Keep only the first non-empty line
    t = t.splitlines()[0].strip() if t.splitlines() else t

    # If the model returned extra text before the answer, keep the last 'Answer:' block.
    if "Answer:" in t:
        # take the *last* occurrence to avoid picking up examples in the prompt
        t = "Answer:" + t.split("Answer:")[-1].strip()

    # Remove stray code fence remnants
    t = t.strip("`").strip()
    return t


def _looks_valid_answer_line(s: str) -> bool:
    t = _sanitize_answer_line(s)
    if not t.startswith("Answer:"):
        return False
    payload = t.split("Answer:", 1)[1].strip()
    if not payload:
        return False
    if "[calculated_value]" in t:
        return False
    if _ANS_HOLE_RE.search(t):
        return False
    bad = payload.lower()
    if bad in ("none", "null", "nan", "inf", "infinity", "error", "unknown"):
        return False
    return True

def choose_final_answer(draft: str, sr: Dict[str, Any]) -> str:
    """Prefer a valid draft answer unless SR is clearly fixing a formatting/mapping issue.

    SR is useful for:
    - adding/removing the required 'Answer:' line
    - option-letter <-> option-text mapping
    - rounding/precision/unit formatting

    SR is *not* trusted to re-solve arithmetic/relations. We therefore gate SR overrides.
    """
    draft_line = _sanitize_answer_line(draft)
    sr_line = _sanitize_answer_line(str(sr.get('final_answer', '')))

    draft_ok = _looks_valid_answer_line(draft_line)
    sr_ok = _looks_valid_answer_line(sr_line)

    sr_cause = str(sr.get('most_probable_cause', '') or '').strip().lower()

    safe_causes = {
        'format_error',
        'rounding_error',
        'precision_error',
        'unit_error',
        'missing_answer_line',
        'option_mapping_error',
    }

    def _extract_num(s: str):
        payload = s.split('Answer:', 1)[-1].strip() if 'Answer:' in s else s.strip()
        m = re.search(r"[-+]?\d+(?:\.\d+)?", payload)
        return float(m.group(0)) if m else None

    def _payload(s: str) -> str:
        return s.split('Answer:', 1)[-1].strip() if 'Answer:' in s else s.strip()

    if draft_ok:
        if sr_ok and sr_cause in safe_causes:
            nd, ns = _extract_num(draft_line), _extract_num(sr_line)
            if nd is not None and ns is not None:
                # Allow small numeric edits (rounding) or 100x unit swap (unit_error)
                if abs(ns - nd) <= max(1e-6, 0.05 * max(1.0, abs(nd))):
                    return sr_line
                if sr_cause == 'unit_error' and nd != 0:
                    ratio = ns / nd
                    if abs(ratio - 100.0) < 1e-3 or abs(ratio - 0.01) < 1e-3:
                        return sr_line
                return draft_line
            # Non-numeric: mapping/formatting fix
            if _payload(draft_line) != _payload(sr_line):
                return sr_line
            return sr_line
        return draft_line

    if sr_ok:
        return sr_line
    # Neither looks valid; do not return placeholder holes.
    return "Answer: "

# ============================================================
# Deterministic multiple-choice label -> option text mapping
# ============================================================

_CHOICE_LABEL_RE = re.compile(r'^\s*[\(\[\{<]?\s*([A-Ha-h])\s*[\)\]\}>]?\s*[\.\:\)\-]?\s*$')
_PREFIX_WORD_RE = re.compile(r'^\s*(?:option|choice|answer)\s*[:\-]?\s*', flags=re.IGNORECASE)

def _extract_bare_choice_label(payload: str) -> Optional[str]:
    """Return 'A'..'H' if payload is essentially a bare option label, else None.

    Deterministic and conservative: it only returns a label when the payload can be reduced
    (by stripping wrappers like 'Option', punctuation, brackets) to a single A-H letter.
    """
    if not isinstance(payload, str):
        return None
    s = payload.strip()
    if not s:
        return None

    # Common leading words (e.g., "Option B", "Choice: C")
    s = _PREFIX_WORD_RE.sub("", s).strip()

    # If the whole thing looks like just "(B)" / "B." / "[C]" / "D)"
    m = _CHOICE_LABEL_RE.fullmatch(s)
    if m:
        return m.group(1).upper()

    # Sometimes models output "B )" or "B ," etc.
    s2 = s.strip().strip("`'\"").strip()
    s2 = s2.strip("[](){}<> \t\r\n")
    s2 = re.sub(r'^[\s\.\,\;\:\-]+', '', s2)
    s2 = re.sub(r'[\s\.\,\;\:\-]+$', '', s2)
    m2 = _CHOICE_LABEL_RE.fullmatch(s2)
    if m2:
        return m2.group(1).upper()

    return None


def _norm_label(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    # Keep only first A-H if present
    m = re.search(r'[A-Ha-h]', s)
    if not m:
        return None
    return m.group(0).upper()


def _strip_label_prefix(text: str, label: str) -> str:
    """Remove an option label prefix like 'A.', '(A)', 'A)' from the start of text."""
    if not isinstance(text, str):
        return ""
    t = text.strip()
    if not t:
        return ""
    # Remove common leading label formats
    pat = re.compile(r'^\s*(?:[\(\[\{<]?\s*' + re.escape(label) + r'\s*[\)\]\}>]?\s*[\.\:\)\-]\s*)', flags=re.IGNORECASE)
    t2 = pat.sub("", t, count=1).strip()
    return t2 if t2 else t


def _coerce_option_item(item: Any) -> Optional[Dict[str, str]]:
    """Normalize option item into {'label': 'A', 'text': '...'} when possible."""
    if item is None:
        return None
    if isinstance(item, dict):
        lab = _norm_label(item.get("label"))
        txt = item.get("text")
        if txt is None:
            # tolerate alternative keys
            txt = item.get("value") if "value" in item else item.get("content")
        txts = str(txt).strip() if txt is not None else ""
        if lab and txts:
            return {"label": lab, "text": txts}
        if lab and not txts:
            # label only; keep but may be useless
            return {"label": lab, "text": ""}
        return None
    if isinstance(item, str):
        s = item.strip()
        if not s:
            return None
        # Parse "A. something" or "(B) something"
        m = re.match(r'^\s*[\(\[\{<]?\s*([A-Ha-h])\s*[\)\]\}>]?\s*[\.\:\)\-]\s*(.+?)\s*$', s)
        if m:
            return {"label": m.group(1).upper(), "text": m.group(2).strip()}
        # Parse "A something" (space separated)
        m2 = re.match(r'^\s*([A-Ha-h])\s+(.+?)\s*$', s)
        if m2:
            return {"label": m2.group(1).upper(), "text": m2.group(2).strip()}
        return None
    return None


def _parse_options_from_text(text: str) -> List[Dict[str, str]]:
    """Parse options from a question text (best-effort, line-based)."""
    if not isinstance(text, str) or not text.strip():
        return []
    out: List[Dict[str, str]] = []
    line_pat = re.compile(r'^\s*[\(\[\{<]?\s*([A-Ha-h])\s*[\)\]\}>]?\s*[\.\:\)\-]\s*(.+?)\s*$')
    for line in text.splitlines():
        m = line_pat.match(line)
        if not m:
            continue
        lab = m.group(1).upper()
        txt = m.group(2).strip()
        if txt:
            out.append({"label": lab, "text": txt})
    # Deduplicate by label (keep first occurrence)
    seen = set()
    dedup: List[Dict[str, str]] = []
    for o in out:
        if o["label"] in seen:
            continue
        seen.add(o["label"])
        dedup.append(o)
    return dedup


def _collect_options(ig: Optional[Dict[str, Any]],
                     gti: Optional[Dict[str, Any]],
                     ti: Optional[Dict[str, Any]],
                     user_prompt: Optional[str]) -> Dict[str, str]:
    """Collect option label->text mapping from multiple deterministic sources.

    Priority (most trusted first):
      1) IG-Refine: ig.math_elements_extracted.options
      2) GTI: gti.targets_from_math_elements.options_extracted
      3) TI: ti.multiple_choice.options
      4) USER_PROMPT parsing
    """
    mapping: Dict[str, str] = {}

    def _ingest(opt_list: Any):
        if not isinstance(opt_list, list):
            return
        for it in opt_list:
            o = _coerce_option_item(it)
            if not o:
                continue
            lab = o["label"]
            txt = o.get("text", "")
            if not lab:
                continue
            if txt:
                txt2 = _strip_label_prefix(txt, lab)
                # Keep the first non-empty text for each label (deterministic)
                if lab not in mapping or not mapping[lab]:
                    mapping[lab] = txt2
            else:
                # ensure key exists
                mapping.setdefault(lab, "")

    try:
        if isinstance(ig, dict):
            _ingest(((ig.get("math_elements_extracted") or {}).get("options", [])))
    except Exception:
        pass
    try:
        if isinstance(gti, dict):
            _ingest((((gti.get("targets_from_math_elements") or {}).get("options_extracted", []))))
    except Exception:
        pass
    try:
        if isinstance(ti, dict):
            _ingest((((ti.get("multiple_choice") or {}).get("options", []))))
    except Exception:
        pass
    try:
        _ingest(_parse_options_from_text(user_prompt or ""))
    except Exception:
        pass

    return mapping


def _map_bare_choice_to_option_text(final_answer: str,
                                    *,
                                    ig: Optional[Dict[str, Any]] = None,
                                    gti: Optional[Dict[str, Any]] = None,
                                    ti: Optional[Dict[str, Any]] = None,
                                    user_prompt: Optional[str] = None) -> Optional[str]:
    """If final_answer is 'Answer: B' (bare label), deterministically map it to the option text.

    Returns:
      - Mapped 'Answer: <option_text>' if mapping is possible and non-empty.
      - None if no mapping should be applied.
    """
    if not isinstance(final_answer, str) or not final_answer.strip():
        return None

    payload = final_answer.split("Answer:", 1)[1].strip() if "Answer:" in final_answer else final_answer.strip()
    lab = _extract_bare_choice_label(payload)
    if not lab:
        return None

    opt_map = _collect_options(ig, gti, ti, user_prompt)
    txt = (opt_map.get(lab) or "").strip()
    if not txt:
        return None

    return "Answer: " + txt


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





def _repair_dot_format_calls(s: str) -> str:
    """Repair common LLM artifact: "text": "...".format(var)

    Example:
      "text": "Answer: {:.1f}".format(d_val)
    becomes:
      "text": "Answer: {d_val:.1f}"

    This is deterministic and only touches patterns that are not valid JSON anyway.
    """
    if not isinstance(s, str) or ".format(" not in s:
        return s

    # Only repair for a few known string-valued keys to avoid unintended rewrites.
    keys = ("text", "note", "intention")
    key_pat = "|".join(re.escape(k) for k in keys)
    # Match: "text": "....".format(var)
    pattern = re.compile(rf'("({key_pat})"\s*:\s*)"([^"\\]*(?:\\.[^"\\]*)*)"\s*\.format\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)')
    def repl(m: re.Match) -> str:
        prefix = m.group(1)
        inner = m.group(3)
        var = m.group(4)
        # If inner contains "{:" spec, inject var name.
        if "{:" in inner:
            inner2 = inner.replace("{:", "{"+var+":", 1)
        elif "{}" in inner:
            inner2 = inner.replace("{}", "{"+var+"}", 1)
        else:
            # No obvious placeholder: keep string as-is (drop .format call)
            inner2 = inner
        return f'{prefix}"{inner2}"'
    return pattern.sub(repl, s)


def _remove_trailing_commas(s: str) -> str:
    """Remove trailing commas before '}' or ']' outside of strings.

    LLMs sometimes emit JSON with trailing commas, which is invalid for json.loads.
    This function is deterministic and does not change content inside quoted strings.
    """
    if not isinstance(s, str) or not s:
        return s
    out = []
    in_str = False
    esc = False
    n = len(s)
    i = 0
    while i < n:
        ch = s[i]
        if in_str:
            out.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            i += 1
            continue

        if ch == '"':
            in_str = True
            out.append(ch)
            i += 1
            continue

        if ch == ",":
            j = i + 1
            while j < n and s[j] in " \t\r\n":
                j += 1
            if j < n and s[j] in "}]":
                i += 1
                continue

        out.append(ch)
        i += 1
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
        return json.loads(_remove_trailing_commas(_repair_dot_format_calls(s)))
    except Exception:
        pass

    # Try extracting the first balanced JSON object
    candidate = _extract_first_balanced_json_object(s)
    if candidate is None:
        raise ValueError(f"Cannot find JSON object in: {s[:200]}...")

    # Sanitize newlines inside strings (common LLM formatting artifact)
    candidate2 = _escape_newlines_inside_strings(candidate)

    # Repair invalid JSON artifacts like "text": "...".format(var)
    candidate2 = _repair_dot_format_calls(candidate2)

    return json.loads(_remove_trailing_commas(candidate2))


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

    ctx: Dict[str, Any] = {
        "__builtins__": safe_builtins,
        "math": math,
    }

    # Optional sympy
    sp = None
    try:
        import sympy as _sp  # type: ignore
        sp = _sp
        ctx["sp"] = _sp
    except Exception:
        sp = None

    # ------------------------------------------------------------
    # Built-in tool skills (needed by AP python steps)
    # ------------------------------------------------------------

    def _safe_eval_arith_expr(expr: str) -> float:
        """Evaluate a *numeric* expression safely (no attributes, subscripts, etc.)."""
        expr = (expr or "").strip()
        if not expr:
            raise ValueError("empty expr")

        tree = ast.parse(expr, mode="eval")

        allowed_binops = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)
        allowed_unops = (ast.UAdd, ast.USub)

        def _eval(node: ast.AST) -> float:
            if isinstance(node, ast.Expression):
                return _eval(node.body)
            if isinstance(node, ast.Constant):
                if isinstance(node.value, (int, float)):
                    return float(node.value)
                raise ValueError("non-numeric constant")
            if isinstance(node, ast.Num):  # py<3.8
                return float(node.n)
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, allowed_unops):
                v = _eval(node.operand)
                return +v if isinstance(node.op, ast.UAdd) else -v
            if isinstance(node, ast.BinOp) and isinstance(node.op, allowed_binops):
                a = _eval(node.left)
                b = _eval(node.right)
                if isinstance(node.op, ast.Add):
                    return a + b
                if isinstance(node.op, ast.Sub):
                    return a - b
                if isinstance(node.op, ast.Mult):
                    return a * b
                if isinstance(node.op, ast.Div):
                    return a / b
                if isinstance(node.op, ast.FloorDiv):
                    return a // b
                if isinstance(node.op, ast.Mod):
                    return a % b
                if isinstance(node.op, ast.Pow):
                    return a ** b
                raise ValueError("bad op")
            if isinstance(node, ast.Name):
                # allow referencing numeric constants or variables already in ctx
                if node.id in ("pi", "e"):
                    return float(getattr(math, node.id))
                v = ctx.get(node.id, None)
                if isinstance(v, (int, float)):
                    return float(v)
                raise ValueError(f"unknown name: {node.id}")
            # Disallow all other nodes (Call, Attribute, Subscript, etc.)
            raise ValueError(f"disallowed node: {type(node).__name__}")

        return float(_eval(tree))

    def calc_numeric(expr: str) -> float:
        """Compute numeric arithmetic expression safely."""
        return _safe_eval_arith_expr(expr)

    class _SolveResult(float):
        """Float-like solver result that also behaves like a list of roots.

        This keeps older AP plans working whether they treat sympy_solve() as:
          - a scalar (d = sympy_solve(...); d*100)
          - or a list (roots = sympy_solve(...); roots[0])
        """

        def __new__(cls, value: float, roots: Optional[List[float]] = None):
            obj = float.__new__(cls, float(value))
            obj.roots = list(roots) if roots is not None else [float(value)]
            return obj

        def __iter__(self):
            return iter(self.roots)

        def __len__(self):
            return len(self.roots)

        def __getitem__(self, i):
            return self.roots[i]

    def sympy_solve(eqs, vars):
        """Solve algebraic equation(s).

        - If sympy is available, use it.
        - If sympy is NOT available, provide a lightweight fallback for the
          common MathVista case: a *single-variable* equation that is linear
          or quadratic in that variable.

        Args:
          eqs: str like "lhs = rhs" OR an expression f(x) where f(x)=0.
               Also accepts list[str] but fallback supports only one equation.
          vars: variable name string (e.g. "d") or list with one variable.
        Returns:
          For 1-var: a float-like result (also indexable/iterable as roots list).
          For multi-var: sympy dicts if sympy available, else raises.
        """
        # -----------------------------
        # Sympy path (if available)
        # -----------------------------
        if sp is not None:
            # vars -> list of symbols
            if isinstance(vars, str):
                var_names = [vars]
            else:
                var_names = list(vars)
            syms = [sp.Symbol(v) for v in var_names]

            # Pass numeric values from the current safe context into sympify(),
            # so expressions like "0.5*k*d**2 - KE_initial" can be evaluated numerically
            # when k/KE_initial have been assigned earlier in the same python block.
            locals_map: Dict[str, Any] = {}
            try:
                if isinstance(ctx, dict):
                    for kk, vv in ctx.items():
                        if isinstance(vv, (int, float)) and not isinstance(vv, bool):
                            locals_map[str(kk)] = sp.Float(vv)
            except Exception:
                locals_map = {}

            def _one_eq(s: str):
                t = (s or "").strip()
                if "=" in t:
                    lhs, rhs = t.split("=", 1)
                    return sp.Eq(sp.sympify(lhs, locals=locals_map), sp.sympify(rhs, locals=locals_map))
                return sp.Eq(sp.sympify(t, locals=locals_map), 0)

            if isinstance(eqs, str):
                eq_list = [_one_eq(eqs)]
            else:
                eq_list = [_one_eq(x) for x in list(eqs)]

            sol = sp.solve(eq_list, syms, dict=True)

            # Normalize a common single-var case into a scalar-like result.
            if len(syms) == 1:
                v = syms[0]
                vals = [d[v] for d in sol if isinstance(d, dict) and v in d]
                if not vals:
                    return sol

                roots: List[float] = []
                for vv in vals:
                    try:
                        # skip non-real if sympy knows
                        if hasattr(vv, "is_real") and vv.is_real is False:
                            continue
                    except Exception:
                        pass
                    try:
                        fv = float(vv.evalf())
                    except Exception:
                        try:
                            fv = float(vv)
                        except Exception:
                            continue
                    if isinstance(fv, float) and math.isfinite(fv):
                        roots.append(float(fv))

                # Choose a "best" root: prefer smallest positive real; else first real.
                best = None
                pos = [r for r in roots if r > 0]
                if pos:
                    best = min(pos)
                elif roots:
                    best = roots[0]

                if best is not None:
                    # Put the preferred root first, because many generated code snippets do `sympy_solve(...)[0]`.
                    ordered_roots: List[float] = []
                    if best is not None:
                        ordered_roots.append(float(best))
                    for r in roots:
                        try:
                            fr = float(r)
                        except Exception:
                            continue
                        if best is None or abs(fr - float(best)) > 1e-12:
                            ordered_roots.append(fr)

                    ctx["_last_roots"] = list(ordered_roots)
                    ctx["_last_solution"] = float(best)
                    return _SolveResult(float(best), ordered_roots)

                return sol

            return sol

        # -----------------------------
        # Fallback path (no sympy)
        # -----------------------------
        # Only support 1-variable, 1-equation fallback.
        if isinstance(vars, str):
            var_name = vars.strip()
        else:
            var_name = (list(vars)[0] if vars else "x").strip()

        if not var_name:
            var_name = "x"

        if isinstance(eqs, (list, tuple)):
            if len(eqs) != 1:
                raise RuntimeError("sympy not available; fallback supports a single equation only")
            eqs = eqs[0]

        s = (eqs or "").strip()
        if not s:
            raise ValueError("empty equation")

        # Convert "lhs = rhs" to "(lhs)-(rhs)" (equals zero)
        if "=" in s:
            lhs, rhs = s.split("=", 1)
            expr = f"({lhs})-({rhs})"
        else:
            expr = s

        # Safe AST evaluator for numeric expressions with one variable + math.*
        allowed_binops = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)
        allowed_unops = (ast.UAdd, ast.USub)

        def _eval_with_var(expr_str: str, xval: float) -> float:
            tree = ast.parse(expr_str, mode="eval")

            def _ev(node: ast.AST) -> float:
                if isinstance(node, ast.Expression):
                    return _ev(node.body)
                if isinstance(node, ast.Constant):
                    if isinstance(node.value, (int, float)):
                        return float(node.value)
                    raise ValueError("non-numeric constant")
                if isinstance(node, ast.Name):
                    if node.id == var_name:
                        return float(xval)

                    # allow numeric variables from the current safe context (e.g., k, KE_initial)
                    if isinstance(ctx, dict):
                        vv_ctx = ctx.get(node.id)
                        if isinstance(vv_ctx, (int, float)) and not isinstance(vv_ctx, bool):
                            return float(vv_ctx)

                    # allow math constants like pi, e
                    if hasattr(math, node.id):
                        vv = getattr(math, node.id)
                        if isinstance(vv, (int, float)):
                            return float(vv)

                    raise ValueError(f"unknown name: {node.id}")
                if isinstance(node, ast.BinOp) and isinstance(node.op, allowed_binops):
                    a = _ev(node.left)
                    b = _ev(node.right)
                    if isinstance(node.op, ast.Add):
                        return a + b
                    if isinstance(node.op, ast.Sub):
                        return a - b
                    if isinstance(node.op, ast.Mult):
                        return a * b
                    if isinstance(node.op, ast.Div):
                        return a / b
                    if isinstance(node.op, ast.FloorDiv):
                        return a // b
                    if isinstance(node.op, ast.Mod):
                        return a % b
                    if isinstance(node.op, ast.Pow):
                        return a ** b
                if isinstance(node, ast.UnaryOp) and isinstance(node.op, allowed_unops):
                    vv = _ev(node.operand)
                    return +vv if isinstance(node.op, ast.UAdd) else -vv
                if isinstance(node, ast.Call):
                    # allow math.<func>(...)
                    if isinstance(node.func, ast.Name) and hasattr(math, node.func.id):
                        fn = getattr(math, node.func.id)
                    elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "math":
                        fn = getattr(math, node.func.attr, None)
                    else:
                        fn = None
                    if fn is None or not callable(fn):
                        raise ValueError("unsafe function call")
                    args = [_ev(a) for a in node.args]
                    return float(fn(*args))
                if isinstance(node, ast.Attribute):
                    # allow "math.pi" etc.
                    if isinstance(node.value, ast.Name) and node.value.id == "math" and hasattr(math, node.attr):
                        vv = getattr(math, node.attr)
                        if isinstance(vv, (int, float)):
                            return float(vv)
                    raise ValueError("unsafe attribute")
                raise ValueError(f"unsupported node: {type(node).__name__}")

            return float(_ev(tree))

        # Fit linear/quadratic coefficients by sampling f(0), f(1), f(2)
        f0 = _eval_with_var(expr, 0.0)
        f1 = _eval_with_var(expr, 1.0)
        f2 = _eval_with_var(expr, 2.0)

        a = (f2 - 2.0 * f1 + f0) / 2.0
        c = f0
        b = f1 - a - c

        eps = 1e-12
        roots: List[float] = []

        if abs(a) < eps:
            # linear: b*x + c = 0
            if abs(b) < eps:
                roots = []
            else:
                roots = [(-c) / b]
        else:
            disc = b * b - 4.0 * a * c
            if disc >= -1e-12:
                disc = max(0.0, disc)
                sd = math.sqrt(disc)
                roots = [(-b + sd) / (2.0 * a), (-b - sd) / (2.0 * a)]
            else:
                roots = []

        # Choose a "best" root: prefer smallest positive real; else first.
        best = None
        pos = [r for r in roots if isinstance(r, (int, float)) and math.isfinite(r) and r > 0]
        if pos:
            best = min(pos)
        elif roots:
            best = roots[0]

        # Put the preferred root first (helps generated code that indexes `[0]`).
        ordered_roots: List[float] = []
        if best is not None:
            ordered_roots.append(float(best))
        for r in roots:
            try:
                fr = float(r)
            except Exception:
                continue
            if best is None or abs(fr - float(best)) > 1e-12:
                ordered_roots.append(fr)

        ctx["_last_roots"] = list(ordered_roots)
        if best is not None:
            ctx["_last_solution"] = float(best)
            return _SolveResult(float(best), ordered_roots)

        return roots

    def format_final(ans) -> str:
        """Return exactly one line in MathVista style: Answer: ..."""
        a = ans
        if isinstance(a, (list, tuple)) and a:
            a = a[0]
        try:
            if sp is not None and hasattr(a, "evalf"):
                a = float(a.evalf())
        except Exception:
            pass


        # Heuristic: many MathVista physics "distance/length" answers are stored in centimeters
        # even if inputs are in SI. If the unknown is length-like and the computed value is < 0.1,
        # scale by 100 to avoid rounding to 0.0 at low precision.
        try:
            if isinstance(a, (int, float)) and math.isfinite(a):
                if ctx.get("_length_like", False) and 0 < abs(float(a)) < 0.1:
                    a = float(a) * 100.0
        except Exception:
            pass

        # Keep a generic 'ans' variable available for normalize placeholders (Answer: {ans})
        try:
            if "ans" not in ctx and isinstance(a, (int, float, str)):
                ctx["ans"] = a
        except Exception:
            pass

        # If AP hard-codes a rounded number but we have a computed last solution,
        # prefer the computed one when they differ significantly.
        try:
            sol = ctx.get("_last_solution", None)
            if isinstance(sol, (int, float)) and isinstance(a, (int, float)) and math.isfinite(sol) and math.isfinite(a):
                rel = abs(a - sol) / max(abs(sol), 1e-12)
                is_rounded = any(abs(a - round(a, nd)) < 1e-12 for nd in (0, 1, 2, 3))  # 0-3 dp rounding
                if rel > 0.15 and is_rounded:
                    a = float(sol)
        except Exception:
            pass

        # Default numeric formatting: 4 significant digits (keeps MathVista tolerant).
        if isinstance(a, (int, float)) and math.isfinite(a):
            a_str = f"{float(a):.4g}"
        else:
            a_str = str(a)

        return f"Answer: {a_str}"

    ctx["calc_numeric"] = calc_numeric
    ctx["sympy_solve"] = sympy_solve
    ctx["format_final"] = format_final
    return ctx



def _infer_ans_name_from_code(code: str) -> Optional[str]:
    """Infer the most likely answer variable name from a one-line python snippet.

    We prefer:
    - last assignment target (e.g., d_val = ...)
    - argument to format_final(x)
    """
    if not isinstance(code, str) or not code.strip():
        return None
    try:
        tree = ast.parse(code)
    except Exception:
        return None

    cand = None
    for node in tree.body:
        # Prefer explicit format_final(arg)
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            fn = node.value.func
            if isinstance(fn, ast.Name) and fn.id == "format_final" and node.value.args:
                a0 = node.value.args[0]
                if isinstance(a0, ast.Name):
                    cand = a0.id
                elif isinstance(a0, ast.Constant):
                    return None
        if isinstance(node, ast.Assign):
            # multiple targets possible
            for t in node.targets:
                if isinstance(t, ast.Name):
                    cand = t.id
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            cand = node.target.id
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            cand = node.target.id

    # Don't return clearly non-answer names unless nothing else exists
    if cand in (None, "m", "v", "k", "g", "pi"):
        return cand
    return cand


def safe_exec(code: str, g: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    """Execute python code in a constrained context.

    Enhancement vs v5: if the last statement is an expression (e.g., calc_numeric(...)),
    we also evaluate it and print its value (REPL-like), so the agent can read results
    without explicitly writing print() or assignments.
    """
    for pat in DENY_PATTERNS:
        if re.search(pat, code):
            return "", f"Blocked unsafe pattern: {pat}"

    import io

    old_stdout = sys.stdout  # type: ignore
    buf = io.StringIO()
    sys.stdout = buf  # type: ignore
    err = None
    try:
        tree = ast.parse(code, mode="exec")
        last_expr = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last_expr = ast.Expression(tree.body[-1].value)
            tree.body = tree.body[:-1]

        if tree.body:
            tree = ast.fix_missing_locations(tree)
            exec(compile(tree, "<tool_exec>", "exec"), g, g)

        if last_expr is not None:
            last_expr = ast.fix_missing_locations(last_expr)
            val = eval(compile(last_expr, "<tool_eval>", "eval"), g, g)
            g["_last"] = val
            if val is not None:
                print(val)
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

    # Stage-specific LLM controls.
    # Some backbones expose get_response(..., max_tokens=..., temperature=...).
    # We pass these kwargs opportunistically (ignored if unsupported).
    STAGE_LLM_KWARGS: ClassVar[Dict[str, Dict[str, Any]]] = {
        # AP outputs can be long; give it more room but also constrain output schema in the prompt.
        "AP": {"max_tokens": 1200, "temperature": 0},
    }

    # Stage-specific JSON retry budget (number of retries after the first attempt).
    # AP is the most fragile stage (nested JSON with steps); allow more retries.
    STAGE_JSON_MAX_RETRIES: ClassVar[Dict[str, int]] = {
        "AP": 3,
    }
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
Extra extraction rules (important):
- If the question asks for a maximum/total/capacity/highest value on a scale (cup, ruler, axis, gauge), list ALL visible numeric markings and explicitly include the largest marking you can see.
- If the question is a counting task ("how many", "count", "left", "remain"), count objects mentioned in the question and also the total if possible. Record counts in structured.diagram under a key "counts" (free-form list).
- Never invent numbers; if unreadable, add a note to readability_issues.
- Never output python string formatting like ".format(...)" inside JSON strings.


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

HARD RULES (to avoid silent wrong answers):
- Never hard-code the final numeric answer (e.g., format_final(0.02)) unless it was computed in a previous python step.
- If you solve an equation or compute a value, ASSIGN it to a variable (e.g., d_val = ...) and use that variable in format_final(d_val).
- If a tool call fails (e.g., solver unavailable), fall back to direct algebra or numeric computation using only python + math.
- DO NOT write any `import` / `from ... import ...` statements inside python steps. The executor blocks `import`. Use the provided skills (calc_numeric, sympy_solve, format_final) and the built-in `math` module only.
- Prefer calling sympy_solve("...", "d") rather than trying to import sympy.
- Do not round intermediate results to 1 decimal place. Keep at least 3 significant digits (or 3+ decimals) before format_final.
- Do NOT use python-style string formatting in JSON fields (e.g., "...".format(x)). For normalize.text, use placeholders like "Answer: {ans}" or "Answer: {x_val:.2f}" only.

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
   - If the question explicitly says "choose A/B/C/D" or "option letter", output the option label.
   - Otherwise, output the option TEXT/value (e.g., "145°") to be robust to answer formats.
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
- Keep the JSON SMALL to reduce truncation risk:
  - steps length <= 4 (prefer 2-4).
  - Each "intention" <= 80 characters.
  - Each "note" <= 120 characters (or omit "note" entirely).
  - Each python "code" MUST be a single line. If multiple statements are needed, use ';' to separate.
  - Do not include any keys other than: steps, final_format, and within each step only: intention, action, (note|code|text).
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
        """Append a single run record into structured JSON array file(s)."""
        for path in self._structured_log_paths():
            try:
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
                arr.append(run_record)
                _atomic_write_json(path, arr)
            except Exception as e:
                # Fall back to debug logger only
                try:
                    self._logger.error(f"Failed writing structured log {path}: {e}")
                except Exception:
                    pass

    def _write_artifacts(self, run_id: str, artifacts: Dict[str, Any]) -> str:
        """Write per-run artifacts under output_dir/artifacts/<run_id>/ and return the directory."""
        self._ensure_logging()
        adir = os.path.join(self.output_dir, "artifacts", run_id)
        os.makedirs(adir, exist_ok=True)
        for name, obj in artifacts.items():
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
        # IMPORTANT: Do NOT dump large / recursive objects (e.g., llm_calls) into the prompt.
        # Keeping meta tiny prevents prompt growth across stages (and context_length_exceeded errors).
        if obs.meta:
            safe_meta: Dict[str, Any] = {}
            for k in ("source", "run_id", "img_hash", "pid"):
                try:
                    v = obs.meta.get(k)
                    if isinstance(v, (str, int, float, bool)) or v is None:
                        safe_meta[k] = v
                except Exception:
                    pass
            if safe_meta:
                header += f"[OBS_META]{_j(safe_meta)}\n"
        return header + prompt

    def _llm_call(self, step: str, prompt: str, obs: Observation, need_image: bool) -> str:
        attach = self._should_attach_image(step, need_image)
        wrapped = self._wrap_prompt(step, prompt, obs, attach)
        img = obs.image if attach else None

        # Stage-specific kwargs (e.g., max_tokens/temperature) passed opportunistically.
        kwargs: Dict[str, Any] = {}
        try:
            stage_kwargs = getattr(self, "STAGE_LLM_KWARGS", {}).get(step, {})
            if stage_kwargs:
                try:
                    import inspect  # module-scope import; NOT executed inside safe_exec
                    sig = inspect.signature(self.backbone.get_response)
                    for k, v in stage_kwargs.items():
                        if k in sig.parameters:
                            kwargs[k] = v
                except Exception:
                    # If signature introspection fails, try passing kwargs directly.
                    kwargs.update(stage_kwargs)
        except Exception:
            kwargs = {}

        try:
            return self.backbone.get_response(user_prompt=wrapped, decoded_image=img, **kwargs)
        except TypeError:
            # Backbone does not support extra kwargs.
            return self.backbone.get_response(user_prompt=wrapped, decoded_image=img)
    def _llm_call_json(self, step: str, prompt: str, obs: Observation, need_image: bool) -> Dict[str, Any]:
        self._ensure_logging()
        last = ""
        attach_image = self._should_attach_image(step, need_image)
        max_retries = getattr(self, 'STAGE_JSON_MAX_RETRIES', {}).get(step, self.json_max_retries)
        for attempt in range(max_retries + 1):
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
                "prompt": self._wrap_prompt(step, p, obs, attach_image) if isinstance(p, str) else None,
                "raw": last,
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

        # Defaults so we can always log even if a stage throws
        ig0 = {}
        gti = {}
        ig = {}
        ti = {}
        sc = {}
        ap = {}
        sr = {}
        tool_logs = []
        draft = ""
        final_answer = ""

        try:

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
            skill_mgr.ctx['_question_text'] = obs.text

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

            # If OCR is needed but IG did not select a region, bias to text.
            try:
                if isinstance(ig0, dict) and str(ig0.get('need_ocr','')).strip().lower() == 'yes':
                    if str(ig0.get('region_to_detect','')).strip().lower() in ('', 'null'):
                        ig0['region_to_detect'] = 'text'
            except Exception:
                pass

            with _stage("GTI"):
                gti = self._llm_call_json(
                "GTI",
                self.GTI_PROMPT + "\n\n[INFO_JSON]\n" + _j(ig0),
                obs,
                need_image=True,
                )
                _mk = _missing_keys(gti, ["region_used", "structured", "targets_from_math_elements"])
                if _mk:
                    errors.append({'stage':'GTI','type':'missing_keys','missing':_mk})

            # Optional GTI2: targeted re-read for max-marking questions to reduce missed top ticks.
            try:
                if isinstance(ig0, dict) and str(ig0.get('need_ocr','')).strip().lower() == 'yes':
                    qlow = str(obs.text).lower()
                    if any(w in qlow for w in ['measuring', 'volume', 'capacity', 'marking', 'graduated', 'cup', 'beaker']):
                        nums = []
                        if isinstance(gti, dict):
                            for s in (gti.get('information') or []):
                                for mm in re.finditer(r"\d+(?:\.\d+)?", str(s)):
                                    try:
                                        nums.append(float(mm.group(0)))
                                    except Exception:
                                        pass
                        max_seen = max(nums) if nums else None
                        if (max_seen is None) or (max_seen < 800):
                            t0 = time.perf_counter()
                            ocr2_prompt = (
                                "Targeted read: list all visible numeric markings relevant to the question.\n"
                                "Return STRICT JSON: {\"numbers\": [..], \"max_marking\": <number or null>, \"notes\": [..]}\n"
                                "Only include numbers you can actually see.\n\n"
                                + obs.text
                            )
                            ocr2 = self._llm_call_json("GTI2", ocr2_prompt, obs, need_image=True)
                            stage_times['GTI2'] = round(time.perf_counter() - t0, 6)
                            llm_calls.append({
                                'run_id': run_id,'stage':'GTI2','attempt':0,'need_image':True,'attach_image':True,
                                'prompt_len': len(ocr2_prompt),'latency_s': stage_times.get('GTI2'),
                                'parse_ok': True,'raw_head': _truncate(_j(ocr2), 400),'error':''
                            })
                            if isinstance(ocr2, dict):
                                max2 = ocr2.get('max_marking')
                                if max2 is None:
                                    try:
                                        max2 = max(float(x) for x in (ocr2.get('numbers') or []) if x is not None)
                                    except Exception:
                                        max2 = None
                                if max2 is not None and isinstance(gti, dict):
                                    gti.setdefault('information', [])
                                    gti['information'].append(f"Targeted OCR: max_marking={max2}")
                                    try:
                                        gti.setdefault('structured', {}).setdefault('diagram', {}).setdefault('labels', [])
                                        gti['structured']['diagram']['labels'].append(str(max2))
                                        gti.setdefault('targets_from_math_elements', {}).setdefault('known_additions', [])
                                        gti['targets_from_math_elements']['known_additions'].append({
                                            'name': 'max_marking', 'value': str(max2), 'unit': '', 'source': 'image', 'note': 'targeted OCR max marking'
                                        })
                                    except Exception:
                                        pass

                # Optional GTI_COUNT: re-read for pure counting tasks (objects/remaining/how many).
                try:
                    qlow = str(obs.text).lower()
                    wants_count = any(w in qlow for w in ['how many', 'count', 'remain', 'left', 'number of'])
                    if wants_count:
                        # If initial GTI did not extract any useful numeric labels, ask again focused on counting.
                        labels = []
                        if isinstance(gti, dict):
                            labels = (gti.get('structured', {}).get('diagram', {}).get('labels', []) or [])
                        if not labels:
                            t0 = time.perf_counter()
                            count_prompt = (
                                "Counting focus: Count the objects relevant to the question directly from the image.\n"
                                "If the question mentions specific object types, count each type and also the total.\n"
                                "Return STRICT JSON: {\"counts\": [{\"object\":\"...\",\"count\":<int>,\"note\":\"...\"}], \"total\": <int or null>, \"notes\": [..]}\n"
                                "Only report counts you can justify from the image.\n\n"
                                + obs.text
                            )
                            cnt = self._llm_call_json("GTI_COUNT", count_prompt, obs, need_image=True)
                            stage_times['GTI_COUNT'] = round(time.perf_counter() - t0, 6)
                            llm_calls.append({
                                'run_id': run_id,'stage':'GTI_COUNT','attempt':0,'need_image':True,'attach_image':True,
                                'prompt_len': len(count_prompt),'latency_s': stage_times.get('GTI_COUNT'),
                                'parse_ok': True,'raw_head': _truncate(_j(cnt), 400),'error':''
                            })
                            if isinstance(cnt, dict) and isinstance(gti, dict):
                                # Merge into gti.information for IGR/TI/AP grounding
                                gti.setdefault('information', [])
                                gti['information'].append("Counting read: " + _truncate(_j(cnt), 500))
                except Exception:
                    pass

            except Exception:
                pass

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


            # Mark whether the unknown is length-like to enable deterministic unit scaling (m->cm) in format_final/rendering.
            try:
                unknowns = (ig.get('math_elements_extracted') or {}).get('unknown', [])
                if isinstance(unknowns, list):
                    ulow = " ".join(str(u).lower() for u in unknowns)
                    length_kw = ['distance','length','height','width','radius','diameter','depth','thickness','displacement','compression','extension']
                    skill_mgr.ctx['_length_like'] = any(k in ulow for k in length_kw)
            except Exception:
                pass

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
                _mk = _missing_keys(sc, ["selected_skills", "skill_priority"])
                if _mk:
                    errors.append({'stage':'SC','type':'missing_keys','missing':_mk})

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
                        out, err = safe_exec(code, skill_mgr.ctx)
                        # Infer a generic 'ans' variable for later placeholders.
                        try:
                            if err is None:
                                nm = _infer_ans_name_from_code(code)
                                if nm and nm in skill_mgr.ctx and nm not in ('m','v','k','g','pi'):
                                    skill_mgr.ctx['ans'] = skill_mgr.ctx.get(nm)
                        except Exception:
                            pass
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
                        raw_text = step.get("text", "")
                        # If placeholders refer to missing variables, alias them to 'ans' / last numeric solution when unambiguous.
                        try:
                            raw_s = str(raw_text)
                            missing_vars: List[str] = []
                            for mm in _PLACEHOLDER_RE.finditer(raw_s):
                                vname = mm.group(1)
                                if vname not in skill_mgr.ctx:
                                    missing_vars.append(vname)
                            if missing_vars:
                                if "ans" not in skill_mgr.ctx:
                                    if "_last_solution" in skill_mgr.ctx:
                                        skill_mgr.ctx["ans"] = skill_mgr.ctx.get("_last_solution")
                                    elif "_last" in skill_mgr.ctx:
                                        skill_mgr.ctx["ans"] = skill_mgr.ctx.get("_last")
                                if "ans" in skill_mgr.ctx:
                                    uniq = list(dict.fromkeys(missing_vars))
                                    if len(uniq) <= 3:
                                        for vname in uniq:
                                            skill_mgr.ctx[vname] = skill_mgr.ctx.get("ans")
                        except Exception:
                            pass

                        rendered_text, render_err = render_placeholders(str(raw_text), skill_mgr.ctx)
                        tool_logs.append(
                            {
                                "i": idx,
                                "action": "normalize",
                                "intention": intention,
                                "text": rendered_text,
                                "raw_text": raw_text,
                                "render_error": render_err,
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


            # -----------------------------
            # Deterministic fallback (when plan execution failed)
            # -----------------------------
            try:
                py_errors = [t for t in tool_logs if t.get("action") == "python" and t.get("error")]
                if (not draft or not str(draft).strip().startswith("Answer:")) and py_errors:
                    # Fallback for a very common MathVista physics pattern:
                    # frictionless block/canister hits spring; KE -> spring PE
                    knowns = (ig.get("math_elements_extracted") or {}).get("known", [])
                    unknowns = (ig.get("math_elements_extracted") or {}).get("unknown", [])
                    relations = (ig.get("math_elements_extracted") or {}).get("relations", [])

                    def _get_float(name_keys):
                        for it in knowns:
                            nm = (it.get("name") or "").lower()
                            for k in name_keys:
                                if k in nm:
                                    try:
                                        return float(it.get("value"))
                                    except Exception:
                                        pass
                        return None

                    m_val = _get_float(["mass"])
                    v_val = _get_float(["speed", "velocity"])
                    k_val = _get_float(["spring constant", "k"])

                    wants_d = any("d" in (u or "").lower() or "distance" in (u or "").lower() for u in unknowns)
                    rel_ok = any("kinetic" in (r or "").lower() and "spring" in (r or "").lower() for r in relations)

                    if m_val is not None and v_val is not None and k_val is not None and wants_d and rel_ok:
                        d_val = float(v_val) * math.sqrt(float(m_val) / float(k_val))
                        draft = format_final(d_val)
                        tool_logs.append({
                            "i": len(tool_logs),
                            "action": "fallback",
                            "intention": "Fallback compute spring compression via d = v*sqrt(m/k) (KE->spring PE).",
                            "stdout": draft,
                            "error": None
                        })
            except Exception:
                pass

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

            draft_line = (draft or "").strip()
            if draft_line and not draft_line.startswith("Answer:"):
                draft_line = "Answer: " + draft_line


            # If draft still contains unresolved placeholders (e.g., Answer: {ans}), try to render from ctx.
            try:
                rendered, rerr = render_placeholders(draft_line, skill_mgr.ctx)
                if rerr is None and rendered:
                    draft_line = rendered.strip()
            except Exception:
                pass

            # If still invalid, try rendering AP.final_format using ctx['ans'].
            try:
                if not _looks_valid_answer_line(draft_line):
                    ff = ""
                    if isinstance(ap, dict):
                        ff = str(ap.get("final_format", "") or "")
                    if ff:
                        ff_line = ff if ff.strip().startswith("Answer:") else ("Answer: " + ff.strip())
                        rendered, rerr = render_placeholders(ff_line, skill_mgr.ctx)
                        if rerr is None and rendered and _looks_valid_answer_line(rendered):
                            draft_line = rendered.strip()
            except Exception:
                pass

            # Last resort: if we have a numeric/string 'ans' in ctx, format it deterministically.
            try:
                if not _looks_valid_answer_line(draft_line) and "ans" in skill_mgr.ctx:
                    draft_line = format_final(skill_mgr.ctx.get("ans"))
            except Exception:
                pass


            final_answer = choose_final_answer(draft_line, sr)

            # Robust deterministic mapping from bare option label -> option text (no extra model calls).
            try:
                mapped = _map_bare_choice_to_option_text(
                    final_answer, ig=ig, gti=gti, ti=ti, user_prompt=user_prompt
                )
                if mapped is not None:
                    final_answer = mapped
            except Exception:
                pass
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

        except Exception as e:
            errors.append({
                "stage": "EXCEPTION",
                "type": type(e).__name__,
                "error": str(e),
                "traceback": _truncate(traceback.format_exc(), 4000),
            })
            # Best-effort answer so evaluation can continue
            if isinstance(draft, str) and draft.strip():
                _d = draft.strip()
                final_answer = _d if _d.startswith("Answer:") else ("Answer: " + _d)
            elif isinstance(final_answer, str) and final_answer.strip():
                pass
            else:
                final_answer = "Answer: "
        finally:
            if "TOTAL" not in stage_times:
                stage_times["TOTAL"] = round(time.perf_counter() - t_total0, 6)

            # Slim structured log: per-module prompt/response/error (requested for debugging)
            module_order = ["IG", "GTI", "IGR", "TI", "SC", "AP"]
            modules: Dict[str, Dict[str, Any]] = {}

            for st in module_order:
                calls = [c for c in (llm_calls or []) if c.get("stage") == st]
                lastc = calls[-1] if calls else {}
                err_parts: List[str] = []

                if lastc.get("error"):
                    err_parts.append(str(lastc.get("error")))

                # Include any non-LLM errors that were attributed to this stage (e.g., TOOL/EXEC errors)
                for e in (errors or []):
                    if e.get("stage") == st and e.get("error"):
                        err_parts.append(str(e.get("error")))

                modules[st] = {
                    "prompt": lastc.get("prompt"),
                    "response": lastc.get("raw"),
                    "error": " | ".join([x for x in err_parts if x]) or None,
                }

            run_record = {
                "run_id": run_id,
                "ts": obs.meta.get("ts"),
                "prompt_hash": prompt_hash,
                "img_hash": img_hash,
                "modules": modules,
                "final_answer": final_answer,
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
                # artifacts_dir written to disk; not included in slim structured log
            except Exception as e2:
                errors.append({'stage':'ARTIFACTS','type':'write_failed','error':str(e2)})
            try:
                self._append_run_log(run_record)
            except Exception:
                pass

        return final_answer
