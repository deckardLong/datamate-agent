"""Official DataMate scorer implementing Penalized Accuracy (PA) and type matching rules."""

from __future__ import annotations

import math
import re
import unicodedata
from typing import Any


def normalize_string(value: Any) -> str:
    """Normalize a string according to DataMate rules.

    1. Unicode NFC
    2. Lowercase
    3. Replace every character that is not a letter, digit, or whitespace with a space
    4. Strip leading/trailing whitespace and collapse runs of whitespace to one space
    Diacritics are preserved.
    """
    text = unicodedata.normalize("NFC", str(value)).lower()
    chars = [
        ch if (ch.isalpha() or ch.isdigit() or ch.isspace()) else " "
        for ch in text
    ]
    return re.sub(r"\s+", " ", "".join(chars)).strip()


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        f = float(value)
        return None if (math.isnan(f) or math.isinf(f)) else f
    return None


def _match_number(pred: Any, ref: Any) -> bool:
    p = _to_float(pred)
    r = _to_float(ref)
    if p is None or r is None:
        return False
    return round(p, 2) == round(r, 2)


def _match_percent(pred: Any, ref: Any) -> bool:
    p = _to_float(pred)
    r = _to_float(ref)
    if p is None or r is None:
        return False
    return round(p, 2) == round(r, 2)


def _match_string(pred: Any, ref: Any, aliases: list[str] | None = None) -> bool:
    if pred is None or ref is None:
        return False
    norm_pred = normalize_string(pred)
    accepted = [normalize_string(ref)]
    if aliases:
        accepted.extend(normalize_string(a) for a in aliases)
    return norm_pred in accepted


def _match_list(pred: Any, ref: Any) -> bool:
    if not isinstance(pred, list) or not isinstance(ref, list):
        return False
    if len(pred) != len(ref):
        return False
    for p_item, r_item in zip(pred, ref):
        if isinstance(r_item, (int, float)) and not isinstance(r_item, bool):
            if not _match_number(p_item, r_item):
                return False
        else:
            if not _match_string(p_item, r_item):
                return False
    return True


def _match_cell(pred: Any, ref: Any) -> bool:
    if isinstance(ref, (int, float)) and not isinstance(ref, bool):
        return _match_number(pred, ref)
    return _match_string(pred, ref)


def _match_table(pred: Any, ref: Any) -> bool:
    if not isinstance(pred, dict) or not isinstance(ref, dict):
        return False
    p_cols = pred.get("columns")
    r_cols = ref.get("columns")
    p_rows = pred.get("rows")
    r_rows = ref.get("rows")
    if not (
        isinstance(p_cols, list)
        and isinstance(r_cols, list)
        and isinstance(p_rows, list)
        and isinstance(r_rows, list)
    ):
        return False
    if len(p_cols) != len(r_cols) or len(p_rows) != len(r_rows):
        return False

    # Match columns by normalized name; if names do not match, fall back to position.
    p_norm_cols = [normalize_string(c) for c in p_cols]
    r_norm_cols = [normalize_string(c) for c in r_cols]
    if set(p_norm_cols) == set(r_norm_cols) and len(set(p_norm_cols)) == len(p_norm_cols):
        col_order = [p_norm_cols.index(c) for c in r_norm_cols]
    else:
        col_order = list(range(len(r_cols)))

    # Reorder predicted rows to match reference column order, then match rows as a set.
    used = [False] * len(p_rows)
    for r_row in r_rows:
        if not isinstance(r_row, list) or len(r_row) != len(r_cols):
            return False
        matched = False
        for idx, p_row in enumerate(p_rows):
            if used[idx] or not isinstance(p_row, list) or len(p_row) != len(p_cols):
                continue
            reordered_p = [p_row[col_order[j]] for j in range(len(r_cols))]
            if all(_match_cell(reordered_p[j], r_row[j]) for j in range(len(r_cols))):
                used[idx] = True
                matched = True
                break
        if not matched:
            return False
    return True


def score_question(pred: dict[str, Any] | None, ref: dict[str, Any]) -> tuple[int, str]:
    """Score a single prediction against the reference answer.

    Returns:
        (score, status) where score is 1 (correct), 0 (abstain), or -1 (wrong).
    """
    if pred is None or pred.get("type") == "abstain":
        return 0, "abstain"

    pred_type = pred.get("type")
    ref_type = ref.get("type")
    if pred_type != ref_type:
        return -1, "wrong"

    if ref_type == "unanswerable":
        return 1, "correct"

    p_ans = pred.get("answer")
    r_ans = ref.get("answer")

    if ref_type == "number":
        ok = _match_number(p_ans, r_ans)
    elif ref_type == "percent":
        ok = _match_percent(p_ans, r_ans)
    elif ref_type == "string":
        ok = _match_string(p_ans, r_ans, ref.get("alias"))
    elif ref_type == "date":
        ok = isinstance(p_ans, str) and isinstance(r_ans, str) and p_ans.strip() == r_ans.strip()
    elif ref_type == "boolean":
        ok = isinstance(p_ans, bool) and isinstance(r_ans, bool) and p_ans is r_ans
    elif ref_type == "list":
        ok = _match_list(p_ans, r_ans)
    elif ref_type == "table":
        ok = _match_table(p_ans, r_ans)
    else:
        ok = False

    return (1, "correct") if ok else (-1, "wrong")

