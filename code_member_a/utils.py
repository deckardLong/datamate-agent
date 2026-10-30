from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd
from unidecode import unidecode

NULL_TOKENS = {"", "nan", "none", "null", "nat", "n/a", "na", "-", "--"}


def fold(v: Any) -> str:
    """Bỏ dấu, lowercase, strip."""
    return unidecode(str(v)).lower().strip()


def clean_name(raw: Any, fallback: str = "col") -> str:
    text = re.sub(r"[^a-z0-9]+", "_", fold(raw)).strip("_")
    if not text:
        return fallback
    if text[0].isdigit():
        text = f"{fallback}_{text}"
    return text


def make_unique(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for n in names:
        if n in seen:
            seen[n] += 1
            out.append(f"{n}_{seen[n]}")
        else:
            seen[n] = 1
            out.append(n)
    return out


def has_kw(col: str, kws: set[str]) -> bool:
    """So khớp theo TOKEN (không phải chuỗi con): 'gia' khớp 'don_gia' nhưng không khớp 'giao_hang'."""
    padded = f"_{col}_"
    return any(f"_{k}_" in padded for k in kws)


def sanitize_value(val: Any) -> Any:
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(val, (pd.Timestamp, datetime, date)):
        return val.isoformat()[:10] if isinstance(val, date) and not isinstance(val, datetime) else val.isoformat()
    if isinstance(val, (pd.Timedelta, np.timedelta64)):
        return str(val)
    if isinstance(val, np.integer):
        return int(val)
    if isinstance(val, np.floating):
        return float(val)
    if isinstance(val, np.bool_):
        return bool(val)
    if isinstance(val, (bytes, bytearray)):
        return val.decode("utf-8", "replace")
    return val


def _is_textual(s: pd.Series) -> bool:
    return pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)
