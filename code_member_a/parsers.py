from __future__ import annotations

import math
import re
from collections import Counter
from datetime import date, datetime, timedelta
from typing import Any, Optional

import numpy as np
import pandas as pd

from .utils import NULL_TOKENS, fold

_RE_ISO = re.compile(r"^(\d{4})([-/.])(\d{1,2})[-/.](\d{1,2})$")
_RE_DMY = re.compile(r"^(\d{1,2})([-/.])(\d{1,2})[-/.](\d{4})$")
_RE_DMY2 = re.compile(r"^(\d{1,2})([-/.])(\d{1,2})[-/.](\d{2})$")
_RE_VI = re.compile(r"^(\d{1,2})\s*(?:thg|thang|th)\.?\s*(\d{1,2})\s*(?:,|nam)?\s*(\d{4})$")
_RE_TIME_TAIL = re.compile(r"[t\s]\d{1,2}:\d{2}(:\d{2})?(\.\d+)?(z|[+-]\d{2}:?\d{2})?$")


def _parse_date_fmt(v: Any) -> tuple[Optional[date], Optional[str]]:
    if v is None:
        return None, None
    if isinstance(v, float) and math.isnan(v):
        return None, None
    if isinstance(v, pd.Timestamp):
        return (None, None) if pd.isna(v) else (v.date(), "native")
    if isinstance(v, datetime):
        return v.date(), "native"
    if isinstance(v, date):
        return v, "native"
    s = fold(v)
    if s in NULL_TOKENS:
        return None, None
    s = re.sub(r"^(ngay|date)\s+", "", s)
    s = _RE_TIME_TAIL.sub("", s).strip()
    try:
        m = _RE_ISO.match(s)
        if m:
            y, sep, mo, d = int(m[1]), m[2], int(m[3]), int(m[4])
            return date(y, mo, d), f"yyyy{sep}mm{sep}dd"
        m = _RE_DMY.match(s)
        if m:
            d, sep, mo, y = int(m[1]), m[2], int(m[3]), int(m[4])
            if mo > 12 >= d:  # dữ liệu kiểu m/d/y lọt vào
                d, mo = mo, d
            return date(y, mo, d), f"dd{sep}mm{sep}yyyy"
        m = _RE_VI.match(s)
        if m:
            return date(int(m[3]), int(m[2]), int(m[1])), "d thg m yyyy"
        m = _RE_DMY2.match(s)
        if m:
            return date(2000 + int(m[4]), int(m[3]), int(m[1])), f"dd{m[2]}mm{m[2]}yy"
        if re.match(r"^\d{5}(\.\d+)?$", s) and 20000 <= float(s) <= 60000:
            return date(1899, 12, 30) + timedelta(days=int(float(s))), "excel_serial"
    except ValueError:
        return None, None
    return None, None


def parse_date_series(s: pd.Series) -> tuple[pd.Series, Counter]:
    uniq = pd.unique(s.dropna())
    cache: dict[Any, Optional[date]] = {}
    fmts: Counter = Counter()
    for u in uniq:
        d, f = _parse_date_fmt(u)
        cache[u] = d
        if f:
            fmts[f] += 1
    parsed = pd.to_datetime(s.map(lambda x: cache.get(x) if pd.notna(x) else None), errors="coerce")
    return parsed, fmts


_RE_NUM = re.compile(r"^([\(\-\+−]?)\s*(\d[\d.,\s]*)\s*([a-z$₫\s]{0,10}?)\s*\)?$")
# Hậu tố nhân (chỉ áp dụng cho cột tiền): 25k = 25.000; 1,5tr = 1.500.000
MONEY_MULTIPLIERS = {"k": 1e3, "nghin": 1e3, "ngan": 1e3, "tr": 1e6, "trieu": 1e6, "m": 1e6, "ty": 1e9, "ti": 1e9}
_CURRENCY_SUFFIX = {"", "d", "vnd", "vnđ", "dong", "usd", "$"}


def _resolve_separators(core: str) -> Optional[float]:
    core = re.sub(r"\s+", "", core).strip(".,")
    has_c, has_d = "," in core, "." in core
    try:
        if has_c and has_d:
            dec = "," if core.rfind(",") > core.rfind(".") else "."
            thou = "." if dec == "," else ","
            return float(core.replace(thou, "").replace(dec, "."))
        if has_c or has_d:
            sep = "," if has_c else "."
            parts = core.split(sep)
            if len(parts) > 2:
                return float("".join(parts))
            if len(parts[1]) == 3 and parts[0] not in ("0", ""):
                return float("".join(parts))  # 560.000 / 560,000 -> nghìn
            return float(parts[0] + "." + parts[1])
        return float(core)
    except ValueError:
        return None


def _split_suffix(suffix: str) -> tuple[float, str]:
    """'k' -> (1000, ''), 'k vnd' -> (1000, 'vnd'), 'trd' -> (1e6, 'd'), 'qua' -> (1, 'qua')."""
    suf = re.sub(r"\s+", "", suffix)
    for key in sorted(MONEY_MULTIPLIERS, key=len, reverse=True):
        if suf.startswith(key) and suf[len(key):] in _CURRENCY_SUFFIX:
            return MONEY_MULTIPLIERS[key], suf[len(key):]
    return 1.0, suf


def parse_number(v: Any, money: bool = False) -> float:
    if v is None:
        return np.nan
    if isinstance(v, (bool, np.bool_)):
        return np.nan
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    s = fold(v)
    if s in NULL_TOKENS:
        return np.nan
    s = re.sub(r"^(vnd|vn|usd)\s*", "", s).replace("₫", "d")
    s = s.lstrip("$")
    m = _RE_NUM.match(s)
    if not m:
        return np.nan
    mult = 1.0
    if money:
        mult, rest = _split_suffix(m[3])
        if rest not in _CURRENCY_SUFFIX:
            return np.nan  # "25 ly", "abc" trong cột tiền -> không phải số tiền
        if mult > 1:
            # với hậu tố k/tr, dấu . hoặc , là dấu thập phân: 1.5tr, 1,5tr, 2.25k
            core = re.sub(r"\s+", "", m[2]).strip(".,")
            # k: '1.5k' thập phân, '1.500k' hàng nghìn. tr/tỷ: 1 dấu phân cách luôn là thập phân ('332,5972tr', '1,250tr')
            if re.fullmatch(r"\d+[.,]\d{1,2}", core) or (mult >= 1e6 and re.fullmatch(r"\d+[.,]\d+", core)):
                num = float(core.replace(",", "."))
                neg = m[1] in ("-", "−") or (m[1] == "(" and s.endswith(")"))
                return (-num if neg else num) * mult
    num = _resolve_separators(m[2])
    if num is None:
        return np.nan
    neg = m[1] in ("-", "−") or (m[1] == "(" and s.endswith(")"))
    num *= mult
    return -num if neg else num


def parse_number_series(s: pd.Series, money: bool = False) -> pd.Series:
    uniq = pd.unique(s.dropna())
    cache = {u: parse_number(u, money=money) for u in uniq}
    return pd.to_numeric(s.map(lambda x: cache.get(x, np.nan) if pd.notna(x) else np.nan), errors="coerce")


def money_style(raw_nn: pd.Series) -> str:
    """Kiểu ghi số của cột. Nếu cột trộn nhiều kiểu (1,234 lẫn 1.234, có hậu tố k/tr) thì trả về 'mixed(...)'."""
    sample = raw_nn.dropna().astype(str).str.strip().drop_duplicates().head(5000)
    folded = sample.map(fold)
    comma = int(sample.str.match(r"^\D*\d{1,3}(,\d{3})+(\.\d+)?\D*$").sum())
    dot = int(sample.str.match(r"^\D*\d{1,3}(\.\d{3})+(,\d+)?\D*$").sum())
    mult = int(folded.str.contains(r"\d\s*(?:k|nghin|ngan|tr|trieu|ty)\b").sum())
    cur = int(sample.str.contains(r"[a-zA-Zđ₫$]").sum())
    styles = [name for name, n in (("1,234", comma), ("1.234", dot), ("k/tr", mult)) if n]
    if len(styles) > 1:
        return "mixed(" + "|".join(styles) + ")" + (" + currency_text" if cur else "")
    base = styles[0] if styles else "plain"
    return base + (" + currency_text" if cur else "")


def sql_parse_expr(semantic: str, raw_fmt: str, raw_col: str) -> Optional[str]:
    """Biểu thức SQL tương đương để parse cột gốc (dùng khi LLM phải làm việc với cột raw)."""
    c = f'"{raw_col}"'
    if semantic == "date":
        if raw_fmt == "d thg m yyyy":
            rx = r"(\d{1,2})\s*thg\s*(\d{1,2})\s*(\d{4})"
            g = lambda i: f"CAST(regexp_extract({c}, '{rx}', {i}) AS INTEGER)"
            return f"make_date({g(3)}, {g(2)}, {g(1)})"
        if raw_fmt == "dd/mm/yyyy":
            return f"CAST(TRY_STRPTIME({c}, '%d/%m/%Y') AS DATE)"
        if raw_fmt.startswith("yyyy-mm-dd"):
            return f"TRY_CAST({c} AS DATE)"
        return None
    if semantic in ("money", "quantity"):
        if raw_fmt.startswith("mixed") or "k/tr" in raw_fmt:
            return None  # không có 1 biểu thức SQL đúng cho mọi dòng -> dùng cột đã chuẩn hoá
        if raw_fmt.startswith("1,234"):
            return f"TRY_CAST(replace(regexp_replace({c}, '[^0-9,.-]', '', 'g'), ',', '') AS DOUBLE)"
        if raw_fmt.startswith("1.234"):
            return (f"TRY_CAST(replace(replace(regexp_replace({c}, '[^0-9,.-]', '', 'g'), '.', ''), ',', '.') "
                    f"AS DOUBLE)")
        return f"TRY_CAST(regexp_replace({c}, '[^0-9.-]', '', 'g') AS DOUBLE)"
    return None
