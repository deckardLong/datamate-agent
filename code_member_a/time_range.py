"""Module thời gian (không dùng date.today())."""
from __future__ import annotations

import calendar
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd
from unidecode import unidecode

DateRange = tuple[str, str]
TRUTHY = {"1", "true", "yes", "y", "co", "x", "t", "1.0"}
_ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4}


def fold(v: Any) -> str:
    return unidecode(str(v)).lower().strip()


def _parse_date_fmt(v: Any) -> tuple[Optional[date], Optional[str]]:
    # Dùng nguyên bản logic parse_date_fmt từ pipeline gốc
    from .pipeline import _parse_date_fmt as _pdf
    return _pdf(v)


def _as_date(x: Union[str, date, datetime]) -> date:
    if isinstance(x, datetime):
        return x.date()
    if isinstance(x, date):
        return x
    d, _ = _parse_date_fmt(str(x))
    if d is None:
        raise ValueError(f"as_of_date không hợp lệ: {x!r}")
    return d


def _month_bounds(y: int, m: int) -> tuple[date, date]:
    return date(y, m, 1), date(y, m, calendar.monthrange(y, m)[1])


def _quarter_bounds(y: int, q: int) -> tuple[date, date]:
    sm = 3 * (q - 1) + 1
    return date(y, sm, 1), _month_bounds(y, sm + 2)[1]


def _to_date_clip(s: date, e: date, as_of: date) -> tuple[date, date]:
    return (s, as_of) if s <= as_of < e else (s, e)


def _shift_year(d: date, n: int, month_end: bool = False) -> date:
    y = d.year - n
    if month_end:
        return date(y, d.month, calendar.monthrange(y, d.month)[1])
    try:
        return d.replace(year=y)
    except ValueError:
        return d.replace(year=y, day=28)


def _truthy(v: Any) -> bool:
    if isinstance(v, (int, float, pd.NA)):
        return bool(v > 0) if pd.notna(v) else False
    return fold(v) in TRUTHY


def load_tet_windows(calendar_path: Path) -> list[tuple[date, date]]:
    if calendar_path is None or not Path(calendar_path).exists():
        raise FileNotFoundError(f"Không tìm thấy calendar.csv: {calendar_path}")
    df = pd.read_csv(calendar_path, dtype=str, keep_default_na=False)
    cols = {c: fold(c) for c in df.columns}
    start_c = next((c for c, f in cols.items() if f in ("start_date", "start", "tu_ngay", "begin_date")), None)
    end_c = next((c for c, f in cols.items() if f in ("end_date", "end", "den_ngay", "finish_date")), None)
    flag_cols = [c for c, f in cols.items() if "tet" in f]
    text_cols = [c for c, f in cols.items() if re.search(r"event|holiday|name|note|ten|su_kien|ghi_chu", f)]

    def row_is_tet(r) -> bool:
        if any(_truthy(r[c]) for c in flag_cols):
            return True
        return any("tet" in fold(r[c]) for c in text_cols)

    mask = df.apply(row_is_tet, axis=1) if len(df) else pd.Series([], dtype=bool)
    if start_c and end_c:
        wins = []
        for _, r in df[mask].iterrows():
            s, e = _parse_date_fmt(r[start_c])[0], _parse_date_fmt(r[end_c])[0]
            if s and e:
                wins.append((s, e))
        return sorted(set(wins))
    date_c = next((c for c, f in cols.items() if f in ("date", "ngay", "day", "ngay_duong")), None) \
        or next((c for c, f in cols.items() if ("date" in f or "ngay" in f)), None)
    if date_c is None:
        raise ValueError("calendar.csv không có cột ngày nhận diện được")
    days = sorted({d for d in (_parse_date_fmt(v)[0] for v in df.loc[mask, date_c]) if d})
    wins, run = [], []
    for d in days:
        if run and (d - run[-1]).days > 7:
            wins.append((run[0], run[-1]))
            run = []
        run.append(d)
    if run:
        wins.append((run[0], run[-1]))
    return wins


def _year_ref(e: str, as_of: date) -> Optional[int]:
    m = re.search(r"\bnam\s*(\d{4})\b|\b(\d{4})\b", e)
    if m:
        return int(m[1] or m[2])
    if re.search(r"nam (ngoai|truoc|roi)", e):
        return as_of.year - 1
    if re.search(r"nam nay", e):
        return as_of.year
    return None


def _resolve_base(e: str, as_of: date, calendar_path: Optional[Path]) -> tuple[date, date]:
    m = re.search(r"(\d+)\s*ngay\s*(qua|gan day|truoc)", e)
    if m:
        n = int(m[1])
        return as_of - timedelta(days=n - 1), as_of
    if "hom nay" in e or e in ("today", "nay"):
        return as_of, as_of
    if "hom qua" in e or "yesterday" in e:
        d = as_of - timedelta(days=1)
        return d, d
    if "hom kia" in e:
        d = as_of - timedelta(days=2)
        return d, d
    m = re.search(r"\b(\d{4}-\d{2}-\d{2})\b|\b(\d{1,2}/\d{1,2}/\d{4})\b", e)
    if m:
        d = _parse_date_fmt(m[0])[0]
        if d:
            return d, d

    if re.search(r"\btet\b", e):
        wins = load_tet_windows(calendar_path)  # type: ignore[arg-type]
        if not wins:
            raise ValueError("calendar.csv không có cửa sổ Tết")
        yr = _year_ref(e, as_of)
        if yr is not None:
            cand = [w for w in wins if w[0].year == yr]
            if not cand:
                raise ValueError(f"calendar.csv không có Tết năm {yr}")
            return cand[0]
        past = [w for w in wins if w[0] <= as_of]
        if not past:
            raise ValueError("không có Tết nào trước as_of_date")
        return past[-1]

    m = re.search(r"quy\s*(iv|iii|ii|i|\d)\b", e)
    if m or re.search(r"quy (nay|truoc|roi)", e):
        cq = (as_of.month - 1) // 3 + 1
        if m:
            q = _ROMAN.get(m[1]) or int(m[1])
            yr = _year_ref(e, as_of)
            y = yr if yr is not None else as_of.year
            s, en = _quarter_bounds(y, q)
            if yr is None and s > as_of:
                s, en = _quarter_bounds(y - 1, q)
        elif "nay" in e:
            s, en = _quarter_bounds(as_of.year, cq)
        else:
            y, q = (as_of.year, cq - 1) if cq > 1 else (as_of.year - 1, 4)
            s, en = _quarter_bounds(y, q)
        return _to_date_clip(s, en, as_of)

    if re.search(r"thang (nay)\b", e):
        s, en = _month_bounds(as_of.year, as_of.month)
        return _to_date_clip(s, en, as_of)
    if re.search(r"thang (truoc|roi|vua roi)", e):
        y, mo = (as_of.year, as_of.month - 1) if as_of.month > 1 else (as_of.year - 1, 12)
        return _month_bounds(y, mo)
    m = re.search(r"thang\s*(\d{1,2})\b", e)
    if m and 1 <= int(m[1]) <= 12:
        mo = int(m[1])
        yr = _year_ref(re.sub(r"thang\s*\d{1,2}", "", e), as_of)
        y = yr if yr is not None else as_of.year
        s, en = _month_bounds(y, mo)
        if yr is None and s > as_of:
            s, en = _month_bounds(y - 1, mo)
        return _to_date_clip(s, en, as_of)

    if re.search(r"tuan nay|this week", e):
        s = as_of - timedelta(days=as_of.weekday())
        return s, as_of
    if re.search(r"tuan (truoc|roi|vua roi)|last week", e):
        s = as_of - timedelta(days=as_of.weekday() + 7)
        return s, s + timedelta(days=6)

    if re.search(r"nam nay", e) and not re.search(r"\d{4}", e):
        return _to_date_clip(date(as_of.year, 1, 1), date(as_of.year, 12, 31), as_of)
    if re.search(r"nam (ngoai|truoc|roi)", e) and not re.search(r"\d{4}", e):
        y = as_of.year - 1
        return date(y, 1, 1), date(y, 12, 31)
    m = re.search(r"\bnam\s*(\d{4})\b", e)
    if m:
        y = int(m[1])
        return _to_date_clip(date(y, 1, 1), date(y, 12, 31), as_of)
    raise ValueError(f"Không nhận diện được biểu thức thời gian: {e!r}")


def resolve_time_range(expression: str, as_of_date: Union[str, date], calendar_path: Optional[Path] = None) -> DateRange:
    as_of = _as_date(as_of_date)
    e = fold(expression)
    e = re.sub(r"\s+", " ", e)
    if re.search(r"cung ky|same period", e):
        n = 2 if re.search(r"(2|hai) nam", e) else 1
        base_e = re.sub(r"cung ky( nam (ngoai|truoc))?|same period( last year)?|nam (ngoai|truoc)", "", e).strip()
        base_e = re.sub(r"(so voi|vs\.?|voi)\s*$", "", base_e).strip() or "thang nay"
        s, en = _resolve_base(base_e, as_of, calendar_path)
        month_end = en.day == calendar.monthrange(en.year, en.month)[1]
        s2, e2 = _shift_year(s, n), _shift_year(en, n, month_end=month_end)
        return s2.isoformat(), e2.isoformat()
    s, en = _resolve_base(e, as_of, calendar_path)
    return s.isoformat(), en.isoformat()


class TimeModule:
    def __init__(self, calendar_csv_path: Optional[Path], as_of_date: Union[str, date]):
        self.as_of_date = _as_date(as_of_date)
        self.calendar_path = Path(calendar_csv_path) if calendar_csv_path else None

    def resolve(self, expression: str) -> DateRange:
        return resolve_time_range(expression, self.as_of_date, self.calendar_path)
