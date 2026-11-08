from __future__ import annotations

import logging
import re
import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd

from .utils import NULL_TOKENS, clean_name, fold, make_unique

log = logging.getLogger("datamate")

HEADER_SCAN_ROWS = 20
_NUMERIC_LIKE = re.compile(r"^[\(\-\+−]?\s*[\d.,\s]+\s*[a-z$₫đ]{0,6}\)?$")
_DATE_LIKE = re.compile(r"^\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}")


def _is_blank(v) -> bool:
    if v is None:
        return True
    try:
        if pd.isna(v):
            return True
    except (TypeError, ValueError):
        pass
    return fold(v) in NULL_TOKENS


def _is_label(v) -> bool:
    """Ô trông giống tên cột: chuỗi chữ, không phải số / ngày / tiền."""
    if _is_blank(v):
        return False
    s = fold(v)
    if len(s) > 60 or not re.search(r"[a-z]", s):
        return False
    return not (_NUMERIC_LIKE.match(s) or _DATE_LIKE.match(s))


def _find_header_row(raw: pd.DataFrame) -> int:
    """Dòng có nhiều ô dạng nhãn nhất trong HEADER_SCAN_ROWS dòng đầu (hòa thì lấy dòng sớm nhất).

    Xử lý file Excel/CSV có dòng tiêu đề ("DANH SÁCH HÓA ĐƠN") hoặc header nhóm gộp ô ở trên header thật.
    """
    best_i, best_score = 0, -1
    for i in range(min(HEADER_SCAN_ROWS, len(raw))):
        score = sum(_is_label(v) for v in raw.iloc[i])
        if score > best_score:
            best_i, best_score = i, score
    return best_i


def _promote_header(raw: pd.DataFrame, source: str) -> pd.DataFrame:
    """Biến DataFrame đọc với header=None thành bảng có header đúng."""
    raw = raw.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)
    if raw.empty:
        return raw
    h = _find_header_row(raw)
    header = [None if _is_blank(v) else str(v).strip() for v in raw.iloc[h]]

    # Header 2 tầng (ô nhóm gộp ở dòng trên): ô trống ở header thật lấy tên nhóm;
    # tên trùng nhau thì ghép thêm tên nhóm phía trước.
    if h > 0:
        group, last = [], None
        for v in raw.iloc[h - 1]:
            last = str(v).strip() if not _is_blank(v) else last
            group.append(last)
        if sum(g is not None for g in group) >= 2:
            for j, name in enumerate(header):
                if name is None and group[j]:
                    header[j] = group[j]
            folded = [clean_name(x) if x else None for x in header]
            for j, name in enumerate(header):
                if name and folded.count(folded[j]) > 1 and group[j] and clean_name(group[j]) != folded[j]:
                    header[j] = f"{group[j]} {name}"

    data = raw.iloc[h + 1:].reset_index(drop=True)
    data.columns = [name if name else f"col_{j}" for j, name in enumerate(header)]
    # bỏ cột không tên và trống hoàn toàn
    keep = [c for j, c in enumerate(data.columns) if header[j] or data.iloc[:, j].notna().any()]
    data = data.loc[:, keep]
    # bỏ dòng header lặp lại giữa bảng (file ghép nhiều trang/sheet)
    named = [c for j, c in enumerate(data.columns) if header[j]]
    if named:
        rep = (data[named].astype(str).apply(lambda col: col.str.strip()) == pd.Series(named, index=named)).all(axis=1)
        data = data[~rep].reset_index(drop=True)
    if h > 0:
        log.info("%s: header ở dòng %d (bỏ %d dòng tiêu đề phía trên)", source, h + 1, h)
    return data


def _read_csv(path: Path) -> pd.DataFrame:
    last: Optional[Exception] = None
    for enc in ("utf-8-sig", "cp1258", "latin-1"):
        try:
            with open(path, encoding=enc) as fh:
                sample = fh.read(50_000)
            first_lines = [ln for ln in sample.splitlines()[:HEADER_SCAN_ROWS] if ln.strip()]
            sep = ";" if sum(ln.count(";") for ln in first_lines) > sum(ln.count(",") for ln in first_lines) else ","
            # Đặt sẵn số cột = dòng rộng nhất trong các dòng đầu: nếu không, pandas lấy độ rộng theo
            # dòng đầu tiên (thường là dòng tiêu đề 1 ô) và lỗi/cắt cột. Cột thừa (trống) bị bỏ ở _promote_header.
            width = max((ln.count(sep) for ln in first_lines), default=0) + 1
            df = pd.read_csv(path, sep=sep, engine="python", names=range(width), header=None, dtype=str,
                             keep_default_na=False, na_values=[""], encoding=enc, on_bad_lines="warn")
            return _promote_header(df, path.name)
        except UnicodeDecodeError as e:
            last = e
    raise last  # type: ignore[misc]


def _read_sqlite(path: Path) -> dict[str, pd.DataFrame]:
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        names = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
        return {n: pd.read_sql_query(f'SELECT * FROM "{n}"', con) for n in names}
    finally:
        con.close()


def load_sources(biz_dir: Path) -> list[tuple[str, pd.DataFrame, str]]:
    """Trả về [(table_name, df, source_file)]. Lỗi một file không làm hỏng cả business."""
    items: list[tuple[str, pd.DataFrame, str]] = []
    for p in sorted(biz_dir.iterdir()):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        try:
            if ext == ".csv" and p.name.lower() != "calendar.csv":
                items.append((clean_name(p.stem, "t"), _read_csv(p), p.name))
            elif ext in (".xlsx", ".xlsm", ".xls"):
                sheets = pd.read_excel(p, sheet_name=None, dtype=str, header=None)
                for sh, d in sheets.items():
                    d = _promote_header(d, f"{p.name}#{sh}")
                    if d.empty:
                        continue
                    name = clean_name(p.stem, "t") if len(sheets) == 1 else clean_name(f"{p.stem}_{sh}", "t")
                    items.append((name, d, f"{p.name}#{sh}"))
            elif ext in (".sqlite", ".sqlite3", ".db"):
                for t, d in _read_sqlite(p).items():
                    items.append((clean_name(f"{p.stem}_{t}", "t"), d, f"{p.name}#{t}"))
        except Exception as e:  # noqa: BLE001
            log.error("[%s] không đọc được %s: %s", biz_dir.name, p.name, e)
    # tên bảng duy nhất
    names = make_unique([n for n, _, _ in items])
    return [(nm, d, src) for nm, (_, d, src) in zip(names, items)]
