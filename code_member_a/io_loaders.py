from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import pandas as pd

from .utils import clean_name, make_unique

log = logging.getLogger("datamate")


def _read_csv(path: Path) -> pd.DataFrame:
    last: Exception | None = None
    for enc in ("utf-8-sig", "cp1258", "latin-1"):
        try:
            df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""], encoding=enc)
            if df.shape[1] == 1 and ";" in str(df.columns[0]):
                df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""], encoding=enc, sep=";")
            return df
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
                sheets = pd.read_excel(p, sheet_name=None, dtype=str)
                for sh, d in sheets.items():
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
