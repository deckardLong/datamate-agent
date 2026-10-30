from __future__ import annotations

import json

from pathlib import Path
from typing import Union

import duckdb

from .core import CARD_FILENAME, DB_FILENAME, EXCLUDED_BUSINESSES, PARQUET_DIRNAME, PROFILE_FILENAME


def load_business_context(split: str, business_id: str, cache_root: Union[str, Path] = "cache") -> dict:
    """context = load_business_context("train", "biz_101")
    -> {"db_path": Path, "profile": dict, "schema_card": str, "rules": dict, "calendar_path": Path|None, ...}"""
    split_dir = Path(cache_root) / split
    biz = split_dir / business_id
    if business_id in EXCLUDED_BUSINESSES:
        raise ValueError(f"{business_id} đã bị loại khỏi dataset (trùng dữ liệu)")
    if not (biz / DB_FILENAME).exists():
        raise FileNotFoundError(f"Chưa có cache cho {split}/{business_id}: {biz / DB_FILENAME}. Chạy pipeline trước.")
    rules_p, cal_p = split_dir / "rules.json", split_dir / "calendar.csv"
    return {
        "business_id": business_id,
        "db_path": biz / DB_FILENAME,
        "parquet_dir": biz / PARQUET_DIRNAME,
        "profile": json.loads((biz / PROFILE_FILENAME).read_text(encoding="utf-8")),
        "schema_card": (biz / CARD_FILENAME).read_text(encoding="utf-8"),
        "rules": json.loads(rules_p.read_text(encoding="utf-8")) if rules_p.exists() else {},
        "calendar_path": cal_p if cal_p.exists() else None,
    }


def open_db(db_path: Union[str, Path]) -> duckdb.DuckDBPyConnection:
    """Mở DuckDB chỉ-đọc (an toàn khi nhiều worker/Docker đọc cùng lúc)."""
    return duckdb.connect(str(db_path), read_only=True)
