from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional

from .core import DataMateDataPipeline


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="DataMate data pipeline")
    ap.add_argument("--split", default="train")
    ap.add_argument("--data-dir", required=True, help="thư mục chứa các business (hoặc thư mục cha chứa data/)")
    ap.add_argument("--cache-root", default="cache")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--only", nargs="*", help="chỉ chạy các business này (vd: biz_101)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = DataMateDataPipeline(a.split, a.data_dir, a.cache_root, a.workers, a.shards, force=a.force)
    m = p.run(only=a.only)
    return 1 if m["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
