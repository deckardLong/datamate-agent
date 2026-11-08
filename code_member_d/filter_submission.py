"""Turn answers that are almost certainly wrong (0, empty list, empty date...) into abstain.

Applies the same check run_all.py uses (inference.suspicious_reason) to a submission that
was produced earlier, so no GPU re-run is needed:

    python -m code_member_d.filter_submission --sub outputs/local_run3/final.jsonl \
        --out outputs/local_run3/final_filtered.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

from code_member_d.inference import suspicious_reason


def filter_rows(sub_path: Path, out_path: Path) -> tuple[int, int, collections.Counter]:
    """Return (rows turned into abstain, answered rows before, reasons)."""
    changed = answered = 0
    reasons: collections.Counter = collections.Counter()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with sub_path.open(encoding="utf-8") as src, out_path.open("w", encoding="utf-8", newline="\n") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("type") not in {"abstain", "unanswerable"}:
                answered += 1
                why = suspicious_reason(row["type"], row.get("answer"))
                if why:
                    changed += 1
                    reasons[why.split(";")[0]] += 1
                    row = {"id": row["id"], "type": "abstain"}
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
    return changed, answered, reasons


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sub", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args(argv)
    if a.out.resolve() == a.sub.resolve():
        p.error("--out must be a new file")
    changed, answered, reasons = filter_rows(a.sub, a.out)
    print(f"{answered} answered rows -> {answered - changed} kept, {changed} turned into abstain -> {a.out}")
    for why, n in reasons.most_common():
        print(f"  {n:3d}  {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
