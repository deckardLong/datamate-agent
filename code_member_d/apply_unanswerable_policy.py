"""Rewrite a finished run's submission with a different unanswerable policy, no GPU needed.

run_all.py's default policy is `abstain`: when the model says "unanswerable" the
row is written as abstain (0 points) because a wrong unanswerable costs -1. The
model's verdict and reason are kept in the per-question log, so a variant that
submits those verdicts can be rebuilt from the log alone:

    python -m code_member_d.apply_unanswerable_policy \
        --log submissions/logs/run1_shard1.jsonl --sub submissions/run1_shard1.jsonl \
        --out submissions/run1_shard1_keep.jsonl --policy keep
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from code_member_d.inference import clean_reason

_NOTE = "policy=abstain"


def model_unanswerable_reasons(log_path: Path) -> dict[str, str]:
    """id -> reason, for questions where the model said unanswerable."""
    found: dict[str, str] = {}
    with log_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            for attempt in record.get("attempts", []):
                if _NOTE in str(attempt.get("note", "")):
                    plan = attempt.get("plan") or {}
                    found[str(record["id"])] = clean_reason(plan.get("reason"))
    return found


def apply_policy(sub_path: Path, log_path: Path, out_path: Path, policy: str) -> tuple[int, int]:
    """Return (rows changed, rows total). Only rows the model called unanswerable change."""
    verdicts = model_unanswerable_reasons(log_path)
    changed = total = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with sub_path.open(encoding="utf-8") as src, \
            out_path.open("w", encoding="utf-8", newline="\n") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            total += 1
            if row["id"] in verdicts and row.get("type") in {"abstain", "unanswerable"}:
                wanted = (
                    {"id": row["id"], "type": "unanswerable", "reason": verdicts[row["id"]]}
                    if policy == "keep"
                    else {"id": row["id"], "type": "abstain"}
                )
                changed += wanted != row
                row = wanted
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
    return changed, total


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log", required=True, type=Path)
    p.add_argument("--sub", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--policy", choices=["keep", "abstain"], required=True)
    a = p.parse_args(argv)
    if a.out.resolve() in {a.sub.resolve(), a.log.resolve()}:
        p.error("--out must be a new file")
    changed, total = apply_policy(a.sub, a.log, a.out, a.policy)
    print(f"policy={a.policy}: {changed} of {total} rows changed -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
