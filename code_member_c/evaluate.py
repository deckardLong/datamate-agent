"""Evaluate a submission.jsonl against answers.jsonl and print PA + error breakdown."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from scorer import score_question


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def evaluate(
    answers_path: Path,
    submission_path: Path,
    questions_path: Path | None = None,
) -> dict[str, Any]:
    refs = load_jsonl(answers_path)
    preds_by_id = {r["id"]: r for r in load_jsonl(submission_path)}

    q_biz: dict[str, str] = {}
    if questions_path and questions_path.exists():
        for q in load_jsonl(questions_path):
            q_biz[q["id"]] = q.get("business_id", "unknown")

    counts = {"correct": 0, "wrong": 0, "abstain": 0}
    total_score = 0
    by_type: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "wrong": 0, "abstain": 0})
    by_biz: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "wrong": 0, "abstain": 0})

    for ref in refs:
        qid = ref["id"]
        pred = preds_by_id.get(qid)
        score, status = score_question(pred, ref)
        total_score += score
        counts[status] += 1
        by_type[ref["type"]][status] += 1
        if q_biz:
            by_biz[q_biz.get(qid, "unknown")][status] += 1

    n = len(refs)
    pa = total_score / n if n else 0.0
    return {
        "total": n,
        "pa": pa,
        "total_score": total_score,
        "counts": counts,
        "by_type": dict(by_type),
        "by_biz": dict(by_biz),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DataMate submission.jsonl")
    parser.add_argument("--answers", type=Path, required=True, help="Path to answers.jsonl")
    parser.add_argument("--submission", type=Path, required=True, help="Path to submission.jsonl")
    parser.add_argument("--questions", type=Path, default=None, help="Optional questions.jsonl for per-business breakdown")
    args = parser.parse_args()

    res = evaluate(args.answers, args.submission, args.questions)
    n = res["total"]
    c = res["counts"]
    print(f"Total questions : {n}")
    print(f"Penalized Acc   : {res['pa']:.4f} (score={res['total_score']})")
    print(f"Correct / Wrong / Abstain: {c['correct']} / {c['wrong']} / {c['abstain']}")

    print("\nBreakdown by reference type:")
    for t, st in sorted(res["by_type"].items()):
        t_n = st["correct"] + st["wrong"] + st["abstain"]
        t_pa = (st["correct"] - st["wrong"]) / t_n if t_n else 0.0
        print(f"  {t:13s} | n={t_n:4d} | PA={t_pa:+.3f} | +{st['correct']} / -{st['wrong']} / ~{st['abstain']}")

    if res["by_biz"]:
        print("\nLowest PA businesses (top 10):")
        biz_rows = []
        for b, st in res["by_biz"].items():
            b_n = st["correct"] + st["wrong"] + st["abstain"]
            b_pa = (st["correct"] - st["wrong"]) / b_n if b_n else 0.0
            biz_rows.append((b_pa, b, b_n, st))
        for b_pa, b, b_n, st in sorted(biz_rows)[:10]:
            print(f"  {b:12s} | n={b_n:3d} | PA={b_pa:+.3f} | +{st['correct']} / -{st['wrong']} / ~{st['abstain']}")


if __name__ == "__main__":
    main()

