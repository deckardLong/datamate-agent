"""Pick training-split questions the current adapter was NOT trained on, for honest scoring.

B trains on `qlora_train.jsonl` after capping unanswerable rows and a stratified
train/validation split (seed 42). This repeats exactly that selection and keeps only the
questions that were never in the training part, so C can measure the real accuracy of the
pipeline against the reference answers.

    python -m code_member_d.make_heldout --questions corpus/train/questions.jsonl \
        --answers corpus/train/answers.jsonl --qlora-train <path>/qlora_train.jsonl \
        --n 150 --out-dir outputs/heldout

Writes questions_heldout.jsonl (input of run_all.py) and answers_heldout.jsonl (the
reference answers of the same ids, for the scorer). Deterministic for a given seed.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path
from typing import Any

from code_member_b.qlora_train import _balance_rows, _read_rows, _stratified_split


def trained_ids(qlora_train: Path, seed: int = 42, validation_fraction: float = 0.1,
                max_unanswerable: int = 20) -> set[str]:
    """Ids of the examples that qlora_train.py really trains on (not validation, not capped)."""
    rows = _balance_rows(_read_rows(qlora_train), max_unanswerable, seed)
    train, _validation = _stratified_split(rows, validation_fraction, seed)
    return {row["id"] for row in train}


def _read(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def pick(questions: list[dict[str, Any]], answers: list[dict[str, Any]], excluded: set[str],
         n: int, seed: int = 7, min_unanswerable: int = 15) -> list[str]:
    """Ids of n unseen questions: every answer type represented in proportion to the data,
    at least `min_unanswerable` unanswerable ones (to measure that verdict), businesses
    spread evenly inside each type."""
    kind = {a["id"]: a["type"] for a in answers}
    business = {q["id"]: q["business_id"] for q in questions}
    pool = [q["id"] for q in questions if q["id"] in kind and q["id"] not in excluded]
    rng = random.Random(seed)
    by_type: dict[str, list[str]] = collections.defaultdict(list)
    for qid in pool:
        by_type[kind[qid]].append(qid)
    total = len(pool)
    quota = {t: max(1, round(n * len(ids) / total)) for t, ids in by_type.items()}
    if "unanswerable" in by_type:
        quota["unanswerable"] = min(len(by_type["unanswerable"]), max(quota["unanswerable"], min_unanswerable))
    chosen: list[str] = []
    for t, ids in by_type.items():
        per_business: dict[str, list[str]] = collections.defaultdict(list)
        for qid in ids:
            per_business[business[qid]].append(qid)
        for lst in per_business.values():
            rng.shuffle(lst)
        order = sorted(per_business)
        rng.shuffle(order)
        picked: list[str] = []
        while len(picked) < min(quota[t], len(ids)):
            for b in order:
                if per_business[b] and len(picked) < quota[t]:
                    picked.append(per_business[b].pop())
        chosen += picked
    return chosen


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--questions", required=True, type=Path)
    p.add_argument("--answers", required=True, type=Path)
    p.add_argument("--qlora-train", required=True, type=Path, help="B's qlora_train.jsonl (the 289 examples)")
    p.add_argument("--out-dir", default=Path("outputs/heldout"), type=Path)
    p.add_argument("--n", type=int, default=150)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--min-unanswerable", type=int, default=15)
    a = p.parse_args(argv)

    questions, answers = _read(a.questions), _read(a.answers)
    excluded = trained_ids(a.qlora_train)
    ids = set(pick(questions, answers, excluded, a.n, a.seed, a.min_unanswerable))
    a.out_dir.mkdir(parents=True, exist_ok=True)
    qs = [q for q in questions if q["id"] in ids]
    ans = [x for x in answers if x["id"] in ids]
    with (a.out_dir / "questions_heldout.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        f.writelines(json.dumps(q, ensure_ascii=False) + "\n" for q in qs)
    with (a.out_dir / "answers_heldout.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        f.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in ans)
    kinds = collections.Counter(x["type"] for x in ans)
    print(f"{len(excluded)} training ids excluded; wrote {len(qs)} held-out questions to {a.out_dir}")
    print("types:", dict(kinds), "| businesses:", len({q["business_id"] for q in qs}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
