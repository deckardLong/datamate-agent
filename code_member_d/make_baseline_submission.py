"""Create and validate a baseline submission.jsonl from a questions.jsonl.

Default mode writes {"id": ..., "type": "abstain"} for every question. That
scores PA = 0 and is only meant to verify the upload format end to end.

Usage:
    python -m code_member_d.make_baseline_submission \
        --questions public_test/questions.jsonl --out outputs/submission.jsonl
    python -m code_member_d.make_baseline_submission \
        --questions public_test/questions.jsonl --validate outputs/submission.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

VALID_TYPES = {
    "number", "percent", "string", "date", "boolean",
    "list", "table", "unanswerable", "abstain",
}


def read_question_ids(path: Path) -> list[str]:
    ids: list[str] = []
    with path.open(encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            qid = json.loads(line).get("id")
            if not isinstance(qid, str) or not qid:
                raise ValueError(f"questions line {number}: missing string 'id'")
            ids.append(qid)
    if len(ids) != len(set(ids)):
        raise ValueError("questions.jsonl has duplicate ids")
    return ids


def write_abstain_submission(ids: list[str], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        for qid in ids:
            handle.write(json.dumps({"id": qid, "type": "abstain"}, ensure_ascii=False) + "\n")


def _check_answer(row: dict) -> str | None:
    kind = row["type"]
    if kind in {"unanswerable", "abstain"}:
        return None
    if "answer" not in row:
        return "missing 'answer'"
    answer = row["answer"]
    if kind in {"number", "percent"}:
        if isinstance(answer, bool) or not isinstance(answer, (int, float)):
            return f"{kind} answer must be numeric"
    elif kind in {"string", "date"}:
        if not isinstance(answer, str):
            return f"{kind} answer must be a string"
    elif kind == "boolean":
        if not isinstance(answer, bool):
            return "boolean answer must be true/false"
    elif kind == "list":
        if not isinstance(answer, list):
            return "list answer must be a list"
    elif kind == "table":
        if not (
            isinstance(answer, dict)
            and isinstance(answer.get("columns"), list)
            and isinstance(answer.get("rows"), list)
        ):
            return "table answer needs 'columns' and 'rows' lists"
    return None


def validate_submission(path: Path, question_ids: list[str]) -> list[str]:
    """Return a list of problems; empty means the file is valid."""
    problems: list[str] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                problems.append(f"line {number}: invalid JSON ({exc.msg})")
                continue
            qid = row.get("id")
            if qid in seen:
                problems.append(f"line {number}: duplicate id {qid!r}")
            seen.add(qid)
            if row.get("type") not in VALID_TYPES:
                problems.append(f"line {number}: bad type {row.get('type')!r}")
                continue
            issue = _check_answer(row)
            if issue:
                problems.append(f"line {number} ({qid}): {issue}")
    expected = set(question_ids)
    if seen - expected:
        problems.append(f"{len(seen - expected)} ids are not in questions.jsonl")
    if expected - seen:
        problems.append(f"{len(expected - seen)} question ids missing (count as abstain)")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", required=True, type=Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--out", type=Path, help="write an all-abstain baseline here")
    group.add_argument("--validate", type=Path, help="validate an existing submission")
    parser.add_argument(
        "--merge-from", nargs="+", type=Path,
        help="with --out: merge these shard submissions (later files win); "
             "questions without an answer become abstain",
    )
    args = parser.parse_args(argv)

    ids = read_question_ids(args.questions)
    if args.merge_from:
        if not args.out:
            parser.error("--merge-from needs --out")
        merged: dict[str, dict] = {}
        for path in args.merge_from:
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        row = json.loads(line)
                        merged[row["id"]] = row
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8", newline="\n") as handle:
            for qid in ids:
                row = merged.get(qid, {"id": qid, "type": "abstain"})
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        answered = sum(1 for q in ids if q in merged and merged[q]["type"] != "abstain")
        print(f"Merged {len(args.merge_from)} files -> {args.out}: "
              f"{answered}/{len(ids)} answered, rest abstain")
        return 0
    if args.out:
        write_abstain_submission(ids, args.out)
        print(f"Wrote {len(ids)} abstain rows to {args.out}")
        return 0
    problems = validate_submission(args.validate, ids)
    if problems:
        print("INVALID:")
        for problem in problems[:50]:
            print(" -", problem)
        return 1
    print(f"OK: {len(ids)} questions, submission is valid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
