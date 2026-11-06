"""Module chia shard theo questions.jsonl."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Optional

EXCLUDED_BUSINESSES = {"diz_114"}


def count_questions(questions_path: Optional[Path]) -> tuple[Counter, dict]:
    counts: Counter = Counter()
    info = {"total": 0, "bad_lines": 0, "missing_business_id": 0, "duplicate_question_ids": []}
    if not questions_path or not Path(questions_path).exists():
        return counts, info
    seen: set = set()
    with open(questions_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                q = json.loads(line)
            except json.JSONDecodeError:
                info["bad_lines"] += 1
                continue
            info["total"] += 1
            b = q.get("business_id")
            if not b:
                info["missing_business_id"] += 1
            else:
                counts[b] += 1
            qid = q.get("id", q.get("question_id"))
            if qid is not None:
                if qid in seen:
                    info["duplicate_question_ids"].append(qid)
                seen.add(qid)
    return counts, info


def generate_shards_config(cache_split_dir: Path, businesses: list[str], questions_path: Optional[Path],
                           output_path: Path, n_shards: int = 4, excluded: set[str] = EXCLUDED_BUSINESSES) -> dict:
    qcount, qinfo = count_questions(questions_path)
    sizes, rows = {}, {}
    for b in businesses:
        d = cache_split_dir / b
        sizes[b] = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 2**20 if d.exists() else 0.0
        try:
            rows[b] = json.loads((d / "profile.json").read_text(encoding="utf-8"))
            rows[b] = sum(t["row_count"] for t in rows[b]["tables"].values())
        except Exception:
            rows[b] = 0
    import numpy as np
    med_rows = float(np.median([r for r in rows.values() if r] or [1]))
    load = {}
    for b in businesses:
        q = qcount.get(b, 0)
        weight = 1.0 + rows[b] / med_rows
        load[b] = (q if q else 0.1) * weight
    shards = [{"shard": i + 1, "businesses": [], "questions": 0, "load": 0.0} for i in range(n_shards)]
    for b in sorted(businesses, key=lambda x: load[x], reverse=True):
        tgt = min(shards, key=lambda s: s["load"])
        tgt["businesses"].append({"business_id": b, "questions": qcount.get(b, 0), "size_mb": round(sizes[b], 3),
                                  "rows": rows[b], "load": round(load[b], 3)})
        tgt["questions"] += qcount.get(b, 0)
        tgt["load"] = round(tgt["load"] + load[b], 3)
    assigned = [x["business_id"] for s in shards for x in s["businesses"]]
    validation = {
        "all_businesses_assigned_once": sorted(assigned) == sorted(businesses) and len(set(assigned)) == len(assigned),
        "questions_for_unknown_or_excluded_business": {b: c for b, c in qcount.items() if b not in businesses},
        "excluded_businesses": sorted(excluded),
        "businesses_without_questions": [b for b in businesses if not qcount.get(b)],
        "questions_total_in_file": qinfo["total"],
        "questions_total_in_shards": sum(s["questions"] for s in shards),
        "questions_file_issues": {k: v for k, v in qinfo.items() if k != "total" and v},
        "load_imbalance_max_over_min": round(max(s["load"] for s in shards) / max(min(s["load"] for s in shards), 1e-9), 3),
    }
    result = {"n_shards": n_shards, "shards": shards, "validation": validation}
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
