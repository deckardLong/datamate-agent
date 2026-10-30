"""Command-line entry point for Colab pseudo-label generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pseudo_label import (
    PseudoLabelPipeline,
    export_qlora_jsonl,
    load_training_records,
    stratified_sample,
)
from .qwen_teacher import QwenTeacher


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/datamate-corpus")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", default="outputs/pseudo_labels/pilot_v1")
    parser.add_argument("--pilot-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--candidate-count", type=int, default=3)
    parser.add_argument(
        "--model-id", default="Qwen/Qwen2.5-Coder-7B-Instruct"
    )
    parser.add_argument("--max-input-tokens", type=int, default=12288)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    records = load_training_records(args.data_root, args.split)
    selected = stratified_sample(records, args.pilot_size, args.seed)
    print(
        f"Loaded {len(records)} records; selected {len(selected)} "
        f"with seed={args.seed}",
        flush=True,
    )
    teacher = QwenTeacher(
        model_id=args.model_id,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
    )
    pipeline = PseudoLabelPipeline(
        teacher,
        data_root=args.data_root,
        split=args.split,
        output_dir=args.output_dir,
        candidate_count=args.candidate_count,
        query_timeout_seconds=args.timeout_seconds,
    )
    report = pipeline.run(selected)
    output_dir = Path(args.output_dir)
    qlora_path = output_dir / "qlora_train.jsonl"
    exported = export_qlora_jsonl(pipeline.accepted_path, qlora_path)
    report["qlora_examples"] = exported
    report["qlora_path"] = str(qlora_path)
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
