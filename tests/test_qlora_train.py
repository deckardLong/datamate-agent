from __future__ import annotations

import json

import pytest

from code_member_b.qlora_train import (
    _balance_rows,
    _encode_example,
    _read_rows,
    _stratified_split,
    build_parser,
)


def _row(row_id: str = "q1", answer_type: str = "number") -> dict:
    return {
        "id": row_id,
        "messages": [
            {"role": "user", "content": "Schema...\nQuestion: doanh thu?"},
            {
                "role": "assistant",
                "content": json.dumps(
                    {"answer_type": answer_type, "sql": "SELECT 1"}
                ),
            },
        ],
    }


def test_read_rows_validates_targets_and_duplicate_ids(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text(
        "\n".join(json.dumps(_row("q1")) for _ in range(2)), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Duplicate example IDs"):
        _read_rows(path)

    invalid = _row("q2")
    invalid["messages"][1]["content"] = '{"answer": 1}'
    path.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid SQL target"):
        _read_rows(path)

    unanswerable = _row("q3")
    unanswerable["messages"][1]["content"] = '{"sql": null}'
    path.write_text(json.dumps(unanswerable), encoding="utf-8")
    assert _read_rows(path) == [unanswerable]


class FakeTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is True
        if len(messages) == 1:
            assert add_generation_prompt is True
            return [1, 2, 3, 4]
        assert add_generation_prompt is False
        return [1, 2, 3, 4, 9, 10]


def test_encode_example_masks_prompt_and_preserves_target_when_truncated():
    encoded = _encode_example(FakeTokenizer(), _row(), max_length=5)
    assert encoded["input_ids"] == [1, 2, 4, 9, 10]
    assert encoded["labels"] == [-100, -100, -100, 9, 10]
    assert encoded["attention_mask"] == [1, 1, 1, 1, 1]


def test_balancing_caps_unanswerable_and_split_is_stratified():
    rows = [_row(f"n{index}", "number") for index in range(20)]
    rows += [_row(f"u{index}", "unanswerable") for index in range(20)]
    for row in rows[20:]:
        row["messages"][1]["content"] = json.dumps(
            {"answer_type": "unanswerable", "sql": None}
        )

    balanced = _balance_rows(rows, max_unanswerable=5, seed=42)
    assert len(balanced) == 25
    training, validation = _stratified_split(balanced, 0.1, seed=42)
    assert len(validation) == 3
    validation_types = {
        json.loads(row["messages"][1]["content"])["answer_type"]
        for row in validation
    }
    assert validation_types == {"number", "unanswerable"}


def test_init_adapter_is_optional_and_parsed():
    base = ["--data-path", "d.jsonl", "--output-dir", "out"]
    assert build_parser().parse_args(base).init_adapter is None
    args = build_parser().parse_args(base + ["--init-adapter", "prev/final_adapter"])
    assert args.init_adapter == "prev/final_adapter"


def test_qlora_defaults_are_t4_safe():
    args = build_parser().parse_args(["--data-path", "data.jsonl", "--output-dir", "out"])
    assert args.model_id == "Qwen/Qwen2.5-Coder-7B-Instruct"
    assert args.max_length == 4096
    assert args.gradient_accumulation_steps == 8
    assert args.max_unanswerable == 20
    assert args.prompt_head_fraction == 0.6
