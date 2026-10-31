from __future__ import annotations

import json

import pytest

from code_member_b.qlora_train import _encode_example, _read_rows, build_parser


def _row(row_id: str = "q1") -> dict:
    return {
        "id": row_id,
        "messages": [
            {"role": "user", "content": "Schema...\nQuestion: doanh thu?"},
            {"role": "assistant", "content": '{"sql":"SELECT 1"}'},
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
    assert encoded["input_ids"] == [2, 3, 4, 9, 10]
    assert encoded["labels"] == [-100, -100, -100, 9, 10]
    assert encoded["attention_mask"] == [1, 1, 1, 1, 1]


def test_qlora_defaults_are_t4_safe():
    args = build_parser().parse_args(["--data-path", "data.jsonl", "--output-dir", "out"])
    assert args.model_id == "Qwen/Qwen2.5-Coder-7B-Instruct"
    assert args.max_length == 4096
    assert args.gradient_accumulation_steps == 8
