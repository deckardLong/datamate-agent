"""T4-safe QLoRA fine-tuning for verified DataMate chat examples."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrainConfig:
    data_path: str
    output_dir: str
    model_id: str
    max_length: int
    epochs: float
    learning_rate: float
    gradient_accumulation_steps: int
    seed: int
    validation_fraction: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--model-id", default="Qwen/Qwen2.5-Coder-7B-Instruct"
    )
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore an existing checkpoint and start a new run.",
    )
    return parser


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            messages = value.get("messages")
            if (
                not isinstance(messages, list)
                or len(messages) != 2
                or messages[0].get("role") != "user"
                or messages[1].get("role") != "assistant"
            ):
                raise ValueError(f"Invalid messages at line {line_number}")
            # Ensure every target is valid JSON before spending GPU time.
            target = json.loads(str(messages[1].get("content", "")))
            if (
                not isinstance(target, dict)
                or "sql" not in target
                or target["sql"] is not None
                and not isinstance(target["sql"], str)
            ):
                raise ValueError(f"Invalid SQL target at line {line_number}")
            rows.append(value)
    if not rows:
        raise ValueError(f"No training examples in {path}")
    ids = [row.get("id") for row in rows]
    if any(not isinstance(row_id, str) or not row_id.strip() for row_id in ids):
        raise ValueError("Every example must have a non-empty string ID")
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate example IDs")
    return rows


def _encode_example(tokenizer: Any, row: dict[str, Any], max_length: int) -> dict:
    user_message, assistant_message = row["messages"]
    prompt_ids = tokenizer.apply_chat_template(
        [user_message],
        tokenize=True,
        add_generation_prompt=True,
    )
    full_ids = tokenizer.apply_chat_template(
        [user_message, assistant_message],
        tokenize=True,
        add_generation_prompt=False,
    )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("Tokenizer chat template does not preserve the prompt prefix")

    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    if len(full_ids) > max_length:
        # The schema is at the beginning; the question and SQL target are at
        # the end. Left truncation therefore preserves every supervised token.
        cut = len(full_ids) - max_length
        full_ids = full_ids[cut:]
        labels = labels[cut:]
    if not any(label != -100 for label in labels):
        raise ValueError(f"SQL target was truncated for example {row.get('id')}")
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


class EncodedDataset:
    def __init__(self, items: list[dict[str, Any]]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.items[index]


class CompletionCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        width = max(len(feature["input_ids"]) for feature in features)
        result = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            padding = width - len(feature["input_ids"])
            result["input_ids"].append(
                feature["input_ids"] + [self.pad_token_id] * padding
            )
            result["attention_mask"].append(
                feature["attention_mask"] + [0] * padding
            )
            result["labels"].append(feature["labels"] + [-100] * padding)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in result.items()}


def main() -> None:
    args = build_parser().parse_args()
    if args.max_length < 512:
        raise ValueError("--max-length must be at least 512")
    if not 0 < args.validation_fraction < 0.5:
        raise ValueError("--validation-fraction must be between 0 and 0.5")

    try:
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            Trainer,
            TrainingArguments,
        )
        from transformers.trainer_utils import get_last_checkpoint
    except ImportError as exc:
        raise RuntimeError(
            "Install: pip install -r code_member_b/requirements-colab.txt"
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA training requires a CUDA GPU")

    config = TrainConfig(
        data_path=args.data_path,
        output_dir=args.output_dir,
        model_id=args.model_id,
        max_length=args.max_length,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
    )
    rows = _read_rows(Path(args.data_path))
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    validation_size = max(1, round(len(rows) * args.validation_fraction))
    validation_rows = rows[:validation_size]
    training_rows = rows[validation_size:]

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    train_items = [
        _encode_example(tokenizer, row, args.max_length) for row in training_rows
    ]
    validation_items = [
        _encode_example(tokenizer, row, args.max_length) for row in validation_rows
    ]
    lengths = [len(item["input_ids"]) for item in train_items + validation_items]
    target_lengths = [
        sum(label != -100 for label in item["labels"])
        for item in train_items + validation_items
    ]
    print(
        json.dumps(
            {
                "examples": len(rows),
                "train": len(train_items),
                "validation": len(validation_items),
                "min_tokens": min(lengths),
                "max_tokens": max(lengths),
                "min_target_tokens": min(target_lengths),
                "max_target_tokens": max(target_lengths),
                "gpu": torch.cuda.get_device_name(0),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=quantization,
        device_map={"": 0},
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=(
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ),
        ),
    )
    model.print_trainable_parameters()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train_config.json").write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        fp16=True,
        bf16=False,
        optim="paged_adamw_8bit",
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=EncodedDataset(train_items),
        eval_dataset=EncodedDataset(validation_items),
        data_collator=CompletionCollator(tokenizer.pad_token_id),
    )
    checkpoint = None if args.no_resume else get_last_checkpoint(str(output_dir))
    if checkpoint:
        print(f"Resuming from {checkpoint}", flush=True)
    trainer.train(resume_from_checkpoint=checkpoint)

    final_dir = output_dir / "final_adapter"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    metrics = trainer.evaluate()
    (output_dir / "final_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved adapter to {final_dir}", flush=True)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
