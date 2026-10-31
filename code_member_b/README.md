# Member B — Agent Core

This package implements the question-to-SQL core owned by member B. It does
not modify member A's cache. `context_adapter.py` supports the cache layout
currently present in this repository and the layout proposed in the team plan.

## Local setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r code_member_a\requirements.txt pytest
```

## Run tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Current contract

```python
from code_member_b import AgentCore

agent = AgentCore(planner=my_planner)
result = agent.answer(
    question="Tháng này bán được bao nhiêu?",
    business_id="biz_101",
    as_of_date="2026-08-24",
    split="train",
)
```

A planner must implement `plan(prompt) -> dict` and return:

```json
{
  "answer_type": "number",
  "sql": "SELECT ...",
  "confidence": 0.9,
  "reason": null
}
```

The online answer path remains planner-pluggable. Tests use `StaticPlanner` so
the data, safety, execution, retry, and formatting paths can be verified end to
end, while the pseudo-label path below provides the concrete offline Qwen
teacher used before QLoRA.

## Execution-guided SQL pseudo-labels

On a Colab T4, install the optional teacher dependencies and run a deterministic
100-question pilot:

```bash
pip install -r code_member_a/requirements.txt -r code_member_b/requirements-colab.txt
python -m code_member_b.pseudo_label_cli \
  --data-root data/datamate-corpus \
  --output-dir outputs/pseudo_labels/pilot_v1 \
  --pilot-size 100 \
  --candidate-count 3
```

The CLI loads `Qwen/Qwen2.5-Coder-7B-Instruct` in NF4 and uses a 4,096-token
SDPA context so it fits a T4. It tries up to three SQL plans sequentially. After
each failure, the next attempt receives the DuckDB validation error or a private
verification-failure signal, but never the gold answer. It accepts only
read-only results matching `answer` or `alias`. Each result is written
immediately to `accepted.jsonl` or `rejected.jsonl`, so rerunning the same
command resumes instead of repeating completed IDs. Verified examples are also
exported to `qlora_train.jsonl` without gold answers or execution traces.

Use `--pilot-size 0` only after reviewing `report.json` and representative
failures from the pilot. Generated data remains under the ignored `outputs/`
directory.

For a fast CPU-only first pass, add `--template-only`. This mode never loads
Qwen: it executes only deterministic schema-derived candidates and checkpoints
the verified subset. A later teacher pass can be seeded with that
`accepted.jsonl` so the model skips examples already solved deterministically.

For long Colab runs, point `--output-dir` at a mounted Google Drive directory;
checkpoint JSONL files are append-only and tolerate an incomplete final line.

## QLoRA training

Train only on the execution-verified `qlora_train.jsonl`. The trainer uses a
4-bit NF4 base model, completion-only loss, gradient checkpointing, and a
batch size of one so the default 4,096-token run fits a Colab T4. Checkpoints
are saved after every epoch and an interrupted command resumes automatically.

```bash
python -m code_member_b.qlora_train \
  --data-path /content/drive/MyDrive/datamate-agent/pseudo_labels/semantic3000_v2/qlora_train.jsonl \
  --output-dir /content/drive/MyDrive/datamate-agent/qlora/qwen25coder7b_semantic_v1 \
  --max-length 4096 \
  --epochs 3 \
  --gradient-accumulation-steps 8
```

The final PEFT adapter and tokenizer are written to `final_adapter/`; validation
metrics are written to `final_metrics.json`. Use a new output directory when
changing the dataset or training configuration, or pass `--no-resume` only
when intentionally restarting the same run.
