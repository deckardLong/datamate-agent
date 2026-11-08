"""Qwen (4-bit) SQL generator with an optional LoRA adapter, for inference.

Needs a CUDA GPU plus: transformers>=4.56,<5, bitsandbytes, accelerate, peft.
Greedy decoding only, so two runs on the same input give the same output.
"""

from __future__ import annotations

import gc
from typing import Any

from code_member_b.qwen_teacher import extract_first_json
from code_member_d.prompt_budget import fit_ids


def parse_plan_text(text: str) -> dict[str, Any]:
    """First JSON object in the model output; a debug-only abstain plan if there is none."""
    try:
        return extract_first_json(text)
    except ValueError:
        # Debug info stays in the log only; it never reaches submission.jsonl.
        return {"answer_type": "abstain", "confidence": 0.0,
                "reason": "model output was not valid JSON", "raw_output": text[:500]}


class QwenAdapterGenerator:
    """generate(prompt) -> dict with answer_type / sql / confidence / reason."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
        adapter_path: str | None = None,
        max_input_tokens: int = 4096,
        max_new_tokens: int = 1024,
        fallback_input_tokens: int = 2560,
    ):
        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
            )
        except ImportError as exc:  # pragma: no cover - needs GPU stack
            raise RuntimeError(
                "pip install -r code_member_b/requirements-colab.txt "
                "&& pip install -U 'transformers>=4.56,<5'"
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError("QwenAdapterGenerator requires a CUDA GPU")

        self.model_id = model_id if not adapter_path else f"{model_id}+{adapter_path}"
        self.max_input_tokens = max_input_tokens
        self.fallback_input_tokens = fallback_input_tokens
        self.max_new_tokens = max_new_tokens
        self._torch = torch
        torch.manual_seed(0)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            ),
            device_map={"": 0},
            dtype=torch.float16,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )
        if adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
        model.eval()
        self.model = model

    def _fit(self, rendered: str, limit: int) -> list[int]:
        """Token ids within `limit`.

        First shrink the low-value sections (join hints, table-name list, sample rows,
        rules, notes) so the exact schema and the question stay intact. Only if that is
        not enough, cut the middle keeping BOTH ends (60% head / 40% tail, as B did for
        training prompts): the head holds the output-format instruction, the tail the
        task and question.
        """
        return fit_ids(
            lambda text: self.tokenizer(text, add_special_tokens=False)["input_ids"],
            rendered, limit,
        )

    def _try_generate(self, rendered: str, limit: int) -> str | None:
        """Generate with the prompt cut to `limit` tokens; None if the GPU ran out of memory.

        The OOM is handled inside this method so its traceback (which pins GPU tensors)
        is released before the caller retries with a shorter prompt.
        """
        try:
            ids = self._fit(rendered, limit)
            input_ids = self._torch.tensor([ids], device=self.model.device)
            with self._torch.inference_mode():
                output = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=self._torch.ones_like(input_ids),
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    repetition_penalty=1.0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                )[0]
            return self.tokenizer.decode(
                output[len(ids):], skip_special_tokens=True
            ).strip()
        except RuntimeError as exc:  # torch.OutOfMemoryError is a RuntimeError
            if "out of memory" not in str(exc).lower():
                raise
            return None
        finally:
            gc.collect()
            self._torch.cuda.empty_cache()

    def generate(self, prompt: str) -> dict[str, Any]:
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        # Prefill memory grows with prompt length (~quadratically for attention on a
        # T4), so on OOM retry once with a much shorter prompt (middle dropped).
        text = self._try_generate(rendered, self.max_input_tokens)
        if text is None:
            text = self._try_generate(rendered, self.fallback_input_tokens)
        if text is None:
            raise RuntimeError(
                f"CUDA out of memory even with a {self.fallback_input_tokens}-token prompt"
            )
        return parse_plan_text(text)
