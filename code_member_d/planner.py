"""Qwen (4-bit) SQL generator with an optional LoRA adapter, for inference.

Needs a CUDA GPU plus: transformers>=4.56,<5, bitsandbytes, accelerate, peft.
Greedy decoding only, so two runs on the same input give the same output.
"""

from __future__ import annotations

from typing import Any

from code_member_b.qwen_teacher import extract_first_json


class QwenAdapterGenerator:
    """generate(prompt) -> dict with answer_type / sql / confidence / reason."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
        adapter_path: str | None = None,
        max_input_tokens: int = 6144,
        max_new_tokens: int = 1024,
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
        self.max_new_tokens = max_new_tokens
        self._torch = torch
        torch.manual_seed(0)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        # Keep the end of an over-long prompt (task + question) rather than the start.
        self.tokenizer.truncation_side = "left"
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

    def generate(self, prompt: str) -> dict[str, Any]:
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(
            rendered,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        ).to(self.model.device)
        with self._torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                repetition_penalty=1.0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )[0]
        text = self.tokenizer.decode(
            output[inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
        self._torch.cuda.empty_cache()
        try:
            return extract_first_json(text)
        except ValueError:
            return {"answer_type": "abstain", "confidence": 0.0,
                    "reason": f"invalid_json: {text[:300]}"}
