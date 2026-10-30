"""Lazy Hugging Face Qwen teacher used by the pseudo-label CLI."""

from __future__ import annotations

import json
from json import JSONDecoder
from typing import Any


def extract_first_json(text: str) -> dict[str, Any]:
    decoder = JSONDecoder()
    for position, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[position:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("Model output does not contain a JSON object")


class QwenTeacher:
    """Four-bit Qwen generator sized for a Colab T4."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
        max_input_tokens: int = 4096,
        max_new_tokens: int = 512,
        temperature: float = 0.35,
        top_p: float = 0.9,
    ):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError(
                "Install the teacher dependencies: pip install -r "
                "code_member_b/requirements-colab.txt"
            ) from exc

        if not torch.cuda.is_available():
            raise RuntimeError("QwenTeacher requires a CUDA GPU")
        self.model_id = model_id
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self._torch = torch
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        # The DataMate prompt puts the question after schema and rules. If a
        # particularly large business context exceeds the T4-safe token budget,
        # preserve that tail instead of truncating the question itself.
        self.tokenizer.truncation_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )
        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def generate(self, prompt: str, candidate_count: int) -> list[dict[str, Any]]:
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
        generation_options = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False,
            "repetition_penalty": 1.05,
            "num_return_sequences": 1,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
        }
        prompt_length = inputs["input_ids"].shape[1]
        candidates = []
        # Generate candidates sequentially. Expanding a long prompt to three
        # return sequences duplicates attention/KV state and can exceed a T4's
        # 16 GB even though the model weights themselves fit in NF4.
        for _ in range(candidate_count):
            with self._torch.inference_mode():
                output = self.model.generate(**inputs, **generation_options)[0]
            text = self.tokenizer.decode(
                output[prompt_length:], skip_special_tokens=True
            ).strip()
            try:
                candidates.append(extract_first_json(text))
            except ValueError:
                candidates.append(
                    {
                        "answer_type": "abstain",
                        "confidence": 0.0,
                        "reason": f"invalid_json: {text[:500]}",
                    }
                )
            self._torch.cuda.empty_cache()
        return candidates
