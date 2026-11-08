"""vLLM engine: the same generate()/plan contract as planner.py, plus generate_batch().

Needs Linux (Colab / Kaggle / the Docker image) with a CUDA GPU; vLLM does not run on
native Windows. The speed-up comes from continuous batching: run_all.py hands it a whole
chunk of prompts at once (`--engine vllm --batch-size 32`).

Greedy decoding (temperature 0, fixed seed). Batched GPU arithmetic can differ in the last
bits depending on which requests share a batch, so two runs may rarely differ on a
borderline token; the competition tolerates |PA(upload) - PA(rerun)| <= 0.03.
"""

from __future__ import annotations

from typing import Any

from code_member_d.planner import parse_plan_text
from code_member_d.prompt_budget import fit_ids


class VllmGenerator:
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
        adapter_path: str | None = None,
        *,
        max_input_tokens: int = 4096,
        max_new_tokens: int = 1024,
        dtype: str = "half",
        quantization: str | None = None,
        gpu_memory_utilization: float = 0.90,
        tensor_parallel_size: int = 1,
        max_num_seqs: int = 32,
        enforce_eager: bool = False,
    ):
        try:
            from vllm import LLM, SamplingParams
            from vllm.lora.request import LoRARequest
        except ImportError as exc:  # pragma: no cover - needs the vLLM stack
            raise RuntimeError(
                "vLLM is not installed (Linux + CUDA only): pip install vllm"
            ) from exc

        self.model_id = model_id if not adapter_path else f"{model_id}+{adapter_path}"
        self.max_input_tokens = max_input_tokens
        kwargs: dict[str, Any] = dict(
            model=model_id,
            dtype=dtype,
            max_model_len=max_input_tokens + max_new_tokens + 64,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            max_num_seqs=max_num_seqs,
            enforce_eager=enforce_eager,
            seed=0,
        )
        if quantization:
            kwargs["quantization"] = quantization
            if quantization == "bitsandbytes":
                kwargs["load_format"] = "bitsandbytes"
        if adapter_path:
            kwargs.update(enable_lora=True, max_lora_rank=16)  # B's adapter is rank 16
        self.llm = LLM(**kwargs)
        self.tokenizer = self.llm.get_tokenizer()
        self.sampling = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
        self.lora = LoRARequest("datamate", 1, str(adapter_path)) if adapter_path else None

    def _ids(self, prompt: str) -> list[int]:
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        return fit_ids(
            lambda text: self.tokenizer(text, add_special_tokens=False)["input_ids"],
            rendered, self.max_input_tokens,
        )

    def generate_batch(self, prompts: list[str]) -> list[dict[str, Any]]:
        if not prompts:
            return []
        outputs = self.llm.generate(
            [{"prompt_token_ids": self._ids(p)} for p in prompts],
            self.sampling,
            lora_request=self.lora,
            use_tqdm=False,
        )
        return [parse_plan_text(o.outputs[0].text.strip()) for o in outputs]

    def generate(self, prompt: str) -> dict[str, Any]:
        return self.generate_batch([prompt])[0]
