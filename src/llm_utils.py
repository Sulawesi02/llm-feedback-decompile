from pathlib import Path
from typing import Any, Tuple
from transformers import AutoTokenizer
from vllm import LLM

def init_engine(model_path: Path, max_prompt_tokens: int, batch_size: int) -> Tuple[LLM, Any]:
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"

    llm = LLM(
        model=str(model_path),
        tokenizer=str(model_path),
        dtype="bfloat16",
        trust_remote_code=True,
        enable_lora=True,
        max_loras=1,
        max_lora_rank=16,
        gpu_memory_utilization=0.7,
        max_model_len=max_prompt_tokens,
        max_num_seqs=batch_size,
        swap_space=4,
        enforce_eager=False,
        tensor_parallel_size=1,
    )

    return llm, tokenizer

def clean_output(text: str) -> str:
    return (
        text.strip()
        .replace("```cpp", "")
        .replace("```c", "")
        .replace("```", "")
        .strip()
    )
