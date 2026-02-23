import json
import random
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm
from datasets import load_dataset
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer

from src.config import (
    PROCESSED_DATA_DIR,
    DPO_DATA_DIR,
    MODEL_DIR,
    MODEL_NAME,
    SFT_DIR,
    BATCH_SIZE,
    MAX_PROMPT_TOKENS,
    MAX_GEN_TOKENS,
)
from src.llm_utils import init_engine, clean_output
from src.compiler import compile_to_obj
from src.prompts import build_generate_text

NUM_CANDIDATES = 3

def generate_dpo_entries(
    input_path: Path,
    llm: LLM,
    tokenizer: AutoTokenizer,
    sampling_params: SamplingParams,
    lora_request: Optional[LoRARequest],
    output_path: Path,
    desc: str,
):
    if output_path.exists():
        print(f"DPO 数据文件已存在，跳过生成: {output_path}")
        return
    
    dataset = load_dataset("json", data_files=str(input_path), split="train")
    total_samples = len(dataset)
    print(f"总样本数: {total_samples}...")

    try:
        data_list = list(dataset)
        max_entries = max(1, int(total_samples * 0.2))
        entries = []
        pbar = tqdm(total=max_entries, desc=desc)
        stop = False
        for start in range(0, len(data_list), BATCH_SIZE):
            prompts: List[str] = []
            func_list: List[str] = []
            asm_list: List[str] = []
            for item in data_list[start : start + BATCH_SIZE]:
                func = item.get("func")
                asm = item.get("asm")
                if not all([func, asm]):
                    continue
                text = build_generate_text(iter=0, tokenizer=tokenizer, sample=item, max_prompt_tokens=MAX_PROMPT_TOKENS)
                if not text:
                    continue
                prompts.append(text)
                func_list.append(func)
                asm_list.append(asm)
            if not prompts:
                continue
            outputs = llm.generate(
                prompts,
                sampling_params,
                lora_request=lora_request,
            )
            for func, asm, request_output in zip(func_list, asm_list, outputs):
                candidates: List[str] = []
                for out in request_output.outputs:
                    text = clean_output(out.text)
                    if text:
                        candidates.append(text)
                if not candidates:
                    continue
                bad_candidates: List[str] = []
                for cand in candidates:
                    success, _ = compile_to_obj(cand)
                    if not success:
                        bad_candidates.append(cand)
                if not bad_candidates:
                    continue
                rejected = random.choice(bad_candidates)
                entry = {
                    "prompt": f"根据给定的汇编代码({asm})，输出一个在语义上完全等价的 C 函数实现",
                    "chosen": func,
                    "rejected": rejected,
                }
                entries.append(entry)
                pbar.update(1)
                if len(entries) >= max_entries:
                    stop = True
                    break
            if stop:
                break
        pbar.close()
        if entries:
            with open(output_path, "w", encoding="utf-8") as f:
                for entry in entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"生成 DPO 数据时出错: {e}")
        import traceback
        traceback.print_exc()

def main():
    train_input = PROCESSED_DATA_DIR / "train_data.jsonl"
    valid_input = PROCESSED_DATA_DIR / "valid_data.jsonl"

    if not train_input.exists():
        print(f"错误: 训练集不存在: {train_input}")
        return
    if not valid_input.exists():
        print(f"错误: 验证集不存在: {valid_input}")
        return
    
    model_path = MODEL_DIR / MODEL_NAME
    llm, tokenizer = init_engine(model_path, MAX_PROMPT_TOKENS, BATCH_SIZE)

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_tokens=MAX_GEN_TOKENS,
        skip_special_tokens=True,
        n=NUM_CANDIDATES,
    )

    lora_request: Optional[LoRARequest] = None
    if SFT_DIR.exists():
        lora_request = LoRARequest(
            lora_name=SFT_DIR.name,
            lora_int_id=1,
            lora_path=str(SFT_DIR),
        )
    
    DPO_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 生成 DPO 训练数据
    print(f"{'='*20}\n开始生成 DPO 训练数据\n{'='*20}")
    train_output = DPO_DATA_DIR / "train_data.jsonl"
    try:
        generate_dpo_entries(
            train_input,
            llm,
            tokenizer,
            sampling_params,
            lora_request,
            train_output,
            "生成 DPO 训练数据进度",
        )
    except Exception as e:
        print(f"生成 DPO 训练数据时出错: {e}")
        return

    # 2. 生成 DPO 验证数据
    print(f"{'='*20}\n开始生成 DPO 验证数据\n{'='*20}")
    valid_output = DPO_DATA_DIR / "valid_data.jsonl"
    try:
        generate_dpo_entries(
            valid_input,
            llm,
            tokenizer,
            sampling_params,
            lora_request,
            valid_output,
            "生成 DPO 验证数据进度",
        )
    except Exception as e:
        print(f"生成 DPO 验证数据时出错: {e}")
        return

if __name__ == "__main__":
    main()
