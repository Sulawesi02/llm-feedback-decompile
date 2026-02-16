import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset
from vllm import LLM, SamplingParams, RequestOutput
from vllm.lora.request import LoRARequest
from datetime import datetime

from src.config import (
    PROCESSED_DATA_DIR,
    MODEL_DIR,
    MODEL_NAME,
    SFT_DIR,
    DPO_DIR,
    BASE_EVAL_DIR,
    SFT_EVAL_DIR,
    DPO_EVAL_DIR,
    MAX_PROMPT_TOKENS,
    MAX_GEN_TOKENS,
)
from src.compiler import compile_test
from src.prompts import construct_infer_prompt, construct_fix_prompt

MAX_ITERS = 3
BATCH_SIZE = 8

def evaluate_model(
    version: str,
    eval_input: Path,
    model_path: Path,
    lora_path: Optional[Path],
    eval_output: Path
):
    eval_results = eval_output / "results.jsonl"
    eval_summary = eval_output / "summary.json"

    total_count = 0
    success_count = 0
    skipped_count = 0

    test_data = load_dataset("json", data_files=str(eval_input), split="train")
    total_samples = len(test_data)

    print(f"总样本数: {total_samples}，开始从头评估...")

    print("加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"

    print("初始化 vLLM 引擎...")
    llm = LLM(
        model=str(model_path),
        tokenizer=str(model_path),
        dtype="bfloat16",
        trust_remote_code=True,
        enable_lora=True,
        max_loras=1,
        max_lora_rank=16,
        gpu_memory_utilization=0.9,
        max_model_len=MAX_PROMPT_TOKENS,
        max_num_seqs=BATCH_SIZE,
        swap_space=4,
        enforce_eager=False,
        tensor_parallel_size=1,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=MAX_GEN_TOKENS,
        skip_special_tokens=True,
    )

    pending_samples: List[Dict[str, Any]] = []

    for item in test_data:
        index = item.get("index")
        func_dep = item.get("func_dep")
        func = item.get("func")
        test = item.get("test")
        asm = item.get("asm")

        if not all([index, func_dep, func, asm, test]):
            continue

        pending_samples.append({
            "index": index,
            "func_dep": func_dep,
            "func": func,
            "test": test,
            "asm": asm,
            "iter": 0,
            "finished": False,            
            "prev_outputs": None,
            "last_error": None,
            "best_outputs": None,
            "history": [],
        })

    print(f"总待处理样本数: {len(pending_samples)}")

    pbar = tqdm(total=len(pending_samples), desc=f"评估 {version} (批处理)")

    try:
        with open(eval_results, "w", encoding="utf-8") as f_out:
            while pending_samples:
                current_samples = [] # 当前批次样本
                current_prompts = [] # 当前批次提示
                sample_indices = [] # 当前批次样本索引：prompt 索引 → pending_samples 索引
                batch_outputs = [] # 当前批次输出
                
                for idx, sample in enumerate(pending_samples):
                    if sample["finished"]:
                        continue
                    if len(current_samples) >= BATCH_SIZE:
                        break

                    if sample["iter"] == 0:
                        messages = construct_infer_prompt(sample["asm"])
                    else:
                        messages = construct_fix_prompt(sample["asm"], sample["prev_outputs"], sample["last_error"])
                    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    token_count = len(tokenizer.encode(text))

                    if token_count > MAX_PROMPT_TOKENS:
                        sample["finished"] = True
                        sample["history"].append({
                            "iter": sample["iter"],
                            "success": False,
                            "outputs": "",
                            "error": f"prompt too long ({token_count} > {MAX_PROMPT_TOKENS})"
                        })
                        batch_outputs.append({
                            "index": sample["index"],
                            "success": False,
                            "func": sample["func"],
                            "asm": sample["asm"],
                            "best_outputs": sample["best_outputs"],
                            "history": sample["history"],
                        })
                        skipped_count += 1
                        total_count += 1
                        pbar.update(1)
                        continue

                    current_samples.append(sample)
                    current_prompts.append(text)
                    sample_indices.append(idx)

                if not current_prompts:
                    break

                lora_request = None
                if lora_path:
                    lora_request = LoRARequest(
                        lora_name=lora_path.name,
                        lora_int_id=1,
                        lora_path=str(lora_path),
                    )

                # 生成
                outputs: List[RequestOutput] = llm.generate(current_prompts, sampling_params, lora_request=lora_request, use_tqdm=False)

                for batch_idx, output in enumerate(outputs):
                    sample_idx = sample_indices[batch_idx]
                    sample = pending_samples[sample_idx]

                    generated_text = output.outputs[0].text.strip()
                    gen_output_text = generated_text.replace("```cpp", "").replace("```c", "").replace("```", "").strip()

                    success, error_msg = compile_test(sample["func_dep"], gen_output_text, sample["test"])

                    sample["history"].append({
                        "iter": sample["iter"],
                        "success": success,
                        "outputs": gen_output_text,
                        "error": error_msg if not success else "",
                    })

                    if success:
                        sample["best_outputs"] = gen_output_text
                        sample["finished"] = True
                    else:
                        sample["prev_outputs"] = gen_output_text
                        sample["last_error"] = error_msg
                        sample["iter"] += 1
                        if sample["iter"] >= MAX_ITERS:
                            sample["finished"] = True

                    if sample["finished"]:
                        result_entry = {
                            "index": sample["index"],
                            "success": sample["best_outputs"] is not None,
                            "func": sample["func"],
                            "asm": sample["asm"],
                            "best_outputs": sample["best_outputs"],
                            "history": sample["history"],
                        }
                        batch_outputs.append(result_entry)

                        if result_entry["success"]:
                            success_count += 1
                        total_count += 1
                        pbar.update(1)

                for result in batch_outputs:
                    f_out.write(json.dumps(result, ensure_ascii=False) + "\n")

                f_out.flush()

                pending_samples = [s for s in pending_samples if not s["finished"]]

    except Exception as e:
        print(f"评估过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        pbar.close()
        accuracy = success_count / total_count if total_count > 0 else 0
        # 保存 summary
        summary = {
            "version": version,
            "timestamp": datetime.now().isoformat(),
            "model_path": str(model_path),
            "lora_path": str(lora_path) if lora_path else None,
            "total_samples": total_samples,
            "success_count": success_count,
            "accuracy": float(accuracy),
            "skipped_count": skipped_count,
            "processed_count": total_count,
            "max_prompt_tokens": MAX_PROMPT_TOKENS,
            "max_gen_tokens": MAX_GEN_TOKENS,
            "max_iters": MAX_ITERS,
            "batch_size": BATCH_SIZE
        }

        with open(eval_summary, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=4)

        print(f"评估总结已保存至: {eval_summary}")

def main():
    eval_input = PROCESSED_DATA_DIR / "test_data.jsonl"
    model_path = MODEL_DIR / MODEL_NAME
    BASE_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    SFT_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    DPO_EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 评估基座模型
    print(f"{'='*20}\n开始评估基座模型\n{'='*20}")
    try:
        evaluate_model("base_model", eval_input, model_path, None, BASE_EVAL_DIR)
    except Exception as e:
        print(f"基座模型评估失败: {e}")

    # 2. 评估 SFT 模型
    if not SFT_DIR.exists():
        print(f"SFT 适配器不存在 {SFT_DIR}，跳过评估")
    else:
        print(f"{'='*20}\n开始评估 SFT 模型\n{'='*20}")
        try:
            evaluate_model("sft_model", eval_input, model_path, SFT_DIR, SFT_EVAL_DIR)
        except Exception as e:
            print(f"SFT 模型评估失败: {e}")
    
    # 3. 评估 DPO 模型
    if not DPO_DIR.exists():
        print(f"DPO 适配器不存在 {DPO_DIR}，跳过评估")
    else:
        print(f"{'='*20}\n开始评估 DPO 模型\n{'='*20}")
        try:
            evaluate_model("dpo_model", eval_input, model_path, DPO_DIR, DPO_EVAL_DIR)
        except Exception as e:
            print(f"DPO 模型评估失败: {e}")

if __name__ == "__main__":
    main()