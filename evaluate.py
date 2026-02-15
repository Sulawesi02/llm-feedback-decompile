import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset
from vllm import LLM, SamplingParams, RequestOutput

from src.config import (
    PROCESSED_DATA_DIR,
    EVAL_DIR,
    MODEL_DIR,
    MODEL_NAME,
    DPO_DIR,
    MAX_PROMPT_TOKENS,
    MAX_GEN_TOKENS,
)
from src.compiler import compile_test
from src.prompts import construct_infer_prompt, construct_fix_prompt

MAX_ITERS = 3
BATCH_SIZE = 8

def evaluate_model(
    version: str,
    test_input: Path,
    model_path: Path,
    dpo_dir: Optional[Path],
    eval_output: Path
):

    success_count = 0
    total_count = 0
    processed_keys = set()
    existing_results = []
    skipped_count = 0

    # 读取已有记录（支持断点续传）
    if eval_output.exists():
        try:
            with open(eval_output, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    for line in content.splitlines():
                        if line.strip():
                            existing_results.append(json.loads(line))

            print(f"发现 {len(existing_results)} 条已有记录")

            for item in existing_results:
                key = item.get("func")
                if key:
                    processed_keys.add(key)
                if item.get("success"):
                    success_count += 1
                total_count += 1

        except Exception as e:
            print(f"读取现有结果失败: {e}, 将重新开始")
            existing_results = []

    test_data = load_dataset("json", data_files=str(test_input), split="train")
    total_samples = len(test_data)
    
    if total_count >= total_samples:
        print(f"已达到目标记录数 ({total_samples} 条)，跳过评估")
        accuracy = success_count / total_count if total_count > 0 else 0
        print(f"总记录数: {total_count}")
        print(f"成功编译数: {success_count}")
        print(f"编译通过率: {accuracy:.2%}")
        return

    print(f"当前进度: {total_count}/{total_samples}, 继续评估...")

    print("加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True
    )
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

    if existing_results:
        f_out = open(eval_output, "a", encoding="utf-8")
    else:
        f_out = open(eval_output, "w", encoding="utf-8")

    # 收集所有待处理样本
    pending_samples: List[Dict[str, Any]] = []

    for item in test_data:
        func_dep = item.get("func_dep")
        func = item.get("func")
        test = item.get("test")
        asm = item.get("asm")

        if not all([func_dep, func, asm, test]):
            continue

        if func in processed_keys:
            continue
        
        # 检查样本长度是否超过最大允许长度
        messages = construct_infer_prompt(asm)
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        prompt_token_ids = tokenizer.encode(prompt_text)
        
        if len(prompt_token_ids) > MAX_PROMPT_TOKENS:
            skipped_count += 1
            continue

        pending_samples.append({
            "func_dep": func_dep,
            "func": func,
            "test": test,
            "asm": asm,
            "iter": 0,
            "prev_outputs": None,
            "last_error": None,
            "history": [],
            "best_outputs": None,
            "finished": False,
            "infer_prompt": prompt_text,  # 缓存第一次的 prompt_text
        })

    print(f"总待处理样本数: {len(pending_samples)}")

    pbar = tqdm(total=len(pending_samples), desc=f"评估 {version} (批处理)")

    while pending_samples:
        # 本批次已完成的样本结果（用于一次性写入）
        batch_results: List[Dict] = []

        # 当前批次的待生成样本
        current_batch: List[Dict[str, Any]] = []
        prompts: List[str] = []
        batch_sample_map: List[int] = []  # prompt 索引 → pending_samples 索引

        # 取最多 BATCH_SIZE 个未完成的样本
        for idx, sample in enumerate(pending_samples):
            if sample["finished"]:
                continue
            if len(current_batch) >= BATCH_SIZE:
                break

            # 构造当前迭代的 prompt
            if sample["iter"] == 0:
                text = sample["infer_prompt"]
            else:
                messages = construct_fix_prompt(
                    sample["asm"],
                    sample["prev_outputs"],
                    sample["last_error"]
                )

                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

            current_batch.append(sample)
            prompts.append(text)
            batch_sample_map.append(idx)

        if not prompts:
            break  # 没有更多待处理

        print(f"批量生成 {len(prompts)} 个 prompt...")

        lora_request = None
        if dpo_dir:
            from vllm.lora.request import LoRARequest
            lora_request = LoRARequest(
                lora_name=dpo_dir.name,
                lora_int_id=1,
                lora_path=str(dpo_dir),
            )

        # 批量生成
        outputs: List[RequestOutput] = llm.generate(
            prompts,
            sampling_params,
            lora_request=lora_request,
            use_tqdm=False
        )

        # 处理这一批的结果
        for batch_idx, output in enumerate(outputs):
            sample_idx = batch_sample_map[batch_idx]
            sample = pending_samples[sample_idx]

            generated_text = output.outputs[0].text.strip()
            gen_output_text = generated_text.replace("```cpp", "").replace("```c", "").replace("```", "").strip()

            # print(f"生成的 C 函数代码 (样本 {sample['func'][:30]}...):\n{gen_output_text}")

            success, error_msg = compile_test(sample["func_dep"], gen_output_text, sample["test"])

            sample["history"].append({
                "iter": sample["iter"],
                "success": success,
                "outputs": gen_output_text,
                "error": error_msg if not success else "",
            })

            # 无论成功失败，都记录本次生成
            if success:
                sample["best_outputs"] = gen_output_text
                sample["finished"] = True
            else:
                sample["prev_outputs"] = gen_output_text
                sample["last_error"] = error_msg
                sample["iter"] += 1
                if sample["iter"] >= MAX_ITERS:
                    sample["finished"] = True

            # 如果样本已完成（成功或达到最大迭代），加入本批次结果
            if sample["finished"]:
                result_entry = {
                    "func": sample["func"],
                    "asm": sample["asm"],
                    "success": sample["best_outputs"] is not None,
                    "best_outputs": sample["best_outputs"],
                    "history": sample["history"],
                }
                batch_results.append(result_entry)
                if result_entry["success"]:
                    success_count += 1
                total_count += 1
                pbar.update(1)

        # 本批次所有已完成样本，一次性写入
        for result in batch_results:
            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")

        f_out.flush()

        # 清理已完成样本
        pending_samples = [s for s in pending_samples if not s["finished"]]

    pbar.close()
    f_out.close()

    accuracy = success_count / total_count if total_count > 0 else 0
    print(f"\n{'='*40}")
    print(f"评估完成: {version}")
    print(f"总样本数: {total_samples}")
    print(f"成功编译数: {success_count}")
    print(f"编译通过率: {accuracy:.2%}")
    print(f"跳过的样本数: {skipped_count}")
    print(f"实际处理的样本数: {total_count}")
    print(f"{'='*40}")

def main():
    test_input = PROCESSED_DATA_DIR / "test_data.jsonl"
    model_path = MODEL_DIR / MODEL_NAME
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 评估基座模型
    eval_output = EVAL_DIR / "base_model.jsonl"
    print(f"{'='*20}\n开始评估基座模型\n{'='*20}")
    try:
        evaluate_model("base_model", test_input, model_path, None, eval_output)
    except Exception as e:
        print(f"基座模型评估失败: {e}")

    # 2. 评估 DPO 模型
    eval_output = EVAL_DIR / "dpo_model.jsonl"
    if not DPO_DIR.exists():
        print(f"DPO 适配器不存在 {DPO_DIR}，跳过评估")
    else:
        print(f"{'='*20}\n开始评估 DPO 模型\n{'='*20}")
        try:
            evaluate_model("dpo_model", test_input, model_path, DPO_DIR, eval_output)
        except Exception as e:
            print(f"DPO 模型评估失败: {e}")

if __name__ == "__main__":
    main()