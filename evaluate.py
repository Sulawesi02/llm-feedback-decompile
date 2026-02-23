import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from tqdm import tqdm
from datasets import load_dataset
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer
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
    BATCH_SIZE,    
    MAX_PROMPT_TOKENS,
    MAX_GEN_TOKENS,
)
from src.llm_utils import init_engine, clean_output
from src.compiler import compile_test
from src.prompts import build_generate_text

MAX_ITERS = 3

def evaluate_model(
    eval_input: Path,
    model_path: Path,
    llm: LLM,
    tokenizer: AutoTokenizer,
    sampling_params: SamplingParams,
    lora_request: Optional[LoRARequest],
    eval_results: Path,
    eval_summary: Path,
    desc: str,
):
    if eval_results.exists() and eval_summary.exists():
        print(f"结果文件已存在，跳过评估: {eval_results}")
        return
    
    processed_count = 0
    success_count = 0
    invalid_sample_count = 0
    prompt_too_long_count = 0
    llm_error_count = 0
    compile_error_count = 0

    dataset = load_dataset("json", data_files=str(eval_input), split="train")
    total_samples = len(dataset)
    print(f"总样本数: {total_samples}...")

    buffer: List[str] = []
    buffer_limit = 32

    try:
        data_list = list(dataset)
        with open(eval_results, "w", encoding="utf-8") as f_out:
            for start in tqdm(range(0, len(data_list), BATCH_SIZE), desc=desc):
                batch_states: List[Dict[str, Any]] = []
                for item in data_list[start : start + BATCH_SIZE]:
                    index = item.get("index")
                    func_dep = item.get("func_dep")
                    func = item.get("func")
                    test = item.get("test")
                    asm = item.get("asm")
                    if not all([index, func_dep, func, asm, test]):
                        invalid_sample_count += 1
                        continue
                    batch_states.append(
                        {
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
                        }
                    )
                active_states = batch_states[:]
                while active_states:
                    prompts = []
                    current_states = []

                    for state in active_states:
                        if state["finished"]:
                            continue
                        text = build_generate_text(iter=state["iter"], tokenizer=tokenizer, sample=state, max_prompt_tokens=MAX_PROMPT_TOKENS)
                        if not text:
                            state["finished"] = True
                            state["history"].append({
                                "iter": state["iter"],
                                "outputs": "",
                                "error": "prompt too long",
                            })
                            result_entry = {
                                "index": state["index"],
                                "success": False,
                                "func": state["func"],
                                "asm": state["asm"],
                                "best_outputs": state["best_outputs"],
                                "history": state["history"],
                            }
                            buffer.append(json.dumps(result_entry, ensure_ascii=False) + "\n")
                            prompt_too_long_count += 1
                            continue
                        prompts.append(text)
                        current_states.append(state)
                    if not prompts:
                        break
                    try:
                        outputs = llm.generate(
                            prompts,
                            sampling_params,
                            lora_request=lora_request,
                        )
                    except Exception as e:
                        error_msg = f"llm_generate_error: {repr(e)}"
                        for state in current_states:
                            state["finished"] = True
                            state["history"].append(
                                {
                                    "iter": state["iter"],
                                    "outputs": "",
                                    "error": error_msg,
                                }
                            )
                            result_entry = {
                                "index": state["index"],
                                "success": False,
                                "func": state["func"],
                                "asm": state["asm"],
                                "best_outputs": state["best_outputs"],
                                "history": state["history"],
                            }
                            buffer.append(json.dumps(result_entry, ensure_ascii=False) + "\n")
                            llm_error_count += 1
                        active_states = [s for s in active_states if not s["finished"]]
                        continue
                    for state, request_output in zip(current_states, outputs):
                        text = clean_output(request_output.outputs[0].text)
                        try:
                            success, error_msg = compile_test(state["func_dep"], text, state["test"])
                        except Exception as e:
                            success = False
                            error_msg = f"compile_test_error: {repr(e)}"
                            compile_error_count += 1
                        state["history"].append(
                            {
                                "iter": state["iter"],
                                "outputs": text,
                                "error": error_msg if not success else "",
                            }
                        )
                        if success:
                            state["finished"] = True
                            state["best_outputs"] = text
                        else:
                            state["prev_outputs"] = text
                            state["last_error"] = error_msg
                            state["iter"] += 1
                            if state["iter"] >= MAX_ITERS:
                                state["finished"] = True
                        if state["finished"]:
                            result_entry = {
                                "index": state["index"],
                                "success": state["best_outputs"] is not None,
                                "func": state["func"],
                                "asm": state["asm"],
                                "best_outputs": state["best_outputs"],
                                "history": state["history"],
                            }
                            buffer.append(json.dumps(result_entry, ensure_ascii=False) + "\n")
                            if result_entry["success"]:
                                success_count += 1
                            processed_count += 1
                    active_states = [s for s in active_states if not s["finished"]]
                    if len(buffer) >= buffer_limit:
                        f_out.writelines(buffer)
                        f_out.flush()
                        buffer.clear()
            if buffer:
                f_out.writelines(buffer)
                f_out.flush()
                buffer.clear()
    except Exception as e:
        print(f"评估过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        if buffer:
            try:
                with open(eval_results, "a", encoding="utf-8") as f_out:
                    f_out.writelines(buffer)
            except Exception:
                pass
            buffer.clear()
    finally:
        accuracy = success_count / processed_count if processed_count > 0 else 0
        # 保存 summary
        summary = {
            "version": lora_request.lora_path.split("/")[-1] if lora_request else "base_model",
            "timestamp": datetime.now().isoformat(),
            "model_path": str(model_path),
            "lora_path": str(lora_request.lora_path) if lora_request else None,
            "total_samples": total_samples,
            "success_count": success_count,
            "accuracy": float(accuracy),
            "invalid_sample_count": invalid_sample_count,
            "prompt_too_long_count": prompt_too_long_count,
            "llm_error_count": llm_error_count,
            "compile_error_count": compile_error_count,
            "processed_count": processed_count,
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
    if not eval_input.exists():
        print(f"错误: 评估集不存在: {eval_input}")
        return
    
    model_path = MODEL_DIR / MODEL_NAME
    llm, tokenizer = init_engine(model_path, MAX_PROMPT_TOKENS, BATCH_SIZE)
    
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=MAX_GEN_TOKENS,
        skip_special_tokens=True,
    )
    
    # 1. 评估基座模型
    print(f"{'='*20}\n开始评估基座模型\n{'='*20}")
    BASE_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    base_result_output = BASE_EVAL_DIR / "results.jsonl"
    base_summary_output = BASE_EVAL_DIR / "summary.json"
    try:
        evaluate_model(
            eval_input,
            model_path,
            llm,
            tokenizer,
            sampling_params,
            None,
            base_result_output,
            base_summary_output,
            "评估基座模型",
        )
    except Exception as e:
        print(f"基座模型评估失败: {e}")

    # 2. 评估 SFT 模型
    print(f"{'='*20}\n开始评估 SFT 模型\n{'='*20}")
    SFT_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    sft_result_output = SFT_EVAL_DIR / "results.jsonl"
    sft_summary_output = SFT_EVAL_DIR / "summary.json"
    if not SFT_DIR.exists():
        print(f"SFT 适配器不存在: {SFT_DIR}")
    else:
        try:
            lora_request: Optional[LoRARequest] = None
            if SFT_DIR:
                lora_request = LoRARequest(
                    lora_name=SFT_DIR.name,
                    lora_int_id=1,
                    lora_path=str(SFT_DIR),
                )
            evaluate_model(
                eval_input, 
                model_path,
                llm,
                tokenizer,
                sampling_params,
                lora_request,
                sft_result_output,
                sft_summary_output,
                "评估 SFT 模型",
            )
        except Exception as e:
            print(f"SFT 模型评估失败: {e}")   

    # 3. 评估 DPO 模型
    print(f"{'='*20}\n开始评估 DPO 模型\n{'='*20}")
    DPO_EVAL_DIR.mkdir(parents=True, exist_ok=True)    
    dpo_result_output = DPO_EVAL_DIR / "results.jsonl"
    dpo_summary_output = DPO_EVAL_DIR / "summary.json"
    if not DPO_DIR.exists():
        print(f"DPO 适配器不存在: {DPO_DIR}")
    else:
        try:
            lora_request: Optional[LoRARequest] = None
            if DPO_DIR:
                lora_request = LoRARequest(
                    lora_name=DPO_DIR.name,
                    lora_int_id=1,
                    lora_path=str(DPO_DIR),
                )
            evaluate_model(
                eval_input, 
                model_path,
                llm,
                tokenizer,
                sampling_params,
                lora_request,
                dpo_result_output,
                dpo_summary_output,
                "评估 DPO 模型",
            )
        except Exception as e:
            print(f"DPO 模型评估失败: {e}")

if __name__ == "__main__":
    main()
