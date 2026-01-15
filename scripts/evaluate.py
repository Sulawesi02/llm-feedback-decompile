import os
import json
import torch
import re
import subprocess
import shutil
import tempfile
import argparse
import time
import glob
import sys
from pathlib import Path
from typing import List, Dict, Optional, Union
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList

# 添加项目根目录到 sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import TEST_DATA_FILE_PATH, MODEL_DIR, EVAL_DIR, TEMP_DIR
from src.utils import compile, construct_initial_prompt, construct_refine_prompt, clean_c_code, load_model_utils, create_timeout_stopping_criteria

class ModelEvaluator:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        try:
            model , self.tokenizer = load_model_utils(model_path)
            self.model = model.eval()
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise

    def generate(self, messages: Union[str, List[Dict[str, str]]]) -> str:
        if not self.model:
            return ""
        
        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            
            timeout_criteria = create_timeout_stopping_criteria(120.0)
            stopping_criteria = StoppingCriteriaList([timeout_criteria])
            
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=2048, 
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
                stopping_criteria=stopping_criteria
            )
            
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            if timeout_criteria.timed_out:
                print("生成超时")
            
            return clean_c_code(response)
        except Exception as e:
            print(f"生成出错: {e}")
            return ""
    
    def unload(self):
        """释放模型资源"""
        if self.model:
            del self.model
        if self.tokenizer:
            del self.tokenizer
        torch.cuda.empty_cache()
        print("模型资源已释放")

def evaluate_model(model_path: str, output_file: str, max_samples: int = None, iters: int = 3):
    model_evaluator = ModelEvaluator(model_path)
    
    results = []
    success_count = 0
    total_count = 0
    
    print(f"开始评估模型: {os.path.basename(model_path)}")
    
    try:
        with open(TEST_DATA_FILE_PATH, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        if max_samples:
            lines = lines[:max_samples]
            
        for i, line in tqdm(enumerate(lines), total=len(lines), desc="Evaluating"):
            try:
                data = json.loads(line)
                
                tasks = []
                for arch, arch_data in data["asm"].items():
                    for opt, opt_data in arch_data.items():
                        tasks.append({
                            "arch": arch,
                            "opt": opt,
                            "machine_code": opt_data.get("machine_code"),
                            "id_suffix": f"{arch}_{opt}"
                        })

                for task in tasks:
                    arch = task["arch"]
                    opt = task["opt"]
                    machine_code = task["machine_code"]
                    
                    if not machine_code:
                        continue
                
                    # 初始反编译
                    messages = construct_initial_prompt(arch, opt, machine_code)
                    c_code = model_evaluator.generate(messages)
                    
                    # 编译验证
                    compile_result = compile(c_code, arch, opt)
                    
                    # 循环反馈 (如果开启)
                    history = [{"iter": 0, "c_code": c_code, "compile_result": compile_result}]
                    
                    final_success = compile_result["success"]
                    final_c_code = c_code
                    
                    if not final_success and iters > 0:
                        current_c_code = c_code
                        current_error = compile_result["error"]
                        
                        for it in range(iters):
                            refine_messages = construct_refine_prompt(current_c_code, current_error[:1000])
                            
                            current_c_code = model_evaluator.generate(refine_messages)
                            if not current_c_code:
                                break
                                
                            current_compile_result = compile(current_c_code, arch, opt)
                            history.append({"iter": it + 1, "c_code": current_c_code, "compile_result": current_compile_result})
                            
                            if current_compile_result["success"]:
                                final_success = True
                                final_c_code = current_c_code
                                # 清理工作目录
                                if current_compile_result.get("workdir"):
                                    shutil.rmtree(current_compile_result["workdir"], ignore_errors=True)
                                break
                            else:
                                current_error = current_compile_result["error"]
                                # 清理工作目录
                                if current_compile_result.get("workdir"):
                                    shutil.rmtree(current_compile_result["workdir"], ignore_errors=True)

                    # 清理初始编译的工作目录
                    if compile_result.get("workdir"):
                        shutil.rmtree(compile_result["workdir"], ignore_errors=True)
                    
                    result_entry = {
                        "id": f"{i}_{task['id_suffix']}" if task['id_suffix'] else i,
                        "arch": arch,
                        "opt": opt,
                        "success": final_success,
                        "final_c_code": final_c_code,
                        "history": history
                    }
                    results.append(result_entry)
                    
                    if final_success:
                        success_count += 1
                    total_count += 1
                
            except json.JSONDecodeError:
                continue
            except Exception as e:
                print(f"处理样本 {i} 时出错: {e}")
                continue
                
    finally:
        model_evaluator.unload()
        
    # 保存结果
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    accuracy = success_count / total_count if total_count > 0 else 0
    print(f"\n评估完成: {os.path.basename(model_path)}")
    print(f"总样本数: {total_count}")
    print(f"成功编译数: {success_count}")
    print(f"编译通过率: {accuracy:.2%}")
    
    return accuracy

def main():
    max_samples = None
    
    # 扫描 merged_model 下的所有版本目录
    model_paths = []
    merged_model_dir = MODEL_DIR / "merged_model"
    if merged_model_dir.exists():
        for entry in os.scandir(merged_model_dir):
            if entry.is_dir():
                model_paths.append(entry.path)
    
    if not model_paths:
        print("未找到可评估的模型")
        return
        
    print(f"将评估以下模型: {[os.path.basename(p) for p in model_paths]}")
    
    summary = {}
    
    for model_path in model_paths:
        version_name = os.path.basename(model_path)
        
        try:
            # 循环反馈反编译评估
            print(f"正在评估 {version_name} ...")
            output_file = EVAL_DIR / f"{version_name}.json"
            
            acc = evaluate_model(
                model_path,
                str(output_file),
                max_samples
            )
            summary[version_name] = acc

        except Exception as e:
            print(f"评估模型 {version_name} 失败: {e}")
            summary[version_name] = "Failed"
            
    print("\n" + "="*30)
    print("所有模型评估汇总")
    print("="*30)
    for ver, acc in summary.items():
        if isinstance(acc, float):
            print(f"{ver}: {acc:.2%}")
        else:
            print(f"{ver}: {acc}")

if __name__ == "__main__":
    main()
