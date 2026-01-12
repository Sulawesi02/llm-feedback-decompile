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
from typing import List, Dict, Optional
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList

# 添加项目根目录到 sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import TEST_DATA_PATH, MODEL_DIR, EVAL_DIR, TEMP_DIR
from src.utils import construct_initial_prompt, construct_refine_prompt, clean_c_code, compile_c_code

# 确保目录存在
EVAL_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

class TimeoutStoppingCriteria(StoppingCriteria):
    def __init__(self, timeout_seconds: float):
        self.start_time = time.time()
        self.timeout_seconds = timeout_seconds
        self.timed_out = False

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if time.time() - self.start_time > self.timeout_seconds:
            self.timed_out = True
            return True
        return False

class ModelEvaluator:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        print(f"正在加载模型: {self.model_path}")
        try:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                quantization_config=quant_config,
                device_map={"": 0}, 
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                attn_implementation="flash_attention_2", 
            )
            
            # 将模型切换到推理模式
            self.model.eval()
            print("模型加载完成")
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise

    def generate(self, messages: List[Dict[str, str]]) -> str:
        if not self.model:
            return ""
        
        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            
            timeout_criteria = TimeoutStoppingCriteria(120.0)
            stopping_criteria = StoppingCriteriaList([timeout_criteria])
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=2048, 
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
                stopping_criteria=stopping_criteria
            )
            
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)
            ]
            response_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            if timeout_criteria.timed_out:
                print("生成超时")
            
            return clean_c_code(response_text)
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
        with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        if max_samples:
            lines = lines[:max_samples]
            
        for i, line in tqdm(enumerate(lines), total=len(lines), desc="Evaluating"):
            try:
                data = json.loads(line)
                arch = data.get('arch')
                opt = data.get('opt')
                machine_code = data.get('machine_code')
                
                # 初始反编译
                messages = construct_initial_prompt(arch, opt, machine_code)
                c_code = model_evaluator.generate(messages)
                
                # 编译验证
                compile_result = compile_c_code(c_code, arch, opt)
                
                # 循环反馈 (如果开启)
                history = [{"iter": 0, "c_code": c_code, "compile_result": compile_result}]
                
                final_success = compile_result["success"]
                final_c_code = c_code
                
                if not final_success and iters > 0:
                    current_c_code = c_code
                    current_error = compile_result["error"]
                    
                    for it in range(iters):
                        refine_msg = construct_refine_prompt(current_c_code, current_error[:1000])
                        
                        current_c_code = model_evaluator.generate(refine_msg)
                        if not current_c_code:
                            break
                            
                        current_compile_result = compile_c_code(current_c_code, arch, opt)
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
                    "id": i,
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
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
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
