import uvicorn
import os
import torch
import time
from fastapi import FastAPI, HTTPException
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList
from typing import List, Dict, Any, Union
from pathlib import Path

from config import MERGED_MODEL_DIR, BASE_MODEL_PATH

app = FastAPI()

# 动态获取模型候选列表
def get_model_candidates():
    candidates = []
    # 优先查找 merged_model 下的版本 (v1, v2...)
    if os.path.exists(MERGED_MODEL_DIR):
        try:
            # 按名称降序 (v2 > v1)
            versions = sorted([d for d in os.listdir(MERGED_MODEL_DIR) if (Path(MERGED_MODEL_DIR) / d).is_dir()], reverse=True)
            for v in versions:
                candidates.append(str(Path(MERGED_MODEL_DIR) / v))
        except Exception as e:
            print(f"扫描模型目录出错: {e}")
            
    # 最后加入 Base Model
    candidates.append(str(BASE_MODEL_PATH))
    return candidates

MODEL_CANDIDATES = get_model_candidates()

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

class ModelServer:
    def __init__(self, model_paths: List[str]):
        self.model_paths = model_paths
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        print(f"可用模型候选列表: {self.model_paths}")
        for model_path in self.model_paths:
            if not os.path.exists(model_path):
                print(f"模型路径不存在，跳过: {model_path}")
                continue
                
            print(f"正在加载模型: {model_path}")
            try:
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )

                self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    quantization_config=quant_config,
                    device_map={"": 0}, 
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    attn_implementation="flash_attention_2", 
                )
                
                self.model.eval()
                print(f"模型加载完成: {model_path}")
                return
            except Exception as e:
                print(f"加载模型失败: {e}")
                
        raise RuntimeError("所有模型路径均加载失败，无法启动服务")

    def generate(self, messages: Union[str, List[Dict[str, str]]]) -> str:
        if not self.model:
            raise RuntimeError("Model not loaded")
        
        if isinstance(messages, str):
            text = messages
        else:
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
            temperature=1.0,
            top_p=1.0,
            top_k=50,
            stopping_criteria=stopping_criteria
        )
        
        if timeout_criteria.timed_out:
            raise TimeoutError("生成超时 (120s)")
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)
        ]
        
        generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return generated_text.strip()

model_server = None

@app.on_event("startup")
async def startup_event():
    global model_server
    model_server = ModelServer(MODEL_CANDIDATES)

@app.post("/generate")
async def generate(messages: Union[str, List[Dict[str, str]]]):
    print("收到生成请求")
    global model_server
    if model_server is None:
        raise HTTPException(status_code=500, detail="Model server not initialized")
    
    try:
        print("正在处理请求...")
        generated_text = model_server.generate(messages)
        print("生成结束")
        return {"text": generated_text}
    except TimeoutError as e:
        print(f"生成超时: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        print(f"Generate error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)
