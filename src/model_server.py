import uvicorn
import os
import torch
import time
from fastapi import FastAPI, HTTPException
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList
from typing import List, Dict, Any, Union
from pathlib import Path
from pydantic import BaseModel

from config import MERGED_MODEL_DIR_PATH, BASE_MODEL_DIR_PATH
from utils import load_model_utils, create_timeout_stopping_criteria

app = FastAPI()

class GenerateRequest(BaseModel):
    messages: Union[str, List[Dict[str, str]]]

MODEL_CANDIDATES = [
    # str(MERGED_MODEL_DIR_PATH / "v2"),
    # str(MERGED_MODEL_DIR_PATH / "v1"),
    str(BASE_MODEL_DIR_PATH)
]

class ModelServer:
    def __init__(self, model_paths: List[str]):
        self.model_paths = model_paths
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        print(f"可用模型候选列表: {self.model_paths}")
        for model_path in self.model_paths:
            try:
                model , self.tokenizer = load_model_utils(model_path)
                self.model = model.eval()
                return
            except Exception as e:
                print(f"加载模型失败: {e}")
                
        raise RuntimeError("所有模型路径均加载失败，无法启动服务")

    def generate(self, messages: Union[str, List[Dict[str, str]]]) -> str:
        if not self.model:
            raise RuntimeError("Model not loaded")
        
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
                temperature=None,
                top_p=None,
                top_k=None,
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
                raise TimeoutError("生成超时 (120s)")
            
            return response.strip()
        
        except TimeoutError:
            print("生成超时 (120s)")
            raise
        except Exception as e:
            print(f"生成出错: {e}")
            return ""

model_server = None

@app.on_event("startup")
async def startup_event():
    global model_server
    model_server = ModelServer(MODEL_CANDIDATES)

@app.post("/generate")
async def generate(req: GenerateRequest):
    print("收到生成请求")
    messages = req.messages
    print(f"请求消息: {messages}")
    global model_server
    if model_server is None:
        raise HTTPException(status_code=500, detail="模型服务器未初始化")
    
    try:
        print("正在处理请求...")
        response = model_server.generate(messages)
        print("生成结束")
        return {
            "text": response,
        }
    except TimeoutError as e:
        print(f"生成超时: {e}")
        raise HTTPException(status_code=408, detail="生成超时 (120s)")
    except Exception as e:
        print(f"Generate error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)
