import uvicorn
from fastapi import FastAPI, HTTPException
from typing import List, Dict, Union
from pydantic import BaseModel

from config import BASE_MODEL_DIR_PATH, MERGED_MODEL_DIR_PATH
from utils import ModelRunner

app = FastAPI()

class GenerateRequest(BaseModel):
    messages: Union[str, List[Dict[str, str]]]

# MODEL_PATH = str(BASE_MODEL_DIR_PATH / "Qwen2.5-Coder-7B-Instruct")
MODEL_PATH = str(MERGED_MODEL_DIR_PATH / "Qwen2.5-Coder-7B-Instruct" / "v1")

model_runner = None

@app.on_event("startup")
async def startup_event():
    global model_runner
    model_runner = ModelRunner(MODEL_PATH)

@app.post("/generate")
async def generate(req: GenerateRequest):
    messages = req.messages
    print(f"收到生成请求: {messages}")
    global model_runner
    if model_runner is None:
        raise HTTPException(status_code=500, detail="模型服务器未初始化")
    
    try:
        print("处理请求...")
        response = model_runner.generate(messages)
        print("生成结束")
        return {"text": response}
    except TimeoutError as e:
        print(f"生成超时: {e}")
        raise HTTPException(status_code=408, detail="生成超时 (120s)")
    except Exception as e:
        print(f"Generate error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)
