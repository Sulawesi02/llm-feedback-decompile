import uvicorn
import torch
import gc
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
from contextlib import asynccontextmanager
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from config import (
    MODEL_DIR, 
    MODEL_NAME,
    DPO_DIR,
    MAX_PROMPT_TOKENS,
    MAX_GEN_TOKENS,
)
from compiler import ( 
    write_machine_code_to_bin,
    disasm_bin,
    extract_asm,
    compile_to_obj,
)
from prompts import construct_infer_prompt, construct_fix_prompt

MAX_ITERS = 3 # 最大迭代次数

@asynccontextmanager
async def lifespan(app: FastAPI):
    base_model_path = MODEL_DIR / MODEL_NAME
    
    print("加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(
        str(base_model_path), 
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    
    print("加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        str(base_model_path),
        trust_remote_code=True,
        device_map={"": torch.cuda.current_device()},
        torch_dtype=torch.bfloat16,
    )
    
    if DPO_DIR.exists():
        print(f"加载 DPO 适配器: {DPO_DIR}")
        model = PeftModel.from_pretrained(
            model,
            str(DPO_DIR),
            device_map={"": torch.cuda.current_device()},
        )
    
    model.eval() # 推理模式
    
    # 存储到 app.state
    app.state.tokenizer = tokenizer
    app.state.model = model
    
    try:
        yield
    finally:
        # 清理资源
        del app.state.tokenizer
        del app.state.model
        gc.collect()
        torch.cuda.empty_cache()

app = FastAPI(lifespan=lifespan)

class DecompileRequest(BaseModel):
    machine_code: str

@app.post("/decompile")
async def decompile(request: Request, body: DecompileRequest):
    if not body.machine_code.strip():
        raise HTTPException(status_code=400, detail="机器码不能为空")
    
    # 从 app.state 获取模型和分词器
    tokenizer = request.app.state.tokenizer
    model = request.app.state.model
    
    machine_code = body.machine_code
    
    outputs = None
    prev_outputs = None
    last_error = None
    
    history = []
    best_outputs = None
    
    for it in range(MAX_ITERS):
        print(f"\n--- 迭代 {it+1}/{MAX_ITERS} ---")
        
        try:
            # ========== 反汇编阶段 ==========
            bin_path = write_machine_code_to_bin(machine_code)            
            disasm_result = disasm_bin(bin_path)
            asm = extract_asm(disasm_result)
            # ========== 生成阶段 ==========
            if it == 0:
                print("构造推理提示...")
                messages = construct_infer_prompt(asm)
            else:
                print("构造修复提示...")
                messages = construct_fix_prompt(asm, prev_outputs, last_error)
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)            
            token_count = len(tokenizer.encode(text))
            
            if token_count > MAX_PROMPT_TOKENS:
                print(f"输入长度 {token_count} 超过最大长度")
                history.append({
                    "iter": it,
                    "success": False,
                    "outputs": "",
                    "error": f"prompt too long ({token_count} > {MAX_PROMPT_TOKENS})"
                })
                break
            
            gen_inputs = tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=MAX_PROMPT_TOKENS,
            ).to(model.device)

            print("生成 C 函数代码...")
            with torch.no_grad():
                outputs = model.generate(
                    **gen_inputs,
                    max_new_tokens=MAX_GEN_TOKENS,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.95,
                    use_cache=True,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )
            output_text = tokenizer.decode(outputs[0][gen_inputs.input_ids.shape[1]:], skip_special_tokens=True).strip().replace("```cpp", "").replace("```c", "").replace("```", "").strip()
            print(f"生成的 C 函数代码:\n{output_text}")
            # ========== 编译阶段 ==========
            print("编译 C 函数代码...")
            success, error_msg = compile_to_obj(output_text)
            if success:
                print("编译成功")
                best_outputs = output_text
                history.append({
                    "iter": it,
                    "success": True,
                    "outputs": output_text,
                    "error": "",
                })
                break
            else:
                print(f"编译失败: {error_msg.splitlines()[0]}")
                history.append({
                    "iter": it,
                    "success": False,
                    "outputs": output_text,
                    "error": error_msg,
                })
                prev_outputs = output_text
                last_error = error_msg
                continue
        except Exception as e:
            print(f"生成 C 函数代码出错: {e}")
            history.append({
                "iter": it,
                "success": False,
                "outputs": "",
                "error": f"生成 C 函数代码出错: {e}",
            })
            break
    print(f"任务结束 | 最终状态: {'成功' if best_outputs else '失败'}")
    
    return {
        "success": best_outputs is not None,
        "best_outputs": best_outputs
    }

# 静态文件服务
STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
