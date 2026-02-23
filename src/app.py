import uvicorn
import torch
import gc
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
from contextlib import asynccontextmanager
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from config import (
    MODEL_DIR, 
    MODEL_NAME,
    DPO_DIR,
    MAX_PROMPT_TOKENS,
    MAX_GEN_TOKENS,
)
from compiler import write_machine_code_to_bin, disasm_bin, extract_asm, compile_test_compare, compile_to_obj
from llm_utils import clean_output
from prompts import build_generate_text, build_test_text

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

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    
    print("加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        str(base_model_path),
        trust_remote_code=True,
        device_map={"": torch.cuda.current_device()},
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
    )
    
    if DPO_DIR.exists():
        print(f"挂载 DPO 适配器: {DPO_DIR}")
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
    
    best_outputs = None
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
            sample = {
                "asm": asm,
                "prev_outputs": prev_outputs,
                "last_error": last_error,
            }
            gen_text = build_generate_text(it, tokenizer, sample, MAX_PROMPT_TOKENS)
            if gen_text is None:
                history.append({
                    "iter": it,
                    "success": False,
                    "outputs": "",
                    "error": "generate prompt too long",
                })
                break
            gen_inputs = tokenizer(
                gen_text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=MAX_PROMPT_TOKENS,
            ).to(model.device)

            print("生成 C 函数代码...")
            with torch.no_grad():
                gen_outputs = model.generate(
                    **gen_inputs,
                    max_new_tokens=MAX_GEN_TOKENS,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.95,
                    use_cache=True,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )
            gen_outputs = tokenizer.decode(gen_outputs[0][gen_inputs.input_ids.shape[1]:], skip_special_tokens=True)
            output_text = clean_output(gen_outputs)
            print(f"生成的 C 函数代码:\n{output_text}")
            
            func_ok, func_error = compile_to_obj(output_text)
            if not func_ok:
                print(f"候选 C 函数编译失败: {func_error.splitlines()[0]}")
                history.append({
                    "iter": it,
                    "success": False,
                    "outputs": output_text,
                    "error": f"func 编译失败: {func_error}",
                })
                prev_outputs = output_text
                last_error = f"func 编译失败: {func_error}"
                continue
            
            test_text = build_test_text(tokenizer, output_text, MAX_PROMPT_TOKENS)
            if test_text is None:
                history.append({
                    "iter": it,
                    "success": False,
                    "outputs": output_text,
                    "error": "test prompt too long",
                })
                prev_outputs = output_text
                last_error = "test prompt too long"
                continue
            test_inputs = tokenizer(
                test_text,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_PROMPT_TOKENS,
            ).to(model.device)
            
            print("生成测试代码...")
            with torch.no_grad():
                test_outputs = model.generate(
                    **test_inputs,
                    max_new_tokens=MAX_GEN_TOKENS,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.95,
                    use_cache=True,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )
            test_output = tokenizer.decode(test_outputs[0][test_inputs.input_ids.shape[1]:], skip_special_tokens=True)
            test_code = clean_output(test_output)
            print(f"生成的测试代码:\n{test_code}")
            print("编译并运行测试用例...")
            success, error_msg = compile_test_compare(asm, output_text, test_code)
            if success:
                print("测试通过")
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
