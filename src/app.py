import uvicorn
import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

app = FastAPI()

# 模型路径配置
MODEL_PATH = "/app/model/base/machine_to_c_merged"

# 请求模型
class DecompileRequest(BaseModel):
    arch: str # x86 / arm
    opt: str # -O0 / -O1 / -O2 / -O3
    machine_code: str

model = None
tokenizer = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEMP_DIR = Path("/tmp/decompile_workdir")
TEMP_DIR.mkdir(exist_ok=True)

def construct_initial_prompt(machine_code: str, arch: str, opt: str) -> str:
    return f"""你是一个专业的二进制反编译专家。
请把下面这段 {arch} {opt} 机器码反编译成可读的 C 语言函数：

{machine_code}

要求：
- 输出完整的可编译 C 函数（包含返回值类型、参数如果能推断）
- 使用标准 C 语法，不要依赖特定编译器扩展
- 输出只包含 C 代码，不要任何解释、注释或额外文字

C 代码：
"""

def construct_refine_prompt(previous_c_code: str, compile_error: str) -> str:
    return f"""你之前生成的 C 代码编译失败了。

编译错误信息如下：
{compile_error}

请仔细阅读错误，修复代码中的问题（语法错误、类型错误、未定义符号等），然后重新输出完整的修正后的 C 函数。

要求：
- 只输出修复后的完整 C 代码
- 不要添加任何解释

修正后的 C 代码：
"""

def compile_c_code(c_code: str, arch: str, opt: str) -> dict:
    """编译 C 代码，返回 {'success': bool, 'error': str, 'binary_path': str}"""
    # 选择编译器
    if arch == "x86":
        cc = "gcc"
        args = ["-m64"]
    elif arch == "arm":
        cc = "aarch64-linux-gnu-gcc"
        args = []
    else:
        return {"success": False, "error": f"不支持的架构: {arch}"}

    # 创建临时目录
    workdir = tempfile.mkdtemp(dir=TEMP_DIR)
    c_path = Path(workdir) / "func.c"
    o_path = Path(workdir) / "func.o"

    try:
        c_path.write_text(c_code, encoding="utf-8")

        cmd = [cc] + args + [f"-{opt}", "-c", str(c_path), "-o", str(o_path)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode == 0:
            return {"success": True, "error": None, "binary_path": str(o_path), "workdir": workdir}
        else:
            error_msg = result.stderr.strip()
            if not error_msg:
                error_msg = result.stdout.strip() or "未知编译错误"
            return {"success": False, "error": error_msg, "workdir": workdir}
    except Exception as e:
        return {"success": False, "error": f"编译过程异常: {str(e)}", "workdir": workdir}

def generate_c_code(prompt: str) -> str:
    global model, tokenizer
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=768,
            temperature=0.2,
            do_sample=False,  # 迭代时优先确定性
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True
        )
    generated_ids = outputs[0]
    input_len = inputs.input_ids.shape[1]
    response_ids = generated_ids[input_len:]
    c_code = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
    return c_code

@app.on_event("startup")
async def load_model():
    global model, tokenizer
    print(f"开始加载模型: {MODEL_PATH}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2" if torch.cuda.is_available() else None
        )
        model.eval()
        print("模型加载成功！")
    except Exception as e:
        print(f"加载模型失败: {e}")
        raise e

# 单次反编译
@app.post("/decompile")
async def decompile(request: DecompileRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="模型未加载成功")
    if not request.machine_code.strip():
        raise HTTPException(status_code=400, detail="机器码不能为空")

    prompt = construct_initial_prompt(request.machine_code, request.arch, request.opt)
    c_code = generate_c_code(prompt)
    return {"c_code": c_code or "// 生成失败"}

# 循环反馈反编译
@app.post("/feedback_decompile")
async def feedback_decompile(request: DecompileRequest, max_iters: int = 8):
    if model is None:
        raise HTTPException(status_code=500, detail="模型未加载成功")
    if not request.machine_code.strip():
        raise HTTPException(status_code=400, detail="机器码不能为空")

    machine_code = request.machine_code.strip()
    arch = request.arch
    opt = request.opt

    history = []
    best_c_code = None

    prompt = construct_initial_prompt(machine_code, arch, opt)
    c_code = generate_c_code(prompt)
    history.append({"iter": 0, "c_code": c_code, "status": "generated"})

    for it in range(1, max_iters + 1):
        compile_result = compile_c_code(c_code, arch, opt)
        workdir = compile_result.get("workdir")

        if compile_result["success"]:
            best_c_code = c_code
            history.append({
                "iter": it,
                "c_code": c_code,
                "status": "compile_success",
                "message": "编译成功！"
            })
            break
        else:
            error_msg = compile_result["error"][:1000]  # 截断过长错误
            history.append({
                "iter": it,
                "c_code": c_code,
                "status": "compile_failed",
                "error": error_msg
            })

            refine_prompt = construct_refine_prompt(c_code, error_msg)
            c_code = generate_c_code(refine_prompt)

        # 清理临时目录（无论成功失败）
        if workdir and Path(workdir).exists():
            shutil.rmtree(workdir, ignore_errors=True)

    # 最终返回
    return {
        "final_c_code": best_c_code or c_code,
        "success": best_c_code is not None,
        "total_iterations": len(history),
        "history": history
    }

# 静态文件服务
STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)