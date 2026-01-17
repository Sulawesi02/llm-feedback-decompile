import re
import json
import hashlib
import subprocess
import tempfile
import shutil
import os
import time
import torch
from pathlib import Path
from typing import Dict, List, Optional, Union
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList

TEMP_DIR = Path("/tmp/workdir")

class ModelRunner:
    def __init__(self, model_path: str, timeout_seconds: float = 120.0):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.timeout_seconds = timeout_seconds
        self._load_model()

    def _load_model(self):
        try:
            model, self.tokenizer = load_model_utils(self.model_path)
            self.model = model.eval()
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise RuntimeError("模型加载失败")

    def generate(self, messages: Union[str, List[Dict[str, str]]]) -> str:
        if not self.model:
            raise RuntimeError("模型未加载")

        print("转换为模型输入格式")
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        print("转换为模型输入张量")
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        timeout_criteria = create_timeout_stopping_criteria(self.timeout_seconds)
        stopping_criteria = StoppingCriteriaList([timeout_criteria])

        try:
            print("开始生成 C 代码")
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

            print("生成完成")
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            print("解码生成结果")
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            if timeout_criteria.timed_out:
                print(f"生成超时 ({self.timeout_seconds}s)")
                raise TimeoutError(f"生成超时 ({self.timeout_seconds}s)")

            return response.strip()
        except Exception as e:
            print(f"生成出错: {e}")
            return ""

    def unload(self):
        if self.model:
            del self.model
        if self.tokenizer:
            del self.tokenizer
        torch.cuda.empty_cache()
        print("模型资源已释放")

def get_compiler_config(arch: str) -> tuple:
    """
    获取指定架构的编译器和参数
    """
    if arch == "x86":
        return "gcc", ["-m64"], ["objdump", "-d"]
    elif arch == "arm":
        return "aarch64-linux-gnu-gcc", [], ["aarch64-linux-gnu-objdump", "-d"]
    else:
        raise ValueError(f"不支持的架构: {arch}")

def compile_to_object(c_code: str, arch: str, opt: str) -> dict:
    """
    编译 C 代码为目标文件
    """
    try:
        cc, args, _ = get_compiler_config(arch)
    except ValueError as e:
        return {"success": False, "error": str(e), "workdir": None, "binary_path": None}

    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    try:
        workdir = tempfile.mkdtemp(dir=TEMP_DIR)
    except Exception:
        workdir = tempfile.mkdtemp()

    c_path = Path(workdir) / "func.c"
    o_path = Path(workdir) / "func.o"

    try:
        c_path.write_text(c_code, encoding="utf-8")

        cmd = [cc] + args + [f"-{opt}", "-c", str(c_path), "-o", str(o_path)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode != 0:
            error_msg = result.stderr.strip() or result.stdout.strip() or "未知编译错误"
            return {"success": False, "error": error_msg, "workdir": workdir, "binary_path": None}

        return {"success": True, "error": None, "workdir": workdir, "binary_path": str(o_path)}
    except Exception as e:
        return {"success": False, "error": f"编译过程异常: {str(e)}", "workdir": workdir, "binary_path": None}

def disassemble_object(binary_path: str, arch: str) -> dict:
    """
    反汇编二进制文件，返回汇编代码和机器码
    """
    try:
        _, _, objdump_cmd = get_compiler_config(arch)
    except ValueError as e:
        return {"asm": None, "machine_code": None, "error": str(e)}

    cmd = objdump_cmd + [str(binary_path)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            error_msg = r.stderr.strip() or r.stdout.strip() or "objdump 失败"
            return {"asm": None, "machine_code": None, "error": error_msg}

        bytes_out = []
        asm_lines = []
        for line in r.stdout.splitlines():
            if ":" not in line:
                continue
            _, rest = line.split(":", 1)
            rest = rest.strip()
            if not rest:
                continue
            tokens = rest.split()

            i = 0
            while i < len(tokens):
                token = tokens[i]
                if all(c in "0123456789abcdefABCDEF" for c in token) and (len(token) == 2 or len(token) == 8):
                    if len(token) == 2:
                        bytes_out.append(token.lower())
                    elif len(token) == 8:
                        bytes_out.extend([token.lower()[j:j+2] for j in range(0, 8, 2)])
                    i += 1
                else:
                    break

            asm_part = " ".join(tokens[i:]).strip()
            if asm_part:
                asm_lines.append(asm_part)

        machine_code = " ".join(bytes_out)
        asm_text = "\n".join(asm_lines).strip()
        if not machine_code or not asm_text:
            return {"asm": None, "machine_code": None, "error": "解析 objdump 输出失败"}

        return {"asm": asm_text, "machine_code": machine_code, "error": None}
    except Exception as e:
        return {"asm": None, "machine_code": None, "error": f"反汇编过程异常: {str(e)}"}

def load_jsonl(file_path: Union[str, Path]) -> List[Dict]:
    """
    加载 JSONL 文件
    """
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"jsonl 解析错误: {e}")
    return data

def load_model_utils(model_path: str):
    """
    加载量化模型和分词器
    """
    try:
        print("定义量化配置...")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        print(f"加载 Tokenizer: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"加载模型: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quant_config,
            device_map={"": 0},
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
        
        print(f"模型加载完成: {model_path}")
        return model, tokenizer
    except Exception as e:
        print(f"模型加载失败: {e}")
        raise

def disassemble(machine_code: str, arch: str) -> str:
    """
    objdump 将机器码反汇编为汇编代码
    """
    try:
        # 写入临时二进制文件
        binary_data = bytes.fromhex(machine_code)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
            f.write(binary_data)
            temp_bin = f.name
        
        # 构建 objdump 命令
        if arch == "x86":
            # 假设是 x86-64
            cmd = ["objdump", "-D", "-b", "binary", "-m", "i386:x86-64", "-M", "intel", temp_bin]
        elif arch == "arm":
            # 假设是 aarch64
            cmd = ["aarch64-linux-gnu-objdump", "-D", "-b", "binary", "-m", "aarch64", temp_bin]
        else:
            return f"; 不支持的反汇编架构: {arch}"

        # 执行命令
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        
        # 清理临时文件
        os.unlink(temp_bin)
        
        if result.returncode != 0:
            return f"; 反汇编失败: {result.stderr}"
            
        return result.stdout
    except Exception as e:
        return f"; 反汇编异常: {str(e)}"

def construct_decompile_prompt(arch: str, opt: str, machine_code: str) -> List[Dict[str, str]]:
    """
    构造反编译提示
    """
    # 反汇编
    assembly_code = disassemble(machine_code, arch)

    prompt = f"""请将以下汇编代码反编译成等效的C函数代码。

架构信息:
- 指令集架构: {arch}
- 优化级别: {opt}

核心任务:
分析汇编指令的数据流和控制流，将其还原为高层 C 语言逻辑。

严格禁止:
- 严禁输出 `__asm__`、`asm` 或任何形式的内联汇编。
- 严禁只是简单地将汇编指令包装在 C 函数中。
- 严禁生成 `__builtin_prefetch` 等非标准内置函数。

反编译要求:
1. 抽象层次: 必须将寄存器操作（如 mov, add）转换为 C 语言的变量赋值和算术运算。
2. 类型推断: 根据寄存器宽度推断变量类型。
   - 64位寄存器 (rax, rdi, rsi, rdx...) 通常对应 `long`, `long long` 或指针 `void*`。
   - 32位寄存器 (eax, edi, esi, edx...) 通常对应 `int`。
   - 8位/16位寄存器 (al, ax...) 对应 `char` / `short`。
3. 内存操作: 准确识别指针解引用。
   - `mov (%rdi), %rax` -> `long val = *ptr;`
   - `mov (%rdi), %eax` -> `int val = *ptr;`
   - 注意数组索引: `mov (%rdi, %rcx, 8)` -> `ptr[i]` (这里步长8暗示是 long 数组)。
4. 算术指令:
    - **乘法优化**: 编译器常使用 `lea` 或位移指令代替乘法。
      - `lea (%rax, %rax, 1), %rcx` -> `rcx = rax * 2` (注意：rax+rax*1 = 2*rax)
      - `shl $2, %rax` -> `rax * 4`
    - 请仔细检查每一条算术指令，不要漏掉系数。
 5. 控制流与循环:
    - **固定循环**: 如果 `cmp` 指令比较的是立即数（如 `cmp $0x2, %eax`），且该寄存器是循环计数器，说明这是固定次数的循环（如 `for(i=0; i<=2; i++)`），**不需要**引入额外的 `count` 参数。
    - 还原 `if/else`、`while`、`for` 结构。
6. 函数签名: 根据 ABI 推断参数和返回值。
   - x86-64: rdi, rsi, rdx, rcx, r8, r9 为前6个参数; rax 为返回值。
   - ARM64: x0-x7 为参数; x0 为返回值。
7. 清理代码: 忽略函数序言/结语 (push rbp, mov rbp, rsp ... pop rbp, ret)。

输出格式要求:
- 仅输出一个 C 函数定义。
- 不要包含 markdown 代码块标记（如 ```c ... ```），直接输出代码。
- 不要包含任何解释性文字。

汇编代码:
{assembly_code}"""

    messages = [
        {
            "role": "system", 
            "content": """你是一个高级 C 语言反编译专家，目标是：
    1. 严格保持语义正确；
    2. 生成“人类工程师愿意维护”的高质量 C 代码。

    整体风格要求：
    - 代码要简洁、结构清晰，比机械翻译汇编更“高层”；
    - 优先使用数组下标、结构体字段访问，而不是生硬的指针算术；
    - 合理推断并使用基础类型：int / long / size_t / 指针等；
    - 使用有意义的函数名与变量名（如 src/dst/len/index/sum），避免 a1/a2/tmp1/tmp2。

    具体约束：
    - 控制流：尽量还原为标准的 if / else / for / while 结构，避免 goto；
    - 循环：
    - 如果 cmp 比较的是小的常数且作为计数器使用，优先还原为固定次数的 for 循环；
    - 不要引入不必要的额外参数（例如无意义的 count 参数），除非从调用约定中明显需要。
    - 内存访问：
    - 能写成 arr[i] 的地方不要写成 *(arr + i * k)；
    - 指针遍历时，优先写成 for (int i = 0; i < n; ++i) 这种形式。
    - 返回值：
    - 如果汇编清晰地返回某个值，就按该语义还原；
    - 不要为了“凑个返回值”随意返回计数器或硬编码的常数。

    输出格式（非常重要）：
    - 只输出一个完整的 C 函数定义；
    - 严禁输出任何 Markdown 代码块标记（如 ```c ... ``` 或 ```）；
    - 严禁输出任何解释性文字、示例说明或注释，只保留纯代码。
    """
        },
        {
            "role": "user", 
            "content": prompt
        }
    ]
    return messages

def construct_refine_decompile_prompt(previous_c_code: str, compile_error: str) -> List[Dict[str, str]]:
    """
    构造修正反编译提示
    """
    messages = [
        {"role": "system", "content": "你是一个专业的二进制反编译专家。"},
        {"role": "user", "content": f"""你之前生成的 C 代码编译失败了。

代码如下：
{previous_c_code}

编译错误信息如下：
{compile_error}

请仔细阅读错误，修复代码中的问题。
要求：
- 只输出修复后的完整 C 代码
- 严禁输出 Markdown 代码块标记
- 严禁输出任何解释"""}
    ]
    return messages

def create_timeout_stopping_criteria(timeout_seconds: float):
    """
    创建超时停止条件
    """
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

    return TimeoutStoppingCriteria(timeout_seconds)

def extract_function_signature(c_code: str) -> Optional[dict]:
    """
    从 C 代码中提取函数签名
    """
    try:
        for line in c_code.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            m = re.match(r"([_a-zA-Z][\w\s\*\d]*?)\s+([_a-zA-Z]\w*)\s*\(([^)]*)\)\s*{?", stripped)
            if m:
                return {
                    "return_type": m.group(1).strip(),
                    "name": m.group(2),
                    "params": m.group(3).strip(),
                }
        return None
    except Exception as e:
        return None

def build_test_harness(c_code: str, test_cases: List[dict]) -> Optional[str]:
    """
    用测试用例构建测试代码
    """
    try:
        sig = extract_function_signature(c_code)
    except Exception as e:
        return None
    if not sig:
        return None
    return_type = sig["return_type"]
    name = sig["name"]
    params = sig["params"]

    lines = []
    lines.append("#include <stdio.h>")
    lines.append("#include <stdlib.h>")
    lines.append("#include <string.h>")
    lines.append("#include <assert.h>")
    lines.append(f"{return_type} {name}({params});")
    lines.append("int main() {")
    for case in test_cases:
        args = case.get("args")
        expected = case.get("expected")
        if not isinstance(args, list) or expected is None:
            continue
        args_str = ", ".join(str(a) for a in args)
        lines.append(f"assert({name}({args_str}) == ({expected}));")
    lines.append("return 0;")
    lines.append("}")
    return "\n".join(lines)

def run_tests_with_harness(obj_path: str, arch: str, test_harness_code: str, timeout_seconds: float = 5.0) -> dict:
    """
    运行包含 main 函数的 C 测试代码，验证反编译函数的行为
    """
    try:
        cc, args, _ = get_compiler_config(arch)
    except ValueError as e:
        return {"success": False, "error": str(e), "stdout": "", "stderr": ""}

    workdir = Path(obj_path).parent
    harness_path = workdir / "test_harness.c"
    exe_path = workdir / "test_exec"

    try:
        harness_path.write_text(test_harness_code, encoding="utf-8")

        compile_cmd = [cc] + args + [str(obj_path), str(harness_path), "-o", str(exe_path)]
        compile_result = subprocess.run(compile_cmd, capture_output=True, text=True, timeout=30)

        if compile_result.returncode != 0:
            error_msg = compile_result.stderr.strip() or compile_result.stdout.strip() or "测试代码编译失败"
            return {
                "success": False,
                "error": error_msg,
                "stdout": compile_result.stdout,
                "stderr": compile_result.stderr,
            }

        if arch == "arm":
            run_cmd = ["qemu-aarch64", str(exe_path)]
        else:
            run_cmd = [str(exe_path)]

        run_result = subprocess.run(run_cmd, capture_output=True, text=True, timeout=timeout_seconds)

        if run_result.returncode == 0:
            return {
                "success": True,
                "error": None,
                "stdout": run_result.stdout,
                "stderr": run_result.stderr,
            }

        msg = run_result.stderr.strip() or run_result.stdout.strip() or f"测试运行失败，退出码 {run_result.returncode}"
        return {
            "success": False,
            "error": msg,
            "stdout": run_result.stdout,
            "stderr": run_result.stderr,
        }
    except Exception as e:
        return {"success": False, "error": f"测试执行异常: {str(e)}", "stdout": "", "stderr": ""}
