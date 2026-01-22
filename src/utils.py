import re
import json
import subprocess
import tempfile
import time
import torch
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList

TEMP_DIR = Path(tempfile.gettempdir()) / "workdir"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

class ModelRunner:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        try:
            model, self.tokenizer = load_model_utils(self.model_path)
            self.model = model.eval()
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise RuntimeError("模型加载失败")

    def generate(self, messages: Union[str, List[Dict[str, str]]], timeout: float = 120.0) -> str:
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

        timeout_criteria = TimeoutStoppingCriteria(timeout)
        stopping_criteria = StoppingCriteriaList([timeout_criteria])

        try:
            print("开始生成 C 函数代码")
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

            if timeout_criteria.timed_out or (time.time() - timeout_criteria.start_time > timeout):
                print(f"生成超时 ({timeout}s)")
                raise TimeoutError(f"生成超时 ({timeout}s)")

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

class TimeoutStoppingCriteria(StoppingCriteria):
    def __init__(self, timeout_seconds: float):
        self.start_time = time.time()
        self.timeout_seconds = timeout_seconds
        self.timed_out = False

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return time.time() - self.start_time > self.timeout_seconds

def get_compiler_config(arch: str) -> tuple:
    """
    获取指定架构的编译器和反汇编工具
    """
    if arch == "x86":
        return "gcc", "objdump", "i386:x86-64"
    elif arch == "arm":
        return "aarch64-linux-gnu-gcc", "aarch64-linux-gnu-objdump", "aarch64"
    else:
        raise ValueError(f"不支持的架构: {arch}")

def compile_to_object(arch: str, c_code: str) -> dict:
    """
    将 C 函数代码编译为 .o文件
    """
    workdir = None
    try:
        cc, _, _ = get_compiler_config(arch)
        
        try:
            workdir = tempfile.mkdtemp(dir=TEMP_DIR)
        except Exception:
            workdir = tempfile.mkdtemp()

        c_path = Path(workdir) / "func.c"
        o_path = Path(workdir) / "func.o"
        c_path.write_text(c_code, encoding="utf-8")

        cmd = [cc, "-c", str(c_path), "-o", str(o_path)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode != 0:
            error_msg = result.stderr.strip() or result.stdout.strip() or "未知编译错误"
            return {
                "success": False, 
                "error": error_msg, 
                "workdir": workdir, 
                "object_path": None
            }

        return {
            "success": True, 
            "error": None, 
            "workdir": workdir, 
            "object_path": str(o_path)
        }
    except Exception as e:
        return {
            "success": False, 
            "error": str(e), 
            "workdir": workdir, 
            "object_path": None
        }

def disasm_object(arch: str, binary_path: str) -> str:
    """
    反汇编 .o 文件
    """
    try:
        _, objdump_cmd, _ = get_compiler_config(arch)
        cmd = [objdump_cmd, "-d", binary_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            return None
        
        return result.stdout
    except Exception:
        return None

def extract_asm_and_machine(arch: str, disasm_output: str) -> Tuple[Optional[str], Optional[str]]:
    """
    从反汇编结果中提取汇编代码和机器码
    """
    try:
        # 匹配机器码 + 汇编代码
        patterns = {
            "x86": re.compile(r'^\s*([0-9a-f]+):\s+((?:[0-9a-f]{2}\s+)+)(.+)$', re.IGNORECASE | re.MULTILINE),
            "arm": re.compile(r'^\s*([0-9a-f]+):\s+((?:[0-9a-f]{8}\s+)?)(.+)$', re.IGNORECASE | re.MULTILINE)
        }
        
        if arch not in patterns:
            return None, None
        
        pattern = patterns[arch]
        asm_lines = []
        machine_bytes = []
        
        for match in pattern.finditer(disasm_output):
            # address = match.group(1).strip() # 地址
            machine = match.group(2).strip() # 机器码
            asm = match.group(3).strip() # 汇编代码
            
            if asm:
                asm_lines.append(asm)
            
            if machine:
                if arch == "x86":
                    # x86: "f3 0f 1e fa" -> ["f3", "0f", "1e", "fa"]
                    bytes_list = machine.split()
                    bytes_list = [b.lower() for b in bytes_list if len(b) == 2]
                    machine_bytes.extend(bytes_list)
                elif arch == "arm":
                    # arm: "d10043ff" -> ["ff", "43", "00", "d1"] (小端转换)
                    if len(machine) == 8:
                        word = machine.lower()
                        bytes_list = [word[i:i+2] for i in range(6, -1, -2)]
                        machine_bytes.extend(bytes_list)
        
        asm_result = '\n'.join(asm_lines) if asm_lines else None
        machine_result = ' '.join(machine_bytes) if machine_bytes else None
        
        return asm_result, machine_result
        
    except Exception:
        return None, None

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

def machine_code_to_binary(machine_code: str) -> Path:
    """
    将机器码字符串写入二进制文件
    """
    hex_str = machine_code.replace(" ", "")
    binary_data = bytes.fromhex(hex_str)
    
    try:
        workdir = tempfile.mkdtemp(dir=TEMP_DIR)
    except Exception:
        workdir = tempfile.mkdtemp()

    temp_bin = Path(workdir) / "temp.bin"
    temp_bin.write_bytes(binary_data)
    
    return temp_bin

def disasm_binary(arch: str, binary_path: str) -> str:
    """
    反汇编 .bin 文件
    """
    try:
        _, objdump_cmd, arch_flag = get_compiler_config(arch)
        cmd = [objdump_cmd, "-D", "-b", "binary", "-m", arch_flag, binary_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        
        if result.returncode != 0:
            return None
        
        return result.stdout
    except Exception:
        return None

def extract_asm(arch: str, disasm_output: str) -> str:
    """
    从反汇编结果中提取汇编代码
    """
    try:
        # 匹配汇编代码
        patterns = {
            "x86": re.compile(r'^\s*[0-9a-f]+:\s+(?:[0-9a-f]{2}\s+)+(.+)$', re.IGNORECASE | re.MULTILINE),
            "arm": re.compile(r'^\s*[0-9a-f]+:\s+(?:[0-9a-f]{8}\s+)?(.+)$', re.IGNORECASE | re.MULTILINE)
        }
        
        if arch not in patterns:
            return None
        
        pattern = patterns[arch]
        asm_lines = []
        
        for match in pattern.finditer(disasm_output):
            asm = match.group(1).strip()
            if asm:
                asm_lines.append(asm)
        
        return '\n'.join(asm_lines) if asm_lines else None
        
    except Exception:
        return None

def construct_decompile_prompt(arch: str, assembly_code: str, previous_c_code: str = None, error_message: str = None) -> List[Dict[str, str]]:
    """
    构造汇编代码到 C 函数代码的反编译提示
    """
    
    system_prompt = f"""你是一个高级 C 语言反编译专家，目标是：
1. 严格保持语义正确；
2. 生成“人类工程师愿意维护”的高质量 C 函数代码。
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
- **类型推断（重要）**：
    - **严禁臆造具体的结构体名称**（如 `struct my_struct`），除非你能同时给出该结构体的完整定义。
    - 如果看不出具体结构体成员，**必须**使用 `long*`、`int*` 或 `void*` 等基本类型指针进行操作。
- 返回值：
    - 如果汇编清晰地返回某个值，就按该语义还原；
    - 不要为了“凑个返回值”随意返回计数器或硬编码的常数。
输出格式：
- 只输出一个完整的 C 函数定义；
- 严禁输出任何 Markdown 代码块标记（如 ```c ... ``` 或 ```）；
- 严禁输出任何解释性文字、示例说明或注释，只保留纯代码。
"""
    if previous_c_code and error_message:
        messages = [
            {
                "role": "system", 
                "content": system_prompt
            },
            {
                "role": "user", 
                "content": f"""你之前生成的 C 函数代码出现错误。
代码如下：
{previous_c_code}

错误信息如下：
{error_message}

请仔细阅读错误，修复代码中的问题。
输出格式：
- 只输出修复后的一个完整的 C 函数定义；
- 严禁输出任何 Markdown 代码块标记（如 ```c ... ``` 或 ```）；
- 严禁输出任何解释性文字、示例说明或注释，只保留纯代码。
"""
            }
        ]
        return messages
        
    user_prompt = f"""请将以下汇编代码反编译成等效的C函数代码。
架构信息:
- 指令集架构: {arch}
核心任务:
分析汇编指令的数据流和控制流，将其还原为高层 C 语言逻辑。
仔细观察寄存器的使用情况，推断函数参数和返回值。
汇编代码:
{assembly_code}
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
    - **注意**: 如果汇编代码中使用了上述寄存器（如 `mov %rdi, -0x8(%rbp)` 或 `str x0, [sp, #8]`），说明函数**一定有参数**。严禁在使用了参数寄存器的情况下生成 `void func(void)`。
7. 清理代码: 忽略函数序言/结语 (push rbp, mov rbp, rsp ... pop rbp, ret)。
输出格式要求:
- 仅输出一个 C 函数定义。
- 不要包含 markdown 代码块标记（如 ```c ... ```），直接输出代码。
- 不要包含任何解释性文字。

"""

    messages = [
        {
                "role": "system", 
                "content": system_prompt
            },
        {
            "role": "user", 
            "content": user_prompt
        }
    ]
    return messages

def clean_code_block(code: str) -> str:
    """
    清理 C 函数代码块标记
    """
    code = code.strip()
    # 移除开头的 ```c 或 ```
    if code.startswith("```"):
        code = code.split("\n", 1)[1]
    # 移除结尾的 ```
    if code.endswith("```"):
        code = code.rsplit("\n", 1)[0]
    return code.strip()

def extract_function_signature(c_code: str) -> Optional[dict]:
    """
    从 C 函数代码中提取函数签名
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
    构建测试代码
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

def run_test_harness(arch: str, obj_path: str, test_harness_code: str, timeout_seconds: float = 5.0) -> dict:
    """
    运行测试代码
    """
    try:
        cc, _, _ = get_compiler_config(arch)
    except ValueError as e:
        return {
            "success": False, 
            "error": str(e), 
            "stdout": "", 
            "stderr": ""
        }

    workdir = Path(obj_path).parent
    harness_path = workdir / "test_harness.c"
    exe_path = workdir / "test_exec"

    try:
        harness_path.write_text(test_harness_code, encoding="utf-8")

        compile_cmd = [cc] + [str(obj_path), str(harness_path), "-o", str(exe_path)]
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
