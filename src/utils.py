import re
import subprocess
import tempfile
import shutil
import time
import torch
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Iterator
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList
from peft import PeftModel
from src.config import TEMP_DIR

ASM_PATTERNS = {
    "x86": re.compile(r'^\s*([0-9a-f]+):\s+((?:[0-9a-f]{2}\s+)+)(.+)$', re.IGNORECASE | re.MULTILINE),
    "arm": re.compile(r'^\s*([0-9a-f]+):\s+((?:[0-9a-f]{8}\s+)?)(.+)$', re.IGNORECASE | re.MULTILINE)
}

DISASM_PATTERNS = {
    "x86": re.compile(r'^\s*[0-9a-f]+:\s+(?:[0-9a-f]{2}\s+)+(.+)$', re.IGNORECASE | re.MULTILINE),
    "arm": re.compile(r'^\s*[0-9a-f]+:\s+(?:[0-9a-f]{8}\s+)?(.+)$', re.IGNORECASE | re.MULTILINE)
}

class ModelRunning:
    def __init__(
        self, 
        base_model_path: str,
        sft_adapter_path: Optional[str] = None,
        dpo_adapter_path: Optional[str] = None,
        offload_folder: str = None,
        offload_buffers: bool = False,
        **kwargs
    ):
        self.model, self.tokenizer = self._load_model_tokenizer(
            base_model_path=base_model_path, 
            offload_folder=offload_folder,
            offload_buffers=offload_buffers,
            **kwargs
        )
        
        # 准备 PeftModel 的通用参数
        peft_kwargs = {}
        if offload_folder:
            peft_kwargs["offload_folder"] = offload_folder

        if sft_adapter_path and dpo_adapter_path:
            print(f"同时加载 SFT 和 DPO 适配器...")
            # 1. 加载 SFT 作为基础
            print(f"加载 SFT: {sft_adapter_path}")
            self.model = PeftModel.from_pretrained(
                self.model, 
                sft_adapter_path, 
                adapter_name="sft_adapter",
                **peft_kwargs
            )
            
            # 2. 加载 DPO
            print(f"加载 DPO: {dpo_adapter_path}")
            self.model.load_adapter(dpo_adapter_path, adapter_name="dpo_adapter")
            
            # 3. 合并
            print("合并适配器 (Weights: 1.0, 1.0)...")
            self.model.add_weighted_adapter(
                adapters=["sft_adapter", "dpo_adapter"],
                weights=[1.0, 1.0],
                adapter_name="combined",
                combination_type="cat" 
            )
            self.model.set_adapter("combined")
            
        elif sft_adapter_path:
            print(f"加载 SFT 适配器: {sft_adapter_path}")
            self.model = PeftModel.from_pretrained(
                self.model, 
                sft_adapter_path, 
                adapter_name="sft_adapter",
                **peft_kwargs
            )
            self.model.set_adapter("sft_adapter")

    @staticmethod
    def _load_model_tokenizer(
        base_model_path: str,
        offload_folder: Optional[str] = None,
        offload_buffers: bool = False,
        **kwargs
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        加载模型和分词器 (内部静态方法)
        """
        try:
            print("启用 4-bit 量化配置...")
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                llm_int8_enable_fp32_cpu_offload=True
            )

            tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model_kwargs = {
                "pretrained_model_name_or_path": base_model_path,
                "torch_dtype": torch.bfloat16,
                "trust_remote_code": True,
                "quantization_config": quant_config,
                "device_map": "cuda:0",
                "attn_implementation": "flash_attention_2" if torch.cuda.is_available() else None,
            }
            
            if offload_folder:
                model_kwargs["offload_folder"] = offload_folder
            
            if offload_buffers:
                model_kwargs["offload_buffers"] = True
            
            # 合并其他参数
            model_kwargs.update(kwargs)
            
            model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
            
            print(f"模型加载完成")
            return model, tokenizer
        except Exception as e:
            print(f"模型和分词器加载失败: {e}")
            raise ValueError(f"模型和分词器加载失败: {e}")

    def generate(
        self, 
        messages: Union[str, List[Dict[str, str]], List[List[Dict[str, str]]], List[str]], 
        remaining_time: Optional[float] = 120.0,
        **kwargs
    ) -> Union[str, List[str]]:
        if not self.model:
            raise RuntimeError("模型未加载")

        # 判断是否为批量输入
        is_batch = False
        if isinstance(messages, list) and len(messages) > 0:
            # 如果列表第一个元素是列表(对话历史)或字符串(Prompt)，则视为批量
            if isinstance(messages[0], list) or isinstance(messages[0], str):
                is_batch = True
                # 特例: 如果是 List[Dict] 且 key 是 role/content，则是单个对话历史
                if isinstance(messages[0], dict):
                    is_batch = False
        
        prompts = []
        if is_batch:
            for msg in messages:
                if isinstance(msg, str):
                    prompts.append(msg)
                else:
                    prompts.append(self.tokenizer.apply_chat_template(
                        msg, tokenize=False, add_generation_prompt=True
                    ))
        else:
            if isinstance(messages, str):
                prompts.append(messages)
            else:
                prompts.append(self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                ))

        device = "cuda" if torch.cuda.is_available() else self.model.device
        
        # 生成时需要左填充
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"

        try:
            model_inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(device)

            stopping_criteria = StoppingCriteriaList()
            if remaining_time is not None:
                timeout_criteria = TimeoutStoppingCriteria(remaining_time)
                stopping_criteria.append(timeout_criteria)

            # 默认生成参数
            gen_kwargs = {
                "max_new_tokens": 2048,
                "do_sample": False,
                "temperature": None,
                "top_p": None,
                "top_k": None,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "use_cache": True,
                "stopping_criteria": stopping_criteria
            }
            # 更新用户自定义参数
            gen_kwargs.update(kwargs)

            # print("开始生成...")
            generated_ids = self.model.generate(
                **model_inputs,
                **gen_kwargs
            )

            # print("生成完成")
            input_len = model_inputs.input_ids.shape[1]
            generated_ids = [
                output_ids[input_len:] for output_ids in generated_ids
            ]
            
            # print("解码生成结果")
            responses = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            if remaining_time is not None and timeout_criteria.timed_out:
                print(f"生成超时 ({remaining_time}s)")
                # 注意：批量生成时，只要有一个超时就算超时，或者需要更复杂的处理
                # 这里简单处理，如果超时则抛出异常
                raise TimeoutError(f"生成超时 ({remaining_time}s)")

            # 根据输入类型返回
            if is_batch:
                return [r.strip() for r in responses]
            else:
                return responses[0].strip()
            
        except Exception as e:
            print(f"生成出错: {e}")
            return [] if is_batch else ""
        finally:
            self.tokenizer.padding_side = original_padding_side

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

def compile_to_object(arch: str, c_code: str) -> tuple:
    """
    将 C 函数代码编译为 .o文件
    """
    workdir = None
    try:
        cc, _, _ = get_compiler_config(arch)
        
        workdir = tempfile.mkdtemp(dir=str(TEMP_DIR))
        
        c_path = Path(workdir) / "func.c"
        c_path.write_text(c_code, encoding="utf-8")
        
        o_path = Path(workdir) / "func.o"

        cmd = [cc, "-c", str(c_path), "-o", str(o_path)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode != 0:
            error_msg = result.stderr.strip() or result.stdout.strip() or "未知编译错误"
            if workdir and Path(workdir).exists():
                shutil.rmtree(workdir, ignore_errors=True)
            return False, error_msg, None

        return True, None, str(o_path)
    except Exception as e:
        if workdir and Path(workdir).exists():
            shutil.rmtree(workdir, ignore_errors=True)
        return False, str(e), None

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
        if arch not in ASM_PATTERNS:
            return None, None
        
        pattern = ASM_PATTERNS[arch]
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

def machine_code_to_binary(machine_code: str) -> Path:
    """
    将机器码字符串写入二进制文件
    """
    try:
        hex_str = machine_code.replace(" ", "")
        binary_data = bytes.fromhex(hex_str)
        
        workdir = tempfile.mkdtemp(dir=str(TEMP_DIR))

        binary_path = Path(workdir) / "temp.bin"
        binary_path.write_bytes(binary_data)
        
        return binary_path
    except Exception as e:
        if binary_path and binary_path.exists():
            binary_path.unlink()
        return None

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
        if arch not in DISASM_PATTERNS:
            return None
        
        pattern = DISASM_PATTERNS[arch]
        asm_lines = []
        
        for match in pattern.finditer(disasm_output):
            asm = match.group(1).strip()
            if asm:
                asm_lines.append(asm)
        
        return '\n'.join(asm_lines) if asm_lines else None
        
    except Exception:
        return None

def extract_compilation_data(item: Dict) -> Iterator[Tuple[str, str, str, Optional[str]]]:
    """
    从数据集条目中提取 (架构, 汇编, C代码, 机器码) 四元组
    """
    if not item.get("c_code") or not item.get("compilations"):
        return

    c_code = item["c_code"]
    compilations = item["compilations"]

    for arch in ["x86", "arm"]:
        if arch in compilations and compilations[arch] and "asm" in compilations[arch]:
            asm = compilations[arch]["asm"]
            machine_code = compilations[arch]["machine_code"]
            if asm:
                yield arch, asm, c_code, machine_code

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