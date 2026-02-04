import re
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple, Iterator

# 工作目录
TEMP_DIR = Path(tempfile.gettempdir()) / "workdir"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

ASM_PATTERNS = {
    "x86": re.compile(r'^\s*([0-9a-f]+):\s+((?:[0-9a-f]{2}\s+)+)(.+)$', re.IGNORECASE | re.MULTILINE),
    "arm": re.compile(r'^\s*([0-9a-f]+):\s+((?:[0-9a-f]{8}\s+)?)(.+)$', re.IGNORECASE | re.MULTILINE)
}

DISASM_PATTERNS = {
    "x86": re.compile(r'^\s*[0-9a-f]+:\s+(?:[0-9a-f]{2}\s+)+(.+)$', re.IGNORECASE | re.MULTILINE),
    "arm": re.compile(r'^\s*[0-9a-f]+:\s+(?:[0-9a-f]{8}\s+)?(.+)$', re.IGNORECASE | re.MULTILINE)
}

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

def compile_to_obj(arch: str, c_code: str) -> tuple:
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

def disasm_obj(arch: str, bin_path: str) -> str:
    """
    反汇编 .o 文件
    """
    try:
        _, objdump_cmd, _ = get_compiler_config(arch)
        cmd = [objdump_cmd, "-d", bin_path]
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

def extract_data(item: Dict) -> Iterator[Tuple[str, str, str, Optional[str]]]:
    """
    从数据集条目中提取 (C代码, 架构, 汇编, 机器码) 四元组
    """
    c_code = item.get("c_code")
    compilations = item.get("compilations", {})
    
    for arch, info in compilations.items():
        if not info:
            continue
        
        asm = info.get("asm")
        machine_code = info.get("machine_code")
        if c_code and asm and machine_code:
            yield c_code, arch, asm, machine_code

def write_machine_code_to_bin(machine_code: str) -> Path:
    """
    将机器码字符串写入二进制文件
    """
    bin_path = None
    try:
        hex_str = machine_code.replace(" ", "")
        bin_data = bytes.fromhex(hex_str)
        
        workdir = tempfile.mkdtemp(dir=str(TEMP_DIR))

        bin_path = Path(workdir) / "temp.bin"
        bin_path.write_bytes(bin_data)
        
        return bin_path
    except Exception as e:
        if bin_path and bin_path.exists():
            bin_path.unlink()
        raise ValueError(f"写入二进制文件失败: {e}")

def disasm_bin(arch: str, bin_path: str) -> str:
    """
    反汇编 .bin 文件
    """
    try:
        _, objdump_cmd, arch_flag = get_compiler_config(arch)
        cmd = [objdump_cmd, "-D", "-b", "binary", "-m", arch_flag, bin_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        
        if result.returncode != 0:
            return None
        
        return result.stdout
    except Exception as e:
        raise ValueError(f"反汇编二进制文件失败: {e}")

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
        
    except Exception as e:
        raise ValueError(f"提取汇编代码失败: {e}")