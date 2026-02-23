import re
import subprocess
import tempfile
import shutil
from pathlib import Path

DISASM_PATTERN = re.compile(r'^\s*[0-9a-f]+:\s+(?:[0-9a-f]{2}\s+)+(.+)$', re.IGNORECASE | re.MULTILINE)

def compile_to_obj(func: str) -> tuple:
    """
    将 C/C++ 函数代码编译为 .o文件
    """
    temp_dir = tempfile.mkdtemp(prefix="obj_")
    try:
        func_path = Path(temp_dir) / "func.c"
        func_path.write_text(func, encoding="utf-8")
        
        o_path = Path(temp_dir) / "func.o"

        cmd = ["g++", "-c", str(func_path), "-o", str(o_path)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            error_msg = result.stderr.strip() or result.stdout.strip() or "未知编译错误"
            return False, error_msg

        return True, None
    except Exception as e:
        return False, str(e)
    finally:
        if temp_dir and Path(temp_dir).exists():
            shutil.rmtree(temp_dir, ignore_errors=True)

def compile_test_compare(
    asm: str,
    func: str,
    test: str,
) -> tuple:
    """
    将候选 C 函数和汇编函数分别与测试代码链接、运行并比较结果
    """
    temp_dir = tempfile.mkdtemp(prefix="test_")
    try:        
        func_path = Path(temp_dir) / "func.c"
        func_path.write_text(func, encoding="utf-8")

        test_path = Path(temp_dir) / "test.c"
        test_path.write_text(test, encoding="utf-8")

        c_exe_path = Path(temp_dir) / "program_c"
        compile_c_cmd = [
            "gcc",
            "-std=c11",
            "-Wall",
            "-Wextra",
            str(func_path),
            str(test_path),
            "-o",
            str(c_exe_path),
        ]
        compile_c_result = subprocess.run(compile_c_cmd, capture_output=True, text=True, timeout=10)
        if compile_c_result.returncode != 0:
            error_msg = compile_c_result.stderr.strip() or compile_c_result.stdout.strip() or "C 版本未知编译错误"
            return False, f"C版本编译失败: {error_msg}"

        run_c_result = subprocess.run([str(c_exe_path)], capture_output=True, text=True, timeout=5)
        if run_c_result.returncode != 0:
            error_msg = run_c_result.stderr.strip() or run_c_result.stdout.strip() or "C 版本未知运行错误"
            return False, f"C版本运行失败: {error_msg}"

        if not asm:
            return True, None

        asm_lines = [line.strip() for line in asm.splitlines() if line.strip()]
        asm_body = "\n".join(f"    {line}" for line in asm_lines)
        asm_source = f""".text
.globl func0
func0:
{asm_body}
"""
        asm_path = Path(temp_dir) / "func0.s"
        asm_path.write_text(asm_source, encoding="utf-8")

        asm_exe_path = Path(temp_dir) / "program_asm"
        compile_asm_cmd = [
            "gcc",
            "-std=c11",
            "-Wall",
            "-Wextra",
            str(test_path),
            str(asm_path),
            "-o",
            str(asm_exe_path),
        ]
        compile_asm_result = subprocess.run(compile_asm_cmd, capture_output=True, text=True, timeout=10)
        if compile_asm_result.returncode != 0:
            error_msg = compile_asm_result.stderr.strip() or compile_asm_result.stdout.strip() or "ASM 版本未知编译错误"
            return False, f"ASM版本编译失败: {error_msg}"

        run_asm_result = subprocess.run([str(asm_exe_path)], capture_output=True, text=True, timeout=5)
        if run_asm_result.returncode != 0:
            error_msg = run_asm_result.stderr.strip() or run_asm_result.stdout.strip() or "ASM 版本未知运行错误"
            return False, f"ASM版本运行失败: {error_msg}"

        if run_c_result.stdout != run_asm_result.stdout:
            return False, "C版本与ASM版本测试输出不一致"

        return True, None
    except Exception as e:
        return False, str(e)
    finally:
        if temp_dir and Path(temp_dir).exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


def compile_test(
    func_dep: str,
    func: str,
    test: str,
) -> tuple:
    """
    将候选代码和测试代码合并编译并运行
    """
    temp_dir = tempfile.mkdtemp(prefix="test_")
    try:        
        full_code = f"""
{func_dep}

{func}

{test}
"""
        test_path = Path(temp_dir) / "test.c"
        test_path.write_text(full_code, encoding="utf-8")

        exe_path = Path(temp_dir) / "program"
        
        cmd = ["g++", "-o", str(exe_path), str(test_path)]
        compile_result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if compile_result.returncode != 0:
            error_msg = compile_result.stderr.strip() or compile_result.stdout.strip() or "未知编译错误"
            return False, error_msg
        
        run_result = subprocess.run([str(exe_path)], capture_output=True, text=True, timeout=5)
        if run_result.returncode != 0:
            error_msg = run_result.stderr.strip() or run_result.stdout.strip() or "未知运行错误"
            return False, error_msg
        
        return True, None
    except Exception as e:
        return False, str(e)
    finally:
        if temp_dir and Path(temp_dir).exists():
            shutil.rmtree(temp_dir, ignore_errors=True)

def write_machine_code_to_bin(machine_code: str) -> Path:
    """
    将机器码字符串写入二进制文件
    """
    bin_path = None
    try:
        hex_str = machine_code.replace(" ", "")
        bin_data = bytes.fromhex(hex_str)
        
        bin_dir = tempfile.mkdtemp(prefix="bin_")

        bin_path = Path(bin_dir) / "temp.bin"
        bin_path.write_bytes(bin_data)
        
        return bin_path
    except Exception as e:
        if bin_path and bin_path.exists():
            bin_path.unlink()
        raise ValueError(f"写入二进制文件失败: {e}")

def disasm_bin(bin_path: str) -> str:
    """
    反汇编 .bin 文件
    """
    try:
        cmd = ["objdump", "-D", "-b", "binary", "-m", "i386:x86-64", bin_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        
        if result.returncode != 0:
            return None
        
        return result.stdout
    except Exception as e:
        raise ValueError(f"反汇编二进制文件失败: {e}")

def extract_asm(disasm_output: str) -> str:
    """
    从反汇编结果中提取汇编代码
    """
    try:
        # 匹配汇编代码
        asm_lines = []
        
        for match in DISASM_PATTERN.finditer(disasm_output):
            asm = match.group(1).strip()
            if asm:
                asm_lines.append(asm)
        
        return '\n'.join(asm_lines) if asm_lines else None
        
    except Exception as e:
        raise ValueError(f"提取汇编代码失败: {e}")

