import re
import subprocess
import tempfile
import shutil
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

# 尝试导入配置
try:
    from src.config import TEMP_DIR
except ImportError:
    try:
        from config import TEMP_DIR
    except ImportError:
        TEMP_DIR = Path("/tmp/decompile_workdir")

def construct_initial_prompt(arch: str, opt: str, machine_code: str) -> str:
    """ 构造初始反编译提示 """
    prompt = f"""你是一个专业的二进制反编译专家。
请把下面这段 {arch} {opt} 机器码反编译成可读的 C 语言函数：

{machine_code}

输出只包含 C 代码，不要解释，不要添加额外文字。

C 代码：
"""
    return prompt

def construct_refine_prompt(previous_c_code: str, compile_error: str) -> str:
    """ 构造修正反编译提示 """
    prompt = f"""你是一个专业的二进制反编译专家。你的任务是根据编译错误修复 C 语言代码。请严格只输出修复后的 C 代码，不要输出任何解释、Markdown 标记或其他无关内容。

你之前生成的 C 代码编译失败了。

代码如下：
{previous_c_code}

编译错误信息如下：
{compile_error}

请仔细阅读错误，修复代码中的问题。
要求：
- 只输出修复后的完整 C 代码
- 严禁输出 Markdown 代码块标记
- 严禁输出任何解释"""

    return prompt

def clean_c_code(text: str) -> str:
    """ 清理模型输出，提取 C 代码 """
    # 尝试提取 markdown 代码块
    pattern = r"```c?(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()
    return text.strip()

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

    # 确保临时目录存在
    if not TEMP_DIR.exists():
        TEMP_DIR.mkdir(parents=True, exist_ok=True)

    # 创建临时目录
    try:
        workdir = tempfile.mkdtemp(dir=TEMP_DIR)
    except Exception as e:
        # Fallback to system temp if configured TEMP_DIR fails
        workdir = tempfile.mkdtemp()
        
    c_path = Path(workdir) / "func.c"
    o_path = Path(workdir) / "func.o"

    try:
        c_path.write_text(c_code, encoding="utf-8")

        cmd = [cc] + args + [f"-{opt}", "-c", str(c_path), "-o", str(o_path)]
        # 设置超时时间，防止编译器卡死
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
