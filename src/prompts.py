from typing import List, Dict

def construct_train_prompt(arch: str, asm: str, c_code: str = None) -> List[Dict[str, str]]:
    """
    构造训练提示
    """
    messages = [
        {"role": "system", "content": """你是一名专业的 C 语言反编译专家，精通 x86-64 和 AArch64 汇编。

你的任务是：根据给定的目标架构和汇编代码，输出一个在语义上完全等价的 C 函数实现。

输出要求：
1. 只输出一个完整的 C 函数定义。
2. 只输出 C 语言代码，严禁输出任何解释、注释、分析、额外文字。
3. 严禁输出头文件、宏定义或 main 函数。
4. 输出代码必须合法、可编译。
5. 严禁输出任何 Markdown（如 ```c）、多余空行或说明。
"""},
        {"role": "user", "content": f"""请将以下汇编代码反编译为在语义上等价的 C 函数代码。
架构：{arch}
汇编代码:
{asm}
"""},
        {"role": "assistant", "content": c_code}
    ]
    return messages

def construct_bad_code_prompt(c_code: str) -> List[Dict[str, str]]:
    """
    构造生成坏代码的提示
    """
    messages = [
        {"role": "system", "content": """你是一名代码生成助手。
你的任务是：基于给定的正确 C 代码，改写生成一份**有细微缺陷或语义不等价**的 C 代码。
要求：
1. 看起来像正确的代码，但存在逻辑错误、边界条件错误或计算错误。
2. 不要改变函数签名。
3. 只输出 C 代码，不要解释。
"""},
        {"role": "user", "content": f"""请根据以下正确的 C 代码，生成一份有缺陷的代码：
{c_code}
"""}
    ]
    return messages

def construct_equivalence_prompt(c_code_1: str, c_code_2: str) -> List[Dict[str, str]]:
    """
    构造语义等价性判断提示
    """
    messages = [
        {"role": "system", "content": """你是一名资深的 C 语言代码审计专家。

你的任务是：判断给定的两个 C 函数在语义上是否完全等价。

判断标准：
1. 功能一致性：两者对相同的输入是否产生相同的输出和副作用。
2. 结构容忍度：忽略变量命名、代码格式、控制流结构（如 while vs for）的差异，只要逻辑等价即可。
3. 类型兼容性：忽略具体的类型名称差异（如 typedef），关注底层数据操作的一致性。

输出要求：
1. 只输出一个数字："0" 或 "1"。
2. 如果等价，输出 "0"。
3. 如果不等价，输出 "1"。
4. 严禁输出任何解释、分析或其他文字。
"""},
        {"role": "user", "content": f"""请判断以下两个 C 函数是否在语义上等价。

函数 1：
{c_code_1}

函数 2：
{c_code_2}
"""}
    ]
    return messages

def construct_infer_prompt(arch: str, asm: str) -> List[Dict[str, str]]:
    """
    构造推理提示
    """
    messages = [
        {"role": "system", "content": """你是一名专业的 C 语言反编译专家，精通 x86-64 和 AArch64 汇编。

你的任务是：根据给定的目标架构和汇编代码，输出一个在语义上完全等价的 C 函数实现。

核心原则：
- 以语义等价为最高目标，不要求逐条汇编指令对应。
- 合理推断数据类型：int / long / 指针 / 数组等。
- 优先恢复 for / while / if 等结构化控制流。
- 合理命名函数与变量，名称应体现语义。
- 忽略无关指令如 nop、栈帧建立，只关注核心数据操作和计算。

输出要求：
1. 只输出一个完整的 C 函数定义。
2. 只输出 C 语言代码，严禁输出任何解释、注释、分析、额外文字。
3. 严禁输出头文件、宏定义或 main 函数。
4. 输出代码必须合法、可编译。
5. 严禁输出任何 Markdown（如 ```c）、多余空行或说明。
"""},
        {"role": "user", "content": f"""请将以下汇编代码反编译为在语义上等价的 C 函数代码。
架构：{arch}
汇编代码:
{asm}
"""}
    ]
    return messages

def construct_fix_prompt(arch: str, asm: str, previous_c_code: str = None, error_message: str = None) -> List[Dict[str, str]]:
    """
    构造修复提示
    """
    messages = [
        {"role": "system", "content": """你是一名专业的 C 语言反编译专家。

你的任务是：根据汇编语义和错误信息，修复之前的 C 代码。

核心原则：
- 以语义等价为最高目标，不要求逐条汇编指令对应。
- 合理推断数据类型：int / long / 指针 / 数组等。
- 优先恢复 for / while / if 等结构化控制流。
- 合理命名函数与变量，名称应体现语义。
- 忽略无关指令如 nop、栈帧建立，只关注核心数据操作和计算。

输出要求：
1. 只输出一个完整的 C 函数定义。
2. 只输出 C 语言代码，严禁输出任何解释、注释、分析、额外文字。
3. 严禁输出头文件、宏定义或 main 函数。
4. 输出代码必须合法、可编译。
5. 严禁输出任何 Markdown（如 ```c）、多余空行或说明。
"""},
        {"role": "user", "content": f"""上次生成的 C 函数代码有错误，请修复。
架构：{arch}
汇编代码:
{asm}
上次生成的 C 函数代码：
{previous_c_code}
错误信息：
{error_message}
"""}
    ]
    return messages
