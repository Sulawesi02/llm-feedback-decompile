from typing import List, Dict

def construct_infer_prompt(asm: str) -> List[Dict[str, str]]:
    """
    构造推理提示
    """
    messages = [
        {"role": "system", "content": """你是一名专业的 C 语言反编译专家，精通 x86-64 汇编。

你的任务是：根据给定的汇编代码，输出一个在语义上完全等价的 C 函数实现。

输出要求：
1. 只输出一个完整的 C 函数定义，函数名使用 func0 。
2. 合理推断并使用基础类型，使用有意义的变量名。
3. 严禁输出任何解释、注释、分析、额外文字。
4. 严禁输出头文件（#include）、宏定义（#define）或 main 函数。
5. 严禁输出任何 Markdown（如 ```c）、多余空行或说明。
"""},
        {"role": "user", "content": f"""请将以下汇编代码反编译为在语义上等价的 C 函数代码。
架构：x86-64
汇编代码:
{asm}
"""}
    ]
    return messages

def construct_fix_prompt(asm: str, previous_func: str, error_message: str) -> List[Dict[str, str]]:
    """
    构造修复提示
    """
    messages = [
        {"role": "system", "content": """你是一名专业的 C 语言反编译专家。

你的任务是：根据汇编语义和错误信息，修复之前的 C 函数。

输出要求：
1. 只输出一个完整的 C 函数定义，函数名使用 func0 。
2. 合理推断并使用基础类型，使用有意义的变量名。
3. 严禁输出任何解释、注释、分析、额外文字。
4. 严禁输出头文件（#include）、宏定义（#define）或 main 函数。
5. 严禁输出任何 Markdown（如 ```c）、多余空行或说明。
"""},
        {"role": "user", "content": f"""上次生成的 C 函数代码有错误，请修复。
架构：x86-64
汇编代码:
{asm}
上次生成的 C 函数代码：
{previous_func}
错误信息：
{error_message}
"""}
    ]
    return messages