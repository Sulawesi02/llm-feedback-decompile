from typing import List, Dict, Any, Optional

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


def build_generate_text(iter: int, tokenizer, sample: Dict[str, Any], max_prompt_tokens: int) -> Optional[str]:
    if iter == 0:
        messages = construct_infer_prompt(sample["asm"])
    else:
        messages = construct_fix_prompt(sample["asm"], sample["prev_outputs"], sample["last_error"])
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    token_count = len(tokenizer.encode(text))
    if token_count > max_prompt_tokens:
        return None
    return text


def construct_test_prompt(func: str) -> List[Dict[str, str]]:
    """
    构造测试提示
    """
    messages = [
        {
            "role": "system",
            "content": """你是一名专业的 C 语言单元测试编写专家。

你的任务是：根据给定的函数实现，生成用于验证函数行为的 C 测试代码。

输出要求：
1. 输出一个完整的 C 测试文件内容，包括：
   - 所有需要的头文件（例如 stdio.h、assert.h 等）的 #include；
   - func0 的函数声明（只声明，不定义），在后续链接时会使用真实实现；
   - 一个完整的 main 函数，在 main 中多次调用 func0，覆盖正常情况、边界情况和异常输入；
2. 在 main 中使用 assert 语句验证 func0 的返回值和副作用，当所有检查通过时 main 返回 0，任意检查失败时返回非 0；
3. 严禁输出 func0 的定义，只能在测试代码中声明并调用 func0；
4. 严禁输出任何解释、注释、分析、额外文字；
5. 严禁输出任何 Markdown（如 ```c）、多余空行或说明。
""",
        },
        {
            "role": "user",
            "content": f"""请根据以下 C 函数实现，生成用于验证其行为的 C 测试代码。
函数实现：
{func}
""",
        },
    ]
    return messages


def build_test_text(tokenizer, func: str, max_prompt_tokens: int) -> Optional[str]:
    messages = construct_test_prompt(func)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    token_count = len(tokenizer.encode(text))
    if token_count > max_prompt_tokens:
        return None
    return text
