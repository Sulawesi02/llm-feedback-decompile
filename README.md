# LLM Feedback Decompiler 

**基于大语言模型 (LLM) 与编译器反馈机制的交互式反编译系统。**

本系统利用**编译器反馈循环 (Compiler Feedback Loop)** 机制，自动捕获模型生成的 C 函数代码中的编译错误，并将其反馈给模型进行自我修正，从而显著提高反编译代码的可编译性和准确性。

## 核心功能
- **交互式反编译**：支持 x86_64 架构的机器码反编译。
- **反馈循环机制**：系统自动验证生成的代码，如果编译测试失败，会将错误信息反馈给模型进行迭代修正。
- **Web 可视化界面**：提供直观的 Web UI，用户可直接输入 Hex 格式的机器码并查看反编译结果。

## 快速开始

### 1. 构建环境
推荐使用 Docker 容器以确保编译器环境一致性：
```bash
docker build -t llm-feedback-decompile:latest .```

### 2. 启动环境

启动容器并将当前目录挂载到容器内的 `/app` 目录：

```bash
docker run --gpus all -it -p 8000:8000 -v ${PWD}:/app llm-feedback-decompile:latest
```

### 3. 启动服务

进入容器后，启动 Web 服务（端口 8000）：

```bash
python src/app.py
```

### 4. 访问界面

服务启动成功后，请在浏览器中访问：
http://localhost:8000

---

## 数据集

本项目模型训练主要基于 **LLM4Decompile** 数据集。

### 1. 下载 LLM4Decompile 数据集

```bash
python scripts/download_data.py
```

- LLM4Decompile 数据格式说明
  - LLM4Decompile 数据集采用 JSONL 格式存储
  - decompile-bench
```json
{
  "name":"demangled name for the function",
  "code":"source code",
  "asm":"assembly",
  "file":"source code path"
}
```
  - decompile-eval
```json
{
  "index":"index of the function", 
  "func_name":"demangled name for he function", 
  "func_dep":"function dependecies (includes, help functions), or the path to the source code", 
  "func":"source code", 
  "test":"unit tests for the function, empty for github data", 
  "opt":"optimization, O0, O1, O2, O3", 
  "language":"language, c or cpp", 
  "asm":"assembly", 
  "ida_asm":"assembly from ida pro", 
  "ida_pseudo":"decompiled results (pseudo code) from ida pro", 
  "ghidra_asm":"assembly from ghidra", 
  "ghidra_pseudo":"decompiled results (pseudo code) from ghidra"
}
```
  - 原始数据保存到 `data/raw_data` 目录。

### 2. 数据处理

```bash
python scripts/process_data.py
```

- decompile-bench 提取 func 和 asm，并按比例划分训练集和验证集；
- decompile-eval 提取 func_dep、func、test 和 asm；
- 通过计算代码的 MinHash 并利用局部敏感哈希（LSH）删除重复样本；
- 去重后的数据保存到 `data/processed_data` 目录。
- 处理后的数据格式
  - decompile-bench
```json
{
  "code":"source code",
  "asm":"assembly",
}
```
  - decompile-eval
```json
{
  "func_dep":"function dependecies (includes, help functions), or the path to the source code", 
  "func":"source code", 
  "test":"unit tests for the function, empty for github data", 
  "asm":"assembly", 
}
```

### 3. 生成 SFT 数据

```bash
python scripts/generate_sft_data.py
```

- SFT 数据集格式
```json
{
  "instruction":"根据目标架构x86和给定的汇编代码（asm），输出一个在语义上完全等价的 C 函数实现",
  "response":"func"
}
```
- 生成的 SFT 数据保存到 `data/sft_data/<train/valid>_data.jsonl`。

### 4. 生成 DPO 数据

```bash
python scripts/generate_dpo_data.py
```

- DPO 数据集格式
```json
{
  "prompt":"根据目标架构x86和给定的汇编代码（asm），输出一个在语义上完全等价的 C 函数实现",
  "chosen":"func",
  "rejected":"对原始正确 C 函数进行**简单规则扰动**（如删除行、交换行、修改运算符/数字）生成的错误代码。"
}
```
- 生成的 DPO 数据保存到 `data/dpo_data/<train/valid>_data.jsonl`。

## 模型

本项目模型主要基于 **Qwen2.5** 模型。

### 1. 下载基座模型

```bash
python scripts/download_model.py
```

- 按配置的 `MODEL_NAME` 下载基座模型，支持断点续传、多次自动重试。
- 基座模型保存到 `model/<模型名>/` 。

### 2. SFT 微调

**核心目标**：通过有监督微调 (Supervised Fine-Tuning)，让模型学会将汇编代码翻译为语义等价的 C 函数（Next Token Prediction）。

```bash
python scripts/train_sft.py
```

- 遍历`VERSIONS`，从 SFT 数据集中按比例采样子集。
- 将每个样本格式化为对话 Prompt。
- 进行 QLoRA 微调，SFT 权重保存到 `adapter/sft/<版本>/`。

### 3. DPO 对齐

**核心目标**：通过直接偏好优化 (Direct Preference Optimization)，抑制模型生成不可编译或低质量代码的倾向。

```bash
python scripts/train_dpo.py
```

- 遍历`VERSIONS`，从 DPO 数据集中按比例采样子集。
- 加载 **基座模型** ，并 **合并 SFT 适配器** 作为新的基座。
- 进行 QLoRA DPO 权重保存到 `adapter/dpo/<版本>/`。

### 4. 模型评估

```bash
python scripts/evaluate.py
```

- 加载 **基座模型** ，挂载 **DPO 适配器**。
- 在测试集上评估反编译测试成功率。
- 结果保存到 `eval/<版本>.jsonl`。





