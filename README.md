# LLM Feedback Decompiler 

**基于大语言模型 (LLM) 与编译器反馈机制的交互式反编译系统。**

本系统利用**编译器反馈循环 (Compiler Feedback Loop)** 机制，自动捕获模型生成的 C 函数代码中的编译错误，并将其反馈给模型进行自我修正，从而显著提高反编译代码的可编译性和准确性。

## 核心功能
- **交互式反编译**：支持 x86_64 和 ARM (aarch64) 架构的机器码反编译。
- **反馈循环机制**：系统自动验证生成的代码，如果编译失败，会将错误信息反馈给模型进行迭代修正。
- **Web 可视化界面**：提供直观的 Web UI，用户可直接输入 Hex 格式的机器码并查看反编译结果。

## 快速开始

### 1. 构建环境
推荐使用 Docker 容器以确保编译器环境一致性：
```bash
docker build -t llm-feedback-decompile-env .
```

### 2. 启动环境

启动容器并将当前目录挂载到容器内的 `/app` 目录：

```bash
docker run --gpus all -it -p 8000:8000 -v ${PWD}:/app llm-feedback-decompile-env
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

本项目模型训练主要基于 **ExeBench** 数据集。

### 1. 下载与准备 ExeBench 数据集

```bash
python scripts/dataset.py
```

- 下载 `train_synth_simple_io`, `valid_synth`, `test_synth` 等关键子集。
- 自动解压并保存到 `data/raw_data/<train/valid/test>` 目录下。

#### ExeBench 数据集统计
| 数据集类别 | 数量 | 描述 |
| :--- | :--- | :--- |
| `train_synth_simple_io` | 0.550M | 训练集合成代码（简单I/O） |
| `train_real_simple_io` | 0.043M | 训练集真实代码（简单I/O） |
| `valid_synth` | 5k | 验证集合成代码 |
| `valid_real` | 2.133k | 验证集真实代码 |
| `test_synth` | 5k | 测试集合成代码 |
| `test_real` | 2.134k | 测试集真实代码 |

#### ExeBench 数据格式说明

ExeBench 数据集采用 JSONL 格式存储，每条记录包含完整的函数信息、汇编代码及编译元数据。

```json
{
  "text": {
    // 基础函数信息
    "path": "源码路径",
    "func_def": "完整函数定义",
    "func_head": "函数声明",
    "fname": "函数名",
    "signature": ["返回类型", "参数类型1", "参数类型2", ...],
    "doc": "文档字符串 (可能为 null)",
    
    // 错误信息
    "angha_error": "",  // angha 系统编译错误信息
    "real_error": null, // 真实编译错误

    // 汇编代码 (包含多种编译器和优化级别)
    "asm": {
      "angha_gcc_x86_O0": {
        "pre_asm": "汇编前导部分（如文件声明、段定义）",
        "func_asm": "函数主体汇编指令",
        "post_asm": "汇编尾部（如符号表、注释）",
        "target": {
          "impl": "gcc",  // 编译器
          "bits": 64,     // 位数
          "lang": "gas",  // 汇编语言格式
          "o": "0"        // 优化级别
        }
      },
      "real_gcc_x86_O0": {...},
      "angha_gcc_x86_Os": {...},
      "real_gcc_x86_Os": {...},
      "angha_gcc_x86_O3": {...},
      "real_gcc_x86_O3": {...},
    },
    
    // 编译依赖
    "angha_deps": "angha系统的依赖代码",
    "real_deps": "真实依赖代码",
    
    // 测试用例 (I/O Pairs)
    "angha_io_pairs": null,
    "real_io_pairs": [
      {
        "input": {"参数1": "值1", "参数2": "值2"},
        "output": {"返回值字段": "返回值", "输出参数": "值"},
        "dummy_funcs": null,
        "dummy_funcs_seed": null
      }
    ],
    
    // 执行包装器
    "angha_exe_wrapper": null,
    "real_exe_wrapper": "完整的C++包装代码",

    // I/O 规范
    "real_iospec": {
      "livein": ["输入参数名列表"],
      "liveout": ["输出参数名列表"],
      "returnvarname": ["返回值变量名"],
      "funname": "函数名",
      "typemap": {
        "参数名": "类型" // e.g. "int32", "string"
      },
      "required_includes": ["需要包含的头文件"]
    },

    // 版本信息
    "ref": "master"
  },
  "meta": {}
}
```

### 2. 数据处理

```bash
python scripts/process_data.py
```

- 从 `data/raw_data/<train/valid/test>` 目录加载原始数据；
- 提取 C 函数代码，编译并提取汇编代码和机器码；
- 通过计算代码的 MinHash 并利用局部敏感哈希（LSH）删除重复样本；
  - train : 10379 -> 9412
  - valid : 1792 -> 1614
  - test : 1872 -> 1650
- 去重样本保存到 `data/dpo_data` 目录。

#### 处理后的样本格式
```json
{
  "c_code": "对应的 C 函数代码",
  "compilations": {
    "x86": {
      "asm": "汇编代码",
      "machine_code": "机器码"
    },
    "arm": {...}
  }
}
```

## 模型

本项目模型主要基于 **Qwen2.5** 模型。

### 1. 下载基座模型

```bash
python scripts/basemodel.py
```

- 按配置的 `MODEL_NAME` 下载基座模型，支持断点续传、多次自动重试。
- 基座模型保存到 `model/base_models/<模型名>/` 。

### 2. 生成 SFT 数据

```bash
python scripts/process_sft_data.py
```

- 构造 SFT 数据集
  - instruction : arch + 汇编代码
  - response : C 代码
- 生成的 SFT 数据保存到 `data/sft_data/<train/valid>_data.jsonl`。

### 3. SFT 微调

**核心目标**：通过有监督微调 (Supervised Fine-Tuning)，让模型学会将汇编代码翻译为语义等价的 C 代码（Next Token Prediction）。

```bash
python scripts/train_sft.py
```

- 遍历`VERSIONS`，从 SFT 数据集中按比例采样子集。
- 将每个样本格式化为对话 Prompt。
- 进行 QLoRA 微调，SFT 权重保存到 `model/sft_adapter/<版本>/`。

### 4. 生成 DPO 数据

```bash
python scripts/process_dpo_data.py
```

- 构造 DPO 数据集
  - prompt : arch + 汇编代码
  - chosen (正例) : 原始数据集中正确的 C 代码。
  - rejected (负例) : 基座模型基于正确 C 代码生成的 **错误/低质量** 代码。
- 生成的 DPO 数据保存到 `data/dpo_data/<train/valid>_data.jsonl`。

### 5. DPO 对齐

**核心目标**：通过直接偏好优化 (Direct Preference Optimization)，抑制模型生成不可编译或低质量代码的倾向。

```bash
python scripts/train_dpo.py
```

- 遍历`VERSIONS`，从 DPO 数据集中按比例采样子集。
- 加载 **基座模型** ，并 **合并 SFT 适配器** 作为新的基座。
- 进行 QLoRA DPO 权重保存到 `model/dpo_adapter/<版本>/`。

### 6. 模型评估

```bash
python scripts/evaluate.py
```

- 加载 **基座模型**，挂载 **DPO 适配器**。
- 在测试集上评估反编译成功率和语义等性。
- 结果保存到 `eval/<版本>.jsonl`。





