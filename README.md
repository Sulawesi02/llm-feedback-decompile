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
```bash
docker run --gpus all -it -p 8000:8000 -v ${PWD}:/app llm-feedback-decompile-env
```

### 3. 启动服务
进入容器后，启动 Web 服务（端口 8000）：

```bash
cd src
python app.py
```
### 4. 访问界面
服务启动成功后，请在浏览器中访问：
http://localhost:8000

---

## 数据集

本项目模型训练主要基于 **ExeBench** 数据集。

### 1. 下载 ExeBench 数据集
https://huggingface.co/datasets/jordiae/exebench/tree/main

### 2. 解压 ExeBench 数据集
```bash
cd scripts
# 解压 ExeBench 数据集
./extract_data.sh
```

- 在 `data/exebench` 目录下创建 `train` 、 `valid` 、 `test` 子目录。
- 在每个子目录下解压下载的 JSONL 文件。

#### ExeBench 数据集统计
| 数据集类别 | 数量 | 描述 |
| :--- | :--- | :--- |
| `train_not_compilable` | 2.357M | 训练集不可编译代码 |
| `train_synth_compilable` | 2.308M | 训练集合成代码（可编译） |
| `train_real_compilable` | 0.675M | 训练集真实代码（可编译） |
| `train_synth_simple_io` | 0.550M | 训练集合成代码（简单I/O） |
| `train_real_simple_io` | 0.043M | 训练集真实代码（简单I/O） |
| `train_synth_rich_io` | 0.097M | 训练集合成代码（丰富I/O） |
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

### 3. 数据预处理
```bash
# 数据预处理
python preprocess.py
```

- 从 `data/exebench` 目录加载 JSONL 数据；
- 提取 C 函数代码，编译并生成汇编代码和机器码；
- 通过计算代码的 MinHash 并利用局部敏感哈希（LSH）删除重复样本；
  - train : 10379 -> 9412
  - valid : 1792 -> 1614
  - test : 1872 -> 1650
- 保存处理后的样本到 `data/processed` 目录。

#### 处理后的样本格式
```json
{
  "c_code": "对应的 C 函数代码",
  "compilations": {
    "x86": {
      "asm": "汇编代码",
      "machine_code": "机器码"
    },
    "arm": {
      "asm": "汇编代码",
      "machine_code": "机器码"
    }
  }
}
```

## 模型

本项目模型主要基于 **Qwen2.5** 模型。

### 1. 下载基座模型

```bash
# 下载基座模型
python download_model.py
```

- 按配置的 `MODEL_NAMES` 列表（如 1.5B/3B/7B 版本）依次下载模型到 [BASE_MODEL_DIR_PATH] 对应目录下，支持断点续传、多次自动重试。

### 2. 模型训练

```bash
# 模型训练（LoRA 微调）
python train.py
```

- 遍历内部配置的 `MODEL_NAMES` 和 `VERSIONS`，按 `(模型名, 版本, 数据比例)` 组合，从去重后的 train/valid 数据集中按比例采样子集进行 LoRA 微调；
- LoRA 权重保存到 `model/lora_checkpoints/<模型名>/<版本>/`；
- 将 LoRA 与基座模型合并后的完整模型保存到 `model/merged_model/<模型名>/<版本>/`。

### 3. 模型评估

```bash
cd scripts
python evaluate.py
```

- 按配置的 `MODEL_PATH` 评估指定目录下的单个合并模型版本；
- 使用 `data/processed/test_asm_to_c_dedup.jsonl` 作为测试集；
- 在评估过程中，对每个样本执行
  - 反编译 -> 编译 -> 使用错误信息做反馈修复；
  - 编译成功后调用 LLM 生成 IO 测试用例；
  - 使用 assert 进行验证。


