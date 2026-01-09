# LLM Feedback Decompiler 

**基于大语言模型 (LLM) 与编译器反馈机制的交互式反编译系统。**

本系统利用**编译器反馈循环 (Compiler Feedback Loop)** 机制，自动捕获模型生成的 C 代码中的编译错误，并将其反馈给模型进行自我修正，从而显著提高反编译代码的可编译性和准确性。

## 🚀 核心功能
- **交互式反编译**：支持 x86_64 和 ARM (aarch64) 架构的机器码反编译。
- **反馈循环机制**：系统自动验证生成的代码，如果编译失败，会将错误信息反馈给模型进行迭代修正。
- **Web 可视化界面**：提供直观的 Web UI，用户可直接输入 Hex 格式的机器码并查看反编译结果。
- **多种优化级别**：支持 O0, O1, O2, O3 等不同优化级别的处理。

## 🛠️ 快速开始
### 1. 构建环境
推荐使用 Docker 容器以确保编译器环境一致性：
```bash
docker build -t llm-feedback-decompile-env .
```

### 2. 启动环境
```bash
docker run --gpus all -it -v ${PWD}:/app llm-feedback-decompile-env
```

### 3. 启动服务
进入容器后，启动后端服务：
```bash
cd src
# 启动 FastAPI 后端服务
python app.py
```

### 4. 访问界面
服务启动成功后，请在浏览器中访问：
http://localhost:8000

---

## 📊 数据集与训练

本项目模型训练主要基于 **ExeBench** 数据集。

### 数据处理流程
```bash
cd scripts
# 1. 生成原始数据
./generate_data.sh
# 2. 预处理数据
python preprocess.py
# 3. 开始训练
python train.py
```

### ExeBench 数据集统计
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

## 📝 数据格式说明 (JSONL)

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
      // ... 其他架构和优化级别的组合
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
