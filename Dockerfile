FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# 系统工具 + 交叉编译 + ninja + capstone
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget ca-certificates unzip git curl bzip2 build-essential vim \
    openjdk-17-jdk-headless tzdata libxext6 libxrender1 libxtst6 libxi6 \
    gcc-aarch64-linux-gnu g++-aarch64-linux-gnu binutils-aarch64-linux-gnu \
    libc6-dev-arm64-cross \
    libcapstone-dev capstone-tool \
    && rm -rf /var/lib/apt/lists/*

# 清华 pip 镜像
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    && pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn

# 升级 pip
RUN pip install --upgrade pip setuptools wheel

# 复制并安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 安装预编译 flash-attn
RUN pip install flash-attn==2.6.3 --no-build-isolation || \
    pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.0.0/flash_attn-2.6.3+cu124torch2.4-cp311-cp311-linux_x86_64.whl

# 清理缓存
RUN pip cache purge

CMD ["bash"]