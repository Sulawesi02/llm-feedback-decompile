FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

ARG DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# 替换 apt 源为清华源
RUN sed -i 's|http://archive.ubuntu.com/ubuntu/|http://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com/ubuntu/|http://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' /etc/apt/sources.list

# 系统工具
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

# pip 清华源
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    && pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn

# 安装 uv
RUN pip install -U pip setuptools wheel uv

# 仅复制 pyproject.toml 和 uv.lock
COPY pyproject.toml uv.lock ./

# 用 uv 安装依赖（直接安装到系统环境，利用基础镜像的 torch）
RUN uv export --frozen --no-emit-project --output-file requirements.txt && \
    uv pip install --system -r requirements.txt

# 安装预编译 flash-attn
RUN uv pip install --system https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

# 清理缓存
RUN pip cache purge

CMD ["bash"]