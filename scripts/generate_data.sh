#!/bin/bash
# generate_data.sh

echo "=== ExeBench数据集解压脚本 ==="

BASE_DIR="/app/data/exebench"
cd $BASE_DIR

# 1. 创建目录
echo "1. 创建目录结构..."
mkdir -p {train,valid,test}

# 2. 解压.tar.gz文件
echo "2. 解压.tar.gz文件..."
tar -xzf train_synth_simple_io.tar.gz --strip-components=1 -C train && rm train_synth_simple_io.tar.gz
tar -xzf train_real_simple_io.tar.gz --strip-components=1 -C train && rm train_real_simple_io.tar.gz
tar -xzf valid_synth.tar.gz --strip-components=1 -C valid && rm valid_synth.tar.gz
tar -xzf valid_real.tar.gz --strip-components=1 -C valid && rm valid_real.tar.gz
tar -xzf test_synth.tar.gz --strip-components=1 -C test && rm test_synth.tar.gz
tar -xzf test_real.tar.gz --strip-components=1 -C test && rm test_real.tar.gz

# 3. 解压.zst文件
echo "3. 解压.zst文件..."
echo "3. 检查.zst文件..."
find . -name "*.jsonl.zst" | while read file; do
    echo "  解压 $file"
    zstd -d "$file" -o "${file%.zst}" && rm "$file"
done

echo "=== 完成 ==="