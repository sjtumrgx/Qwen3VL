#!/bin/bash
# 使用 ModelScope 下载 Qwen3-VL-32B-Instruct 模型脚本

set -euo pipefail

MODEL_NAME="Qwen/Qwen3-VL-32B-Instruct"
MODEL_DIR="./models/Qwen3-VL-32B-Instruct"

# 可选：强制走 modelscope.ai（你给的链接就是这个站）
export MODELSCOPE_DOMAIN=www.modelscope.ai

# 可选：修改默认缓存目录（默认在 ~/.cache/modelscope/hub）
# export MODELSCOPE_CACHE="/data/modelscope-cache"

# 检查 modelscope CLI 是否安装
if ! command -v modelscope &> /dev/null; then
  echo "错误: modelscope 未安装（找不到 modelscope 命令）"
  echo "请运行: uv pip install -U modelscope"
  exit 1
fi

mkdir -p "$MODEL_DIR"

echo "=========================================="
echo "开始下载 $MODEL_NAME （ModelScope）"
echo "模型较大，请耐心等待..."
echo "MODELSCOPE_DOMAIN=${MODELSCOPE_DOMAIN}"
echo "保存到: $MODEL_DIR"
echo "=========================================="

# 下载到指定目录
# 说明：modelscope download 支持 --local_dir 把仓库内容落到指定路径（官方模型页示例也是这种写法）
modelscope download --model "$MODEL_NAME" --local_dir "$MODEL_DIR"

echo "=========================================="
echo "模型下载完成！"
echo "模型路径: $MODEL_DIR"
echo "=========================================="

# 验证关键文件
if [ -f "$MODEL_DIR/config.json" ]; then
  echo "✓ config.json 存在"
else
  echo "✗ 警告: config.json 不存在"
fi

if ls "$MODEL_DIR"/*.safetensors 1> /dev/null 2>&1; then
  echo "✓ safetensors 模型文件存在"
else
  echo "✗ 警告: safetensors 模型文件不存在"
fi
