#!/bin/bash
# 服务器环境验证脚本

set -e

echo "=========================================="
echo "Qwen3-VL 服务器环境验证"
echo "=========================================="

# 检查 Docker
echo -n "检查 Docker... "
if command -v docker &> /dev/null; then
    DOCKER_VERSION=$(docker --version | awk '{print $3}' | sed 's/,//')
    echo "✓ 已安装 (版本: $DOCKER_VERSION)"
else
    echo "✗ 未安装"
    exit 1
fi

# 检查 Docker Compose
echo -n "检查 Docker Compose... "
if command -v docker compose &> /dev/null; then
    COMPOSE_VERSION=$(docker compose version --short)
    echo "✓ 已安装 (版本: $COMPOSE_VERSION)"
else
    echo "✗ 未安装"
    exit 1
fi

# 检查 NVIDIA Container Toolkit
echo -n "检查 NVIDIA Container Toolkit... "
if docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo "✓ 已安装并正常工作"
else
    echo "✗ 未安装或配置错误"
    exit 1
fi

# 检查 GPU
echo ""
echo "GPU 信息:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | while read line; do
    echo "  GPU $line"
done

# 检查 GPU 数量
GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
echo ""
if [ "$GPU_COUNT" -eq 4 ]; then
    echo "✓ GPU 数量正确: $GPU_COUNT 张"
else
    echo "✗ 警告: 期望 4 张 GPU，实际检测到 $GPU_COUNT 张"
fi

# 检查目录结构
echo ""
echo "检查目录结构:"
for dir in models logs config; do
    if [ -d "./$dir" ]; then
        echo "  ✓ $dir/ 存在"
    else
        echo "  ✗ $dir/ 不存在，正在创建..."
        mkdir -p "./$dir"
    fi
done

# 检查模型文件
echo ""
echo -n "检查模型文件... "
if [ -f "./models/Qwen3-VL-32B-Instruct/config.json" ]; then
    echo "✓ 模型已下载"
else
    echo "✗ 模型未下载"
    echo "  请运行: bash scripts/download_model.sh"
fi

echo ""
echo "=========================================="
echo "环境验证完成！"
echo "=========================================="
