#!/bin/bash
# 快速启动脚本

set -e

echo "=========================================="
echo "Qwen3-VL 推理服务启动"
echo "=========================================="

# 停止旧容器
echo "停止旧容器..."
docker compose down 2>/dev/null || true

# 启动新容器
echo "启动服务..."
docker compose up -d

echo ""
echo "=========================================="
echo "服务启动中，请等待模型加载（约 3-5 分钟）"
echo "=========================================="
echo ""
echo "查看日志："
echo "  docker compose logs -f"
echo ""
echo "检查状态："
echo "  docker compose ps"
echo "  curl http://localhost:20000/health"
echo ""
echo "GPU 状态："
echo "  nvidia-smi"
echo "  curl http://localhost:20000/gpu/status"
echo ""
echo "=========================================="
