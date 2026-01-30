#!/bin/bash
# 容器启动脚本

set -e

echo "=========================================="
echo "环境信息..."
echo "=========================================="

python3 - <<'PY'
import torch

print("python:", __import__("sys").version.replace("\n", " "))
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("torch.cuda.is_available:", torch.cuda.is_available())
print("torch.cuda.device_count:", torch.cuda.device_count())

try:
    import lmdeploy  # noqa: F401
    print("lmdeploy: import ok")
except Exception as e:
    print("lmdeploy: import failed:", repr(e))
    raise
PY

echo "=========================================="
echo "启动 LMDeploy OpenAI API Server..."
echo "=========================================="

MODEL_PATH="${MODEL_PATH:-/models/Qwen3-VL-32B-Instruct}"
TP="${TENSOR_PARALLEL_SIZE:-4}"
SESSION_LEN="${MAX_MODEL_LEN:-}"
LMDEPLOY_HOST="${LMDEPLOY_HOST:-0.0.0.0}"
LMDEPLOY_PORT="${LMDEPLOY_PORT:-23333}"
LMDEPLOY_BASE_URL="${LMDEPLOY_BASE_URL:-http://127.0.0.1:${LMDEPLOY_PORT}}"
export LMDEPLOY_BASE_URL
export LMDEPLOY_SKIP_WARMUP="${LMDEPLOY_SKIP_WARMUP:-true}"
export LMDEPLOY_ENABLE_CHECK_ENV="${LMDEPLOY_ENABLE_CHECK_ENV:-false}"

lmdeploy_help="$(lmdeploy serve api_server --help 2>&1 || true)"

host_flag=()
if echo "${lmdeploy_help}" | grep -q -- "--host"; then
  host_flag=(--host "${LMDEPLOY_HOST}")
elif echo "${lmdeploy_help}" | grep -q -- "--server-name"; then
  host_flag=(--server-name "${LMDEPLOY_HOST}")
fi

port_flag=()
if echo "${lmdeploy_help}" | grep -q -- "--port"; then
  port_flag=(--port "${LMDEPLOY_PORT}")
elif echo "${lmdeploy_help}" | grep -q -- "--server-port"; then
  port_flag=(--server-port "${LMDEPLOY_PORT}")
fi

backend_flag=()
if echo "${lmdeploy_help}" | grep -q -- "--backend"; then
  backend_flag=(--backend pytorch)
fi

tp_flag=()
if echo "${lmdeploy_help}" | grep -q -- "--tp"; then
  tp_flag=(--tp "${TP}")
elif echo "${lmdeploy_help}" | grep -q -- "--tensor-parallel-size"; then
  tp_flag=(--tensor-parallel-size "${TP}")
fi

trust_flag=()
if echo "${lmdeploy_help}" | grep -q -- "--trust-remote-code"; then
  trust_flag=(--trust-remote-code)
elif echo "${lmdeploy_help}" | grep -q -- "--trust_remote_code"; then
  trust_flag=(--trust_remote_code)
fi

metrics_flag=()
if echo "${lmdeploy_help}" | grep -q -- "--disable-metrics"; then
  metrics_flag=(--disable-metrics)
fi

executor_flag=()
if echo "${lmdeploy_help}" | grep -q -- "--distributed-executor-backend"; then
  executor_flag=(--distributed-executor-backend "${LMDEPLOY_EXECUTOR_BACKEND:-mp}")
fi

session_len_flag=()
if [ -n "${SESSION_LEN}" ] && echo "${lmdeploy_help}" | grep -q -- "--session-len"; then
  session_len_flag=(--session-len "${SESSION_LEN}")
fi

set +e
lmdeploy serve api_server "${MODEL_PATH}" \
  "${backend_flag[@]}" \
  "${tp_flag[@]}" \
  "${host_flag[@]}" \
  "${port_flag[@]}" \
  "${trust_flag[@]}" \
  "${metrics_flag[@]}" \
  "${executor_flag[@]}" \
  "${session_len_flag[@]}" \
  2>&1 | sed -u 's/^/[lmdeploy] /' &
lmdeploy_pid="$!"
set -e

cleanup() {
  echo "正在停止所有服务..."
  for pid in "${webui_pid:-}" "${fastapi_pid:-}" "${lmdeploy_pid:-}"; do
    if [ -n "${pid}" ] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT

echo "等待 LMDeploy 就绪: ${LMDEPLOY_BASE_URL}/v1/models"
for i in $(seq 1 600); do
  if curl -fsS "${LMDEPLOY_BASE_URL}/v1/models" >/dev/null 2>&1; then
    echo "LMDeploy 已就绪"
    break
  fi
  if ! kill -0 "${lmdeploy_pid}" 2>/dev/null; then
    echo "LMDeploy 进程已退出，请检查上方日志"
    exit 1
  fi
  sleep 1
done

echo "=========================================="
echo "启动 FastAPI 服务..."
echo "=========================================="
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --log-level info 2>&1 | sed -u 's/^/[fastapi] /' &
fastapi_pid="$!"

# 等待 FastAPI 就绪
echo "等待 FastAPI 就绪..."
for i in $(seq 1 60); do
  if curl -fsS "http://127.0.0.1:8000/health" >/dev/null 2>&1; then
    echo "FastAPI 已就绪"
    break
  fi
  if ! kill -0 "${fastapi_pid}" 2>/dev/null; then
    echo "FastAPI 进程已退出，请检查上方日志"
    exit 1
  fi
  sleep 1
done

echo "=========================================="
echo "启动 VLN WebUI..."
echo "=========================================="
# WebUI 连接容器内部的 FastAPI (port 8000)
python3 -m app.vln_webui --host 0.0.0.0 --port 7860 --api-url http://127.0.0.1:8000 2>&1 | sed -u 's/^/[webui] /' &
webui_pid="$!"

echo "=========================================="
echo "所有服务已启动"
echo "  - LMDeploy: http://0.0.0.0:${LMDEPLOY_PORT}"
echo "  - FastAPI:  http://0.0.0.0:8000"
echo "  - WebUI:    http://0.0.0.0:7860"
echo "=========================================="

# 等待任意进程退出
wait -n "${lmdeploy_pid}" "${fastapi_pid}" "${webui_pid}" 2>/dev/null || wait "${fastapi_pid}"
