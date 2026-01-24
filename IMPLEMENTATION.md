# Qwen3-VL 推理服务实现说明（LMDeploy 后端）

本项目采用 **LMDeploy（PyTorch 后端）+ FastAPI 适配层** 的结构运行 Qwen3-VL-32B-Instruct，面向 4×RTX3090（CUDA 12.2 驱动约束）稳定部署。

## 架构概览

- **LMDeploy OpenAI API Server（容器内 :23333）**
  - 负责模型加载与推理
  - 启动方式：`lmdeploy serve api_server <MODEL_PATH> --backend pytorch --tp 4 ...`
  - 默认使用多进程执行器（`--distributed-executor-backend mp`），避免 Ray 相关噪音与兼容性问题
- **FastAPI 适配层（容器内 :8000，宿主映射 :20000）**
  - 保留本仓库现有端点（`/health`、`/infer`、`/analyze/*`、`/gpu/status` 等）
  - 对外提供 OpenAI 兼容端点 `/v1/chat/completions`
  - 内部将请求 **HTTP 转发**到 LMDeploy 的 `/v1/chat/completions`

## 关键代码

### 1) 推理引擎（HTTP 转发）

文件：`app/engine.py`

- `LMDeployEngine` 通过 `requests` 调用：
  - `GET ${LMDEPLOY_BASE_URL}/v1/models`（启动时探测模型 ID）
  - `POST ${LMDEPLOY_BASE_URL}/v1/chat/completions`（推理）
- 自动规范化 `messages` 的多模态结构，确保 `image_url.url` 为 `data:image/...;base64,` 形式

### 2) FastAPI 应用（对外 API）

文件：`app/main.py`

核心端点：

- `GET /health`：健康检查 + GPU 数量
- `GET /v1/models`：对外模型列表（OpenAI 兼容）
- `POST /v1/chat/completions`：对外 OpenAI 兼容接口（内部转发到 LMDeploy）
- `POST /infer`、`/infer/upload`：简化接口（文本/单图）
- `POST /analyze`、`/analyze/upload`：图像分析接口
- `GET /gpu/status`：GPU 详情（便于运维观察）

### 3) 配置（环境变量）

文件：`app/config.py`

常用项：

- 模型：`MODEL_PATH`、`TENSOR_PARALLEL_SIZE`、`MAX_MODEL_LEN`
- 后端：`LMDEPLOY_BASE_URL`、`LMDEPLOY_TIMEOUT_S`
- 采样：`MAX_TOKENS`、`TEMPERATURE`、`TOP_P`

## 容器启动流程

### Docker 镜像

文件：`Dockerfile`

- 基础：`nvidia/cuda:12.2.0-devel-ubuntu22.04`
- 使用 `uv` + venv 安装依赖
- 固定 driver 兼容的 PyTorch CUDA wheel：`torch==2.5.1+cu121`
- 安装：`lmdeploy==0.11.1`、`triton==3.1.0`、`setuptools` 等

### Entrypoint（先起 LMDeploy，再起 FastAPI）

文件：`scripts/entrypoint.sh`

- 启动 `lmdeploy serve api_server`（后台）
  - `--backend pytorch`
  - `--tp ${TENSOR_PARALLEL_SIZE}`
  - `--distributed-executor-backend ${LMDEPLOY_EXECUTOR_BACKEND:-mp}`
  - `--disable-metrics`（降低 Ray/metrics 相关噪音）
  - 默认：`LMDEPLOY_SKIP_WARMUP=true`，并关闭环境检查 warning（`LMDEPLOY_ENABLE_CHECK_ENV=false`）
- 轮询等待 `GET ${LMDEPLOY_BASE_URL}/v1/models` 可用
- 再启动 `uvicorn app.main:app`

## 调优建议

- OOM：降低 `MAX_MODEL_LEN`、减少并发请求、确认没有其它进程占用显存
- 吞吐：优先保持 `LMDEPLOY_EXECUTOR_BACKEND=mp`；如确实需要 Ray 再切换
- 多模态：客户端传图建议使用 `image_url` 的 data URL 形式（`data:image/jpeg;base64,...`）

## 测试

- 文本：`python test_api.py`
- 单图：`python test_api.py --image /path/to/image.jpg`
- 扩展：`python test_api_advanced.py --image /path/to/image.jpg`

