# Qwen3-VL-32B 推理服务

基于 LMDeploy 的 Qwen3-VL-32B-Instruct 视觉语言模型推理服务，支持四卡 3090 并行推理，提供 OpenAI 兼容的 REST API 接口。

## 特性

- ✅ **多 GPU 并行推理**：使用 LMDeploy Tensor Parallel（`--tp 4`），充分利用 4 张 RTX 3090
- ✅ **OpenAI 兼容 API**：支持 `/v1/chat/completions` 等标准端点
- ✅ **视觉语言理解**：支持图像输入（Base64 编码）和文本指令
- ✅ **容器化部署**：Docker Compose 一键启动，开箱即用
- ✅ **健康检查**：自动监控服务状态，异常自动重启
- ✅ **日志持久化**：日志保存到宿主机，便于排查问题

## 系统要求

### 硬件要求

- **GPU**：4x NVIDIA RTX 3090 (24GB VRAM each)
- **内存**：建议 64GB+
- **存储**：至少 100GB 可用空间（模型约 60GB）

### 软件要求

- **操作系统**：Linux (CentOS 7 / Ubuntu 20.04+)
- **Docker**：>= 20.10
- **Docker Compose**：>= 2.0
- **NVIDIA Driver**：>= 470
- **NVIDIA Container Toolkit**：>= 1.13

## 快速开始

### 1. 环境验证

```bash
# 克隆或进入项目目录
cd /home/gexu/Qwen3VL

# 验证环境
bash scripts/verify_env.sh
```

### 2. 下载模型

```bash
# 安装 HuggingFace CLI（如未安装）
pip install huggingface_hub

# 下载模型（约 60GB，需要较长时间）
bash scripts/download_model.sh

# 如果下载速度慢，可以使用国内镜像：
# export HF_ENDPOINT=https://hf-mirror.com
# bash scripts/download_model.sh
```

### 3. 启动服务

```bash
# 构建 Docker 镜像（首次会下载/安装 torch、lmdeploy 等依赖）
docker compose build

# 启动服务（后台运行）
docker compose up -d

# 查看启动日志
docker compose logs -f
```

模型加载通常需要数分钟，请耐心等待直到看到 "Application startup complete" 日志。

### 4. 测试服务

```bash
# 安装测试依赖
pip install requests

# 运行测试脚本
python test_api.py

# 测试图像推理（需要提供图像文件）
python test_api.py --image /path/to/image.jpg
```

## API 使用示例

### 健康检查

```bash
curl http://localhost:20000/health
```

### 获取模型列表

```bash
curl http://localhost:20000/v1/models
```

### 文本推理

```bash
curl -X POST http://localhost:20000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-VL-32B-Instruct",
    "messages": [
      {"role": "user", "content": "你好，请介绍一下你自己。"}
    ],
    "max_tokens": 512,
    "temperature": 0.7
  }'
```

### 图像+文本推理

```python
import base64
import requests

# 读取图像并编码
with open("image.jpg", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode("utf-8")

# 发送请求
response = requests.post(
    "http://localhost:20000/v1/chat/completions",
    json={
        "model": "Qwen3-VL-32B-Instruct",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": "请描述这张图片的内容。"
                    }
                ]
            }
        ],
        "max_tokens": 1024
    }
)

print(response.json()["choices"][0]["message"]["content"])
```

### 使用 OpenAI SDK

```python
from openai import OpenAI

# 配置客户端
client = OpenAI(
    base_url="http://localhost:20000/v1",
    api_key="dummy"  # 本服务默认不校验 API key（OpenAI SDK 必填字段）
)

# 文本推理
response = client.chat.completions.create(
    model="Qwen3-VL-32B-Instruct",
    messages=[
        {"role": "user", "content": "你好！"}
    ]
)

print(response.choices[0].message.content)
```

## 服务管理

### 查看服务状态

```bash
docker compose ps
```

### 查看日志

```bash
# 实时日志
docker compose logs -f

# 最近 100 行日志
docker compose logs --tail 100
```

### 重启服务

```bash
docker compose restart
```

### 停止服务

```bash
docker compose down
```

### 查看 GPU 使用情况

```bash
nvidia-smi
```

## 性能调优

### 调整上下文长度

如果遇到显存不足（OOM），可以降低最大上下文长度：

```yaml
# docker-compose.yml
environment:
  - MAX_MODEL_LEN=4096  # 从 8192 降低到 4096（用于 LMDeploy 的 session 长度）
```

### 调整并行与执行器后端

```yaml
# docker-compose.yml
environment:
  - TENSOR_PARALLEL_SIZE=4
  - LMDEPLOY_EXECUTOR_BACKEND=mp  # 默认使用多进程（不依赖 Ray）
```

## 故障排查

### 问题 1：容器启动失败

**症状：** `docker compose up` 失败

**排查步骤：**

1. 检查 GPU 驱动：`nvidia-smi`
2. 检查 Docker GPU 支持：`docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi`
3. 查看详细日志：`docker compose logs`

### 问题 2：模型加载失败

**症状：** 日志显示 "Model not found" 或类似错误

**排查步骤：**

1. 检查模型文件是否存在：`ls -la models/Qwen3-VL-32B-Instruct/`
2. 确认 `config.json` 和 `.safetensors` 文件完整
3. 重新下载模型：`bash scripts/download_model.sh`

### 问题 3：显存不足（OOM）

**症状：** 日志显示 "CUDA out of memory"

**解决方案：**

1. 降低 `--max-model-len`（见性能调优）
2. 减少 `--max-num-seqs`
3. 确认没有其他程序占用 GPU

### 问题 4：推理速度慢

**排查步骤：**

1. 检查 GPU 利用率：`nvidia-smi`
2. 确认使用了 4 卡并行：LMDeploy 启动参数应包含 `--tp 4`
3. 检查网络延迟（如果远程访问）

### 问题 5：健康检查失败

**症状：** 容器不断重启

**排查步骤：**

1. 增加 `start_period`：模型加载可能需要更长时间
2. 手动测试健康端点：`curl http://localhost:20000/health`
3. 查看容器日志：`docker compose logs`

## 目录结构

```
.
├── docker-compose.yml          # Docker Compose 配置
├── .env.example                # 环境变量示例
├── README.md                   # 本文档
├── test_api.py                 # API 测试脚本
├── scripts/
│   ├── download_model.sh       # 模型下载脚本
│   └── verify_env.sh           # 环境验证脚本
├── models/                     # 模型文件目录
│   └── Qwen3-VL-32B-Instruct/
│       ├── config.json
│       ├── *.safetensors
│       └── ...
└── logs/                       # 日志文件目录
    └── (可选：自定义日志输出)
```

## 配置说明

### 端口映射

- **20000**：主服务端口（映射到容器内 8000）
- **20001-20010**：备用端口（用于未来扩展）

### Volume 挂载

- `./models:/models:ro`：模型文件（只读）
- `./logs:/logs`：日志文件（读写）

### 环境变量

参考 `.env.example` 文件进行配置。

## 参考资料

- [LMDeploy 文档](https://lmdeploy.readthedocs.io/)
- [Qwen3-VL 模型卡片](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct)
- [OpenAI API 规范](https://platform.openai.com/docs/api-reference)

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！
