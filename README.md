# Qwen3-VL 推理服务 & VLN 视觉语言导航系统

基于 LMDeploy 的 Qwen3-VL-32B-Instruct 视觉语言模型推理服务，支持四卡 3090 并行推理，提供 OpenAI 兼容的 REST API 接口，并集成了面向四足机器人的视觉语言导航（VLN）系统。

## 特性

### 基础推理服务
- ✅ **多 GPU 并行推理**：使用 LMDeploy Tensor Parallel（`--tp 4`），充分利用 4 张 RTX 3090
- ✅ **OpenAI 兼容 API**：支持 `/v1/chat/completions` 等标准端点
- ✅ **视觉语言理解**：支持图像/视频输入和文本指令
- ✅ **流式输出**：支持 SSE 流式响应
- ✅ **容器化部署**：Docker Compose 一键启动

### VLN 视觉语言导航
- ✅ **实时导航**：30-50Hz 非阻塞响应，满足机器人实时控制需求
- ✅ **任务管理**：支持多任务创建、切换、历史记录
- ✅ **关键帧采样**：智能帧差检测，减少 VLM 推理负载
- ✅ **路径可视化**：环境描述、动作标注、路径曲线渲染
- ✅ **WebUI 界面**：Gradio 可视化界面，支持视频输入和实时显示
- ✅ **WebSocket 支持**：实时双向通信

## 系统要求

### 硬件要求
- **GPU**：4x NVIDIA RTX 3090 (24GB VRAM each)
- **内存**：建议 64GB+
- **存储**：至少 100GB 可用空间（模型约 60GB）

### 软件要求
- **操作系统**：Linux (Ubuntu 20.04+)
- **Docker**：>= 20.10
- **Docker Compose**：>= 2.0
- **NVIDIA Driver**：>= 470
- **NVIDIA Container Toolkit**：>= 1.13

## 目录结构

```
Qwen3VL/
├── app/                           # 应用代码
│   ├── __init__.py
│   ├── main.py                    # FastAPI 主应用入口
│   ├── config.py                  # 配置管理（环境变量）
│   ├── models.py                  # Pydantic 数据模型
│   ├── engine.py                  # LMDeploy 推理引擎封装
│   ├── video.py                   # 视频处理工具（抽帧）
│   ├── vln_api.py                 # VLN REST API 路由
│   ├── vln_webui.py               # VLN Gradio WebUI
│   └── vln/                       # VLN 核心模块
│       ├── __init__.py            # 数据模型（Waypoint, VLNTask 等）
│       ├── task_manager.py        # 任务生命周期管理
│       ├── frame_sampler.py       # 关键帧采样器
│       ├── inference_pipeline.py  # 异步推理管道
│       ├── waypoint_parser.py     # VLM 输出解析器
│       ├── visualizer.py          # OpenCV 可视化渲染
│       └── prompts.py             # VLN 提示词模板
│
├── scripts/                       # 脚本
│   ├── entrypoint.sh              # 容器启动脚本（LMDeploy + FastAPI + WebUI）
│   ├── download_model.sh          # 模型下载脚本
│   ├── verify_env.sh              # 环境验证脚本
│   └── start.sh                   # 本地启动脚本
│
├── tests/                         # 测试代码
│   └── vln/
│       ├── test_vln_api.py        # VLN API 单元测试
│       └── test_vln_video.py      # VLN 视频测试（支持任务切换）
│
├── testapi/                       # API 测试脚本
│   ├── test_api.py                # 基础 API 测试
│   ├── test_api_advanced.py       # 高级 API 测试
│   └── test_media.py              # 多媒体测试
│
├── models/                        # 模型文件目录（需下载）
│   └── Qwen3-VL-32B-Instruct/
│
├── cache/                         # 缓存目录
├── video/                         # 视频文件目录
│
├── Dockerfile                     # Docker 镜像构建
├── docker-compose.yml             # Docker Compose 配置
├── pyproject.toml                 # Python 项目配置
├── requirements.txt               # 依赖清单
├── environment.yml                # Conda 环境配置
├── .env.example                   # 环境变量示例
├── .gitignore
├── .dockerignore
├── LICENSE
└── README.md                      # 本文档
```

## 快速开始

### 1. 下载模型

```bash
# 安装 ModelScope CLI
pip install -U modelscope

# 下载模型（约 60GB）
bash scripts/download_model.sh
```

### 2. 启动服务

```bash
# 构建并启动（首次需要较长时间）
docker compose build
docker compose up -d

# 查看日志
docker compose logs -f
```

容器启动后会自动运行三个服务：
- **LMDeploy** (端口 23333)：VLM 推理后端
- **FastAPI** (端口 8000 → 映射到 20000)：REST API
- **WebUI** (端口 7860)：Gradio 可视化界面

### 3. 访问服务

```bash
# SSH 隧道（推荐）
ssh -p 2005 -L 20000:localhost:20000 -L 7860:localhost:7860 user@server

# 健康检查
curl http://localhost:20000/health

# WebUI
open http://localhost:7860
```

## API 文档

### 基础推理 API

#### 健康检查
```bash
GET /health
```

#### 文本推理
```bash
POST /v1/chat/completions
Content-Type: application/json

{
  "model": "Qwen3-VL-32B-Instruct",
  "messages": [{"role": "user", "content": "你好"}],
  "max_tokens": 512
}
```

#### 图像推理
```python
import base64
import requests

with open("image.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

response = requests.post("http://localhost:20000/v1/chat/completions", json={
    "model": "Qwen3-VL-32B-Instruct",
    "messages": [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            {"type": "text", "text": "描述这张图片"}
        ]
    }]
})
```

#### 视频分析
```bash
POST /analyze/video
Content-Type: application/json

{
  "video": "<base64_encoded_video>",
  "instruction": "描述视频内容",
  "max_frames": 8
}
```

### VLN 导航 API

#### 创建导航任务
```bash
POST /vln/task/create
Content-Type: application/json

{
  "instruction": "向前走，找到门",
  "config": {
    "output_fps": 5.0,
    "keyframe_threshold": 0.2,
    "enable_visualization": true
  }
}

# 响应
{
  "task_id": "vln_1234567890_abc123",
  "instruction": "向前走，找到门",
  "status": "active"
}
```

#### 处理视频帧（非阻塞，推荐）
```bash
POST /vln/frame
Content-Type: application/json

{
  "task_id": "vln_1234567890_abc123",
  "frame": "<base64_encoded_jpeg>",
  "timestamp": 1234567890.123
}

# 响应（21-40ms）
{
  "task_id": "vln_1234567890_abc123",
  "environment": "室内走廊，前方有一扇木门",
  "action": "继续向前走",
  "waypoints": [
    {"dx": 0.5, "dy": 0.0, "dtheta": 0.0, "confidence": 0.9}
  ],
  "linear_vel": 0.3,
  "angular_vel": 0.0,
  "visualized_frame": "<base64_encoded_jpeg>",
  "inference_time": 0.025
}
```

#### 处理视频帧（同步阻塞）
```bash
POST /vln/frame/sync
# 参数同上，等待实际推理完成（5-8秒）
```

#### 获取任务状态
```bash
GET /vln/task/{task_id}
```

#### 停止任务
```bash
POST /vln/task/{task_id}/stop
```

#### WebSocket 实时流
```javascript
const ws = new WebSocket("ws://localhost:20000/vln/stream/vln_1234567890_abc123");
ws.onmessage = (event) => {
  const result = JSON.parse(event.data);
  console.log(result.environment, result.action);
};
// 发送帧
ws.send(JSON.stringify({frame: base64Frame, timestamp: Date.now() / 1000}));
```

## VLN 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        VLN WebUI (Gradio)                       │
│                      http://localhost:7860                      │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FastAPI REST API                            │
│                   http://localhost:20000                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ /vln/task/* │  │ /vln/frame  │  │ /vln/stream/{task_id}   │  │
│  │ 任务管理     │  │ 帧处理      │  │ WebSocket 实时流         │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                      VLN Core Modules                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │ TaskManager  │  │ FrameSampler │  │ InferencePipeline    │   │
│  │ 任务生命周期  │  │ 关键帧检测    │  │ 异步推理队列          │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │ WaypointParser│ │ Visualizer   │  │ Prompts              │   │
│  │ 输出解析      │  │ 可视化渲染    │  │ 提示词模板            │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LMDeploy Inference Engine                    │
│                   http://localhost:23333                        │
│                    Qwen3-VL-32B-Instruct                        │
│                      4x RTX 3090 (TP=4)                         │
└─────────────────────────────────────────────────────────────────┘
```

### 核心模块说明

| 模块 | 文件 | 功能 |
|------|------|------|
| **TaskManager** | `app/vln/task_manager.py` | 任务创建、状态管理、历史记录、关键帧存储 |
| **FrameSampler** | `app/vln/frame_sampler.py` | 帧差检测（直方图+像素+边缘），智能采样 |
| **InferencePipeline** | `app/vln/inference_pipeline.py` | 异步队列、后台推理、结果缓存 |
| **WaypointParser** | `app/vln/waypoint_parser.py` | JSON/文本解析、离散动作映射、航点验证 |
| **Visualizer** | `app/vln/visualizer.py` | 环境/动作框、路径曲线、速度指示器 |
| **Prompts** | `app/vln/prompts.py` | 系统提示词、历史上下文构建 |

### 输出格式

VLN 系统输出航点序列，每个航点包含：
- `dx`: 前进距离 (0-1m)
- `dy`: 横向距离 (正值向左)
- `dtheta`: 旋转角度 (-0.5 到 0.5 rad)
- `confidence`: 置信度 (0-1)

可通过以下公式转换为速度命令：
```python
linear_vel = sum(wp.dx for wp in waypoints) / len(waypoints) * output_fps
angular_vel = sum(wp.dtheta for wp in waypoints) / len(waypoints) * output_fps
```

## VLN 测试

### 视频测试（支持任务切换）

```bash
# 基础测试
python tests/vln/test_vln_video.py \
  --url http://localhost:20000 \
  --video "https://example.com/video.mp4" \
  --instruction "向前走，找到门"

# 中途切换任务
python tests/vln/test_vln_video.py \
  --url http://localhost:20000 \
  --video video.mp4 \
  --instruction "向前走" \
  --switch 100 "转向左边" \
  --switch 200 "停下来"

# 预览模式（按 's' 手动切换任务）
python tests/vln/test_vln_video.py \
  --video video.mp4 \
  --instruction "向前走" \
  --preview
```

### WebUI 使用

1. 访问 `http://localhost:7860`
2. 点击「检查服务」确认连接
3. 输入视频 URL 或本地路径，点击「加载视频」
4. 输入导航指令，点击「发布任务」
5. 点击「开始」查看实时导航画面

## 性能指标

| 指标 | 数值 |
|------|------|
| 模型加载时间 | 3-5 分钟 |
| VLM 推理延迟 | 5-8 秒 |
| 非阻塞响应延迟 | 21-40 ms |
| 有效输出帧率 | 30-50 Hz |
| GPU 显存使用 | ~23GB/卡 |

## 配置说明

### 端口映射

| 端口 | 服务 | 说明 |
|------|------|------|
| 20000 | FastAPI | REST API（映射自容器 8000） |
| 7860 | WebUI | Gradio 可视化界面 |
| 23333 | LMDeploy | VLM 推理后端（容器内部） |

### 环境变量

```bash
# 模型配置
MODEL_PATH=/models/Qwen3-VL-32B-Instruct
TENSOR_PARALLEL_SIZE=4
MAX_MODEL_LEN=8192

# LMDeploy 配置
LMDEPLOY_PORT=23333
LMDEPLOY_BASE_URL=http://127.0.0.1:23333
LMDEPLOY_EXECUTOR_BACKEND=mp
```

## 故障排查

### 容器启动失败
```bash
# 检查 GPU
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# 查看日志
docker compose logs
```

### VLN 推理无响应
```bash
# 检查服务状态
curl http://localhost:20000/health
curl http://localhost:20000/vln/task/{task_id}

# 检查 LMDeploy
curl http://localhost:23333/v1/models
```

### WebUI 无法访问
```bash
# 确认端口映射
docker compose ps
# 确认 SSH 隧道
ssh -L 7860:localhost:7860 ...
```

## 参考资料

- [LMDeploy 文档](https://lmdeploy.readthedocs.io/)
- [Qwen3-VL 模型](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct)
- [Uni-NaVid](https://github.com/OpenRobotLab/Uni-NaVid) - VLN 参考实现
- [StreamVLN](https://github.com/OpenRobotLab/StreamVLN) - 流式 VLN 参考

## 许可证

MIT License
