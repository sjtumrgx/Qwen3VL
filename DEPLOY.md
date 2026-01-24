# Qwen3-VL 部署指南

## 部署状态

✅ **本地配置文件已创建**
✅ **文件已同步到服务器**
✅ **服务器环境验证通过**

## 下一步操作

### 1. 下载模型（必需）

模型文件约 60GB，下载需要较长时间（取决于网络速度）。

```bash
# SSH 连接到服务器
ssh -p 2005 -i ~/.ssh/id_mac_to_3090 gexu@202.121.181.114

# 进入项目目录
cd /home/gexu/Qwen3VL

# 安装 HuggingFace CLI（如未安装）
pip install huggingface_hub

# 下载模型
bash scripts/download_model.sh

# 如果下载速度慢，使用国内镜像：
# export HF_ENDPOINT=https://hf-mirror.com
# bash scripts/download_model.sh
```

**预计时间：** 30 分钟 - 2 小时（取决于网络速度）

### 2. 启动服务

模型下载完成后，启动 Docker 服务：

```bash
# 构建自定义镜像（首次会下载/安装 torch、lmdeploy 等依赖）
docker compose build

# 启动服务（后台运行）
docker compose up -d

# 查看启动日志（实时）
docker compose logs -f
```

**提示：**
- 首次 `docker compose build` 需要下载/编译依赖，时间会明显长于拉取镜像
- 模型加载通常需要数分钟

等待日志中出现 "Application startup complete" 表示服务启动成功。

### 3. 测试服务

```bash
# 安装测试依赖
pip install requests

# 运行测试脚本
python test_api.py

# 如果需要测试图像推理
python test_api.py --image /path/to/image.jpg
```

### 4. 验证 GPU 使用

```bash
# 查看 GPU 使用情况
nvidia-smi

# 应该看到 4 张 GPU 都在使用中
```

## 快速命令参考

### 服务管理

```bash
# 查看服务状态
docker compose ps

# 查看日志
docker compose logs -f

# 重启服务
docker compose restart

# 停止服务
docker compose down

# 停止并删除容器
docker compose down -v
```

### 健康检查

```bash
# 本地测试
curl http://localhost:20000/health

# 远程测试（从本地机器）
curl http://202.121.181.114:20000/health
```

### API 测试

```bash
# 获取模型列表
curl http://localhost:20000/v1/models

# 文本推理
curl -X POST http://localhost:20000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-VL-32B-Instruct",
    "messages": [{"role": "user", "content": "你好"}],
    "max_tokens": 100
  }'
```

## 故障排查

### 问题：模型下载失败

**解决方案：**
1. 检查网络连接
2. 使用国内镜像：`export HF_ENDPOINT=https://hf-mirror.com`
3. 手动下载后上传到服务器

### 问题：容器启动失败

**解决方案：**
1. 检查 Docker 日志：`docker compose logs`
2. 验证 GPU 可用：`nvidia-smi`
3. 确认模型文件完整：`ls -la models/Qwen3-VL-32B-Instruct/`

### 问题：显存不足（OOM）

**解决方案：**
1. 降低 `--max-model-len` 参数（在 docker-compose.yml 中）
2. 确认没有其他程序占用 GPU
3. 重启服务：`docker compose restart`

## 性能预期

| 指标 | 预期值 |
|------|--------|
| 模型加载时间 | 3-5 分钟 |
| 首次推理延迟 | 5-10 秒 |
| 后续推理延迟 | 1-3 秒 |
| GPU 利用率 | 80-95% |
| 显存使用 | ~23GB/卡 |

## 远程访问

如果需要从本地机器访问服务器上的 API：

```bash
# 方法 1：SSH 隧道
ssh -p 2005 -i ~/.ssh/id_mac_to_3090 -L 20000:localhost:20000 gexu@202.121.181.114

# 然后在本地访问
curl http://localhost:20000/health

# 方法 2：直接访问（如果防火墙允许）
curl http://202.121.181.114:20000/health
```

## 文件清单

已创建的文件：

- ✅ `docker-compose.yml` - Docker Compose 配置
- ✅ `.env.example` - 环境变量示例
- ✅ `README.md` - 使用文档
- ✅ `test_api.py` - API 测试脚本
- ✅ `scripts/download_model.sh` - 模型下载脚本
- ✅ `scripts/verify_env.sh` - 环境验证脚本
- ✅ `DEPLOY.md` - 本部署指南

## 联系方式

如有问题，请查看：
1. README.md 中的故障排查章节
2. LMDeploy 文档：https://lmdeploy.readthedocs.io/
3. Qwen3-VL 模型文档：https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct
