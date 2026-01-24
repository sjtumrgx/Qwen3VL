"""
Qwen3-VL 推理服务 FastAPI 应用

使用 LMDeploy 作为推理后端，提供 REST API 接口
"""

import base64
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import config
from .engine import LMDeployEngine, get_engine
from .models import (
    HealthResponse,
    InferenceRequest,
    InferenceResponse,
    MultiModalRequest,
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# 全局引擎实例
engine: Optional[LMDeployEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global engine

    # 启动时加载模型
    logger.info("=" * 50)
    logger.info("Qwen3-VL 推理服务启动中...")
    logger.info("=" * 50)

    start_time = time.time()
    engine = get_engine()
    load_time = time.time() - start_time

    logger.info(f"模型加载完成，耗时: {load_time:.2f} 秒")
    logger.info("=" * 50)
    logger.info("服务就绪！")
    logger.info("=" * 50)

    yield

    # 关闭时清理
    logger.info("服务关闭中...")


# 创建 FastAPI 应用
app = FastAPI(
    title="Qwen3-VL 推理服务",
    description="基于 LMDeploy 的 Qwen3-VL-32B-Instruct 视觉语言模型推理服务",
    version="1.0.0",
    lifespan=lifespan,
)

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查端点"""
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

    return HealthResponse(
        status="healthy" if engine is not None else "loading",
        model=config.model_path.split("/")[-1],
        gpu_count=gpu_count,
    )


@app.get("/v1/models")
async def list_models():
    """获取模型列表（OpenAI 兼容）"""
    model_name = config.model_path.split("/")[-1]
    return {
        "object": "list",
        "data": [
            {
                "id": model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
            }
        ],
    }


@app.post("/infer", response_model=InferenceResponse)
async def infer(request: InferenceRequest):
    """
    简单推理接口

    支持文本和单张图像输入
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="模型尚未加载完成")

    try:
        result = engine.generate(
            prompt=request.prompt,
            image=request.image,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )

        return InferenceResponse(**result)

    except Exception as e:
        logger.error(f"推理失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/infer/upload")
async def infer_with_upload(
    prompt: str = Form(..., description="文本提示"),
    image: UploadFile = File(None, description="图像文件"),
    max_tokens: Optional[int] = Form(None, description="最大生成 token 数"),
    temperature: Optional[float] = Form(None, description="采样温度"),
    top_p: Optional[float] = Form(None, description="Top-p 采样"),
):
    """
    文件上传推理接口

    支持直接上传图像文件
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="模型尚未加载完成")

    try:
        # 处理上传的图像
        image_base64 = None
        if image is not None:
            image_data = await image.read()
            image_base64 = base64.b64encode(image_data).decode("utf-8")

        result = engine.generate(
            prompt=prompt,
            image=image_base64,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        return InferenceResponse(**result)

    except Exception as e:
        logger.error(f"推理失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/chat/completions")
async def chat_completions(request: MultiModalRequest):
    """
    OpenAI 兼容的聊天补全接口

    支持多模态输入（文本 + 图像）
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="模型尚未加载完成")

    try:
        result = engine.generate_from_messages(
            messages=request.messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )

        # 构建 OpenAI 兼容的响应格式
        model_name = config.model_path.split("/")[-1]
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": result["text"],
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "total_tokens": result["total_tokens"],
            },
        }

    except Exception as e:
        logger.error(f"推理失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze")
async def analyze_image(
    image: str = Form(..., description="Base64 编码的图像"),
    instruction: str = Form(..., description="分析指令"),
    max_tokens: Optional[int] = Form(None, description="最大生成 token 数"),
):
    """
    图像分析接口

    专门用于图像分析任务
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="模型尚未加载完成")

    try:
        result = engine.generate(
            prompt=instruction,
            image=image,
            max_tokens=max_tokens,
        )

        return {
            "analysis": result["text"],
            "tokens": {
                "prompt": result["prompt_tokens"],
                "completion": result["completion_tokens"],
                "total": result["total_tokens"],
            },
        }

    except Exception as e:
        logger.error(f"分析失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/upload")
async def analyze_image_upload(
    image: UploadFile = File(..., description="图像文件"),
    instruction: str = Form(..., description="分析指令"),
    max_tokens: Optional[int] = Form(None, description="最大生成 token 数"),
):
    """
    图像分析接口（文件上传）

    支持直接上传图像文件进行分析
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="模型尚未加载完成")

    try:
        # 读取并编码图像
        image_data = await image.read()
        image_base64 = base64.b64encode(image_data).decode("utf-8")

        result = engine.generate(
            prompt=instruction,
            image=image_base64,
            max_tokens=max_tokens,
        )

        return {
            "analysis": result["text"],
            "tokens": {
                "prompt": result["prompt_tokens"],
                "completion": result["completion_tokens"],
                "total": result["total_tokens"],
            },
        }

    except Exception as e:
        logger.error(f"分析失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/gpu/status")
async def gpu_status():
    """获取 GPU 状态"""
    if not torch.cuda.is_available():
        return {"available": False, "gpus": []}

    gpus = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        memory_total = props.total_memory / (1024**3)  # GB
        memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)
        memory_reserved = torch.cuda.memory_reserved(i) / (1024**3)

        gpus.append(
            {
                "index": i,
                "name": props.name,
                "memory_total_gb": round(memory_total, 2),
                "memory_allocated_gb": round(memory_allocated, 2),
                "memory_reserved_gb": round(memory_reserved, 2),
            }
        )

    return {"available": True, "gpu_count": len(gpus), "gpus": gpus}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=config.host,
        port=config.port,
        reload=False,
    )
