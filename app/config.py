"""
配置管理模块
"""

import os
from dataclasses import dataclass


@dataclass
class Config:
    """应用配置"""

    # 模型配置
    model_path: str = os.getenv("MODEL_PATH", "/models/Qwen3-VL-32B-Instruct")
    tensor_parallel_size: int = int(os.getenv("TENSOR_PARALLEL_SIZE", "4"))
    max_model_len: int = int(os.getenv("MAX_MODEL_LEN", "8192"))
    trust_remote_code: bool = True

    # 推理配置
    max_tokens: int = int(os.getenv("MAX_TOKENS", "2048"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.7"))
    top_p: float = float(os.getenv("TOP_P", "0.9"))

    # 服务配置
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))

    # LMDeploy 配置（FastAPI 将请求转发到 LMDeploy OpenAI API Server）
    lmdeploy_base_url: str = os.getenv(
        "LMDEPLOY_BASE_URL",
        f"http://127.0.0.1:{os.getenv('LMDEPLOY_PORT', '23333')}",
    )
    lmdeploy_timeout_s: float = float(os.getenv("LMDEPLOY_TIMEOUT_S", "600"))

    # GPU 配置
    gpu_memory_utilization: float = float(
        os.getenv("GPU_MEMORY_UTILIZATION", "0.9")
    )


config = Config()
