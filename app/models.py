"""
数据模型定义
"""

from typing import List, Optional, Union
from pydantic import BaseModel, Field


class ImageInput(BaseModel):
    """图像输入（Base64 编码或 URL）"""

    type: str = Field(default="image", description="输入类型")
    data: str = Field(..., description="Base64 编码的图像数据或图像 URL")


class TextInput(BaseModel):
    """文本输入"""

    type: str = Field(default="text", description="输入类型")
    text: str = Field(..., description="文本内容")


class InferenceRequest(BaseModel):
    """推理请求"""

    prompt: str = Field(..., description="文本提示")
    image: Optional[str] = Field(None, description="Base64 编码的图像（可选）")
    max_tokens: Optional[int] = Field(None, description="最大生成 token 数")
    temperature: Optional[float] = Field(None, description="采样温度")
    top_p: Optional[float] = Field(None, description="Top-p 采样")


class MultiModalRequest(BaseModel):
    """多模态推理请求（支持多图像）"""

    messages: List[dict] = Field(..., description="对话消息列表")
    max_tokens: Optional[int] = Field(None, description="最大生成 token 数")
    temperature: Optional[float] = Field(None, description="采样温度")
    top_p: Optional[float] = Field(None, description="Top-p 采样")


class InferenceResponse(BaseModel):
    """推理响应"""

    text: str = Field(..., description="生成的文本")
    prompt_tokens: int = Field(..., description="输入 token 数")
    completion_tokens: int = Field(..., description="生成 token 数")
    total_tokens: int = Field(..., description="总 token 数")


class HealthResponse(BaseModel):
    """健康检查响应"""

    status: str = Field(..., description="服务状态")
    model: str = Field(..., description="模型名称")
    gpu_count: int = Field(..., description="GPU 数量")
