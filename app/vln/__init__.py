"""
VLN (Vision-Language Navigation) 模块

为机器狗提供基于 Qwen3-VL 的视觉语言导航能力
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional
import time
import uuid


class TaskStatus(Enum):
    """任务状态"""
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Waypoint:
    """航点数据"""
    dx: float  # 前进距离 (m)
    dy: float  # 横向距离 (m)，正值为左
    dtheta: float  # 旋转角度 (rad)，正值为逆时针
    confidence: float = 1.0  # 置信度

    def to_dict(self) -> dict:
        return {
            "dx": self.dx,
            "dy": self.dy,
            "dtheta": self.dtheta,
            "confidence": self.confidence,
        }


@dataclass
class FrameContext:
    """帧上下文"""
    frame_id: int
    timestamp: float
    image_base64: str
    is_keyframe: bool = False
    environment_desc: str = ""
    action_desc: str = ""
    waypoints: List[Waypoint] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "is_keyframe": self.is_keyframe,
            "environment_desc": self.environment_desc,
            "action_desc": self.action_desc,
            "waypoints": [w.to_dict() for w in self.waypoints],
        }


@dataclass
class NavigationHistory:
    """导航历史记录"""
    keyframes: List[FrameContext] = field(default_factory=list)
    trajectory: List[Waypoint] = field(default_factory=list)  # 累积轨迹
    summary: str = ""  # 经历摘要
    max_keyframes: int = 10

    def add_keyframe(self, frame: FrameContext) -> None:
        """添加关键帧，超出限制时移除最旧的"""
        self.keyframes.append(frame)
        if len(self.keyframes) > self.max_keyframes:
            self.keyframes.pop(0)

    def add_waypoints(self, waypoints: List[Waypoint]) -> None:
        """添加航点到轨迹"""
        self.trajectory.extend(waypoints)

    def get_recent_keyframes(self, n: int = 5) -> List[FrameContext]:
        """获取最近 n 个关键帧"""
        return self.keyframes[-n:]

    def to_dict(self) -> dict:
        return {
            "keyframe_count": len(self.keyframes),
            "trajectory_length": len(self.trajectory),
            "summary": self.summary,
        }


@dataclass
class VLNTaskConfig:
    """VLN 任务配置"""
    output_fps: float = 5.0  # 输出帧率
    history_frames: int = 10  # 历史帧数量
    keyframe_threshold: float = 0.3  # 关键帧阈值
    max_waypoints_per_inference: int = 5  # 每次推理最大航点数
    enable_visualization: bool = True  # 是否启用可视化


@dataclass
class VLNTask:
    """VLN 导航任务"""
    task_id: str
    instruction: str  # 自然语言导航指令
    status: TaskStatus = TaskStatus.CREATED
    config: VLNTaskConfig = field(default_factory=VLNTaskConfig)
    history: NavigationHistory = field(default_factory=NavigationHistory)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    frame_count: int = 0
    current_environment: str = ""
    current_action: str = ""
    current_waypoints: List[Waypoint] = field(default_factory=list)
    error_message: str = ""

    @classmethod
    def create(cls, instruction: str, config: Optional[VLNTaskConfig] = None) -> "VLNTask":
        """创建新任务"""
        task_id = f"vln_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        return cls(
            task_id=task_id,
            instruction=instruction,
            config=config or VLNTaskConfig(),
        )

    def update_status(self, status: TaskStatus, error: str = "") -> None:
        """更新任务状态"""
        self.status = status
        self.updated_at = time.time()
        if error:
            self.error_message = error

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "instruction": self.instruction,
            "status": self.status.value,
            "frame_count": self.frame_count,
            "current_environment": self.current_environment,
            "current_action": self.current_action,
            "current_waypoints": [w.to_dict() for w in self.current_waypoints],
            "history": self.history.to_dict(),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "error_message": self.error_message,
        }


# 导出
__all__ = [
    "TaskStatus",
    "Waypoint",
    "FrameContext",
    "NavigationHistory",
    "VLNTaskConfig",
    "VLNTask",
]
