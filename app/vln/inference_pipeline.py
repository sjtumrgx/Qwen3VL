"""
异步推理流水线

管理 VLN 推理请求队列，确保 VLM 处理实时图像
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from ..engine import get_engine
from . import FrameContext, TaskStatus, Waypoint
from .frame_sampler import FrameSampler
from .prompts import build_messages_with_images, build_navigation_prompt
from .task_manager import get_task_manager
from .waypoint_parser import WaypointParser

logger = logging.getLogger(__name__)


@dataclass
class InferenceRequest:
    """推理请求"""
    task_id: str
    frame_id: int
    timestamp: float
    image_base64: str
    priority: int = 0  # 优先级，越高越优先


@dataclass
class InferenceResult:
    """推理结果"""
    task_id: str
    frame_id: int
    timestamp: float
    waypoints: List[Waypoint]
    environment: str
    action: str
    reasoning: str
    progress: float
    reached_goal: bool
    inference_time: float
    error: str = ""


class InferencePipeline:
    """异步推理流水线"""

    _instance: Optional["InferencePipeline"] = None

    def __init__(
        self,
        max_queue_size: int = 10,
        drop_stale_frames: bool = True,
        stale_threshold: float = 0.5,
    ) -> None:
        """
        初始化流水线

        Args:
            max_queue_size: 最大队列长度
            drop_stale_frames: 是否丢弃过时帧
            stale_threshold: 过时阈值（秒）
        """
        self.max_queue_size = max_queue_size
        self.drop_stale_frames = drop_stale_frames
        self.stale_threshold = stale_threshold

        self._queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self._results: Dict[str, InferenceResult] = {}  # task_id -> latest result
        self._callbacks: Dict[str, Callable] = {}  # task_id -> callback
        self._running = False
        self._worker_task: Optional[asyncio.Task] = None

        self._frame_samplers: Dict[str, FrameSampler] = {}
        self._waypoint_parser = WaypointParser()
        self._task_manager = get_task_manager()

        # 统计
        self._stats = {
            "total_requests": 0,
            "processed": 0,
            "dropped": 0,
            "errors": 0,
            "avg_inference_time": 0.0,
        }

    @classmethod
    def get_instance(cls) -> "InferencePipeline":
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def start(self) -> None:
        """启动流水线"""
        if self._running:
            return
        self._running = True
        self._worker_task = asyncio.create_task(self._worker_loop())
        logger.info("推理流水线已启动")

    async def stop(self) -> None:
        """停止流水线"""
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        logger.info("推理流水线已停止")

    def get_frame_sampler(self, task_id: str, threshold: float = 0.3) -> FrameSampler:
        """获取任务的帧采样器"""
        if task_id not in self._frame_samplers:
            self._frame_samplers[task_id] = FrameSampler(threshold=threshold)
        return self._frame_samplers[task_id]

    async def submit_frame(
        self,
        task_id: str,
        image_base64: str,
        timestamp: Optional[float] = None,
        callback: Optional[Callable] = None,
    ) -> bool:
        """
        提交帧进行推理

        Args:
            task_id: 任务 ID
            image_base64: Base64 编码的图像
            timestamp: 时间戳
            callback: 结果回调函数

        Returns:
            是否成功提交
        """
        if not self._running:
            logger.warning("流水线未启动")
            return False

        task = self._task_manager.get_task(task_id)
        if not task:
            logger.warning(f"任务不存在: {task_id}")
            return False

        timestamp = timestamp or time.time()

        # 关键帧采样
        sampler = self.get_frame_sampler(task_id, task.config.keyframe_threshold)
        is_keyframe, diff_score, _ = sampler.should_sample_base64(image_base64, timestamp)

        if not is_keyframe:
            # 非关键帧，返回上次结果
            return False

        # 创建请求
        request = InferenceRequest(
            task_id=task_id,
            frame_id=task.frame_count + 1,
            timestamp=timestamp,
            image_base64=image_base64,
        )

        # 注册回调
        if callback:
            self._callbacks[task_id] = callback

        # 尝试入队
        try:
            if self._queue.full():
                # 队列满，丢弃最旧的请求
                try:
                    old_request = self._queue.get_nowait()
                    self._stats["dropped"] += 1
                    logger.debug(f"丢弃旧请求: task={old_request.task_id}, frame={old_request.frame_id}")
                except asyncio.QueueEmpty:
                    pass

            await asyncio.wait_for(
                self._queue.put(request),
                timeout=0.1,
            )
            self._stats["total_requests"] += 1
            return True

        except asyncio.TimeoutError:
            logger.warning("提交请求超时")
            return False

    async def _worker_loop(self) -> None:
        """工作循环"""
        engine = get_engine()

        while self._running:
            try:
                # 获取请求
                request = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=1.0,
                )

                # 检查是否过时
                if self.drop_stale_frames:
                    age = time.time() - request.timestamp
                    if age > self.stale_threshold:
                        self._stats["dropped"] += 1
                        logger.debug(f"丢弃过时帧: age={age:.2f}s")
                        continue

                # 执行推理
                result = await self._process_request(request, engine)

                # 存储结果
                self._results[request.task_id] = result

                # 更新任务
                self._update_task_with_result(request, result)

                # 调用回调
                callback = self._callbacks.get(request.task_id)
                if callback:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(result)
                        else:
                            callback(result)
                    except Exception as e:
                        logger.error(f"回调执行失败: {e}")

                self._stats["processed"] += 1

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"推理工作循环错误: {e}")
                self._stats["errors"] += 1

    async def _process_request(
        self,
        request: InferenceRequest,
        engine: Any,
    ) -> InferenceResult:
        """处理单个推理请求"""
        start_time = time.time()

        try:
            # 获取任务上下文
            task = self._task_manager.get_task(request.task_id)
            if not task:
                raise ValueError(f"任务不存在: {request.task_id}")

            # 获取历史上下文
            history_ctx = self._task_manager.get_history_context(
                request.task_id,
                n_frames=3,
            )

            # 构建 prompt
            prompt = build_navigation_prompt(
                instruction=task.instruction,
                history_summary=history_ctx.get("summary", "") if history_ctx else "",
                current_environment=history_ctx.get("current_environment", "") if history_ctx else "",
            )

            # 构建消息
            messages = build_messages_with_images(
                prompt=prompt,
                image_base64_list=[request.image_base64],
            )

            # 调用 VLM（在线程池中执行同步调用）
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: engine.generate_from_messages(
                    messages=messages,
                    max_tokens=512,
                    temperature=0.3,
                ),
            )

            # 解析结果
            parsed = self._waypoint_parser.parse(response.get("text", ""))

            inference_time = time.time() - start_time

            # 更新平均推理时间
            n = self._stats["processed"]
            avg = self._stats["avg_inference_time"]
            self._stats["avg_inference_time"] = (avg * n + inference_time) / (n + 1)

            return InferenceResult(
                task_id=request.task_id,
                frame_id=request.frame_id,
                timestamp=request.timestamp,
                waypoints=parsed["waypoints"],
                environment=parsed["environment"],
                action=parsed["action"],
                reasoning=parsed["reasoning"],
                progress=parsed["progress"],
                reached_goal=parsed["reached_goal"],
                inference_time=inference_time,
            )

        except Exception as e:
            logger.error(f"推理失败: {e}")
            return InferenceResult(
                task_id=request.task_id,
                frame_id=request.frame_id,
                timestamp=request.timestamp,
                waypoints=[],
                environment="",
                action="",
                reasoning="",
                progress=0.0,
                reached_goal=False,
                inference_time=time.time() - start_time,
                error=str(e),
            )

    def _update_task_with_result(
        self,
        request: InferenceRequest,
        result: InferenceResult,
    ) -> None:
        """用推理结果更新任务"""
        frame = FrameContext(
            frame_id=request.frame_id,
            timestamp=request.timestamp,
            image_base64=request.image_base64,
            is_keyframe=True,
            environment_desc=result.environment,
            action_desc=result.action,
            waypoints=result.waypoints,
        )

        self._task_manager.add_frame(request.task_id, frame)

        # 检查是否到达目标
        if result.reached_goal:
            self._task_manager.update_task_status(
                request.task_id,
                TaskStatus.COMPLETED,
            )

    def get_latest_result(self, task_id: str) -> Optional[InferenceResult]:
        """获取任务的最新推理结果"""
        return self._results.get(task_id)

    def remove_task(self, task_id: str) -> None:
        """移除任务相关资源"""
        self._results.pop(task_id, None)
        self._callbacks.pop(task_id, None)
        self._frame_samplers.pop(task_id, None)

    @property
    def stats(self) -> dict:
        """获取统计信息"""
        return {
            **self._stats,
            "queue_size": self._queue.qsize(),
            "running": self._running,
        }


def get_inference_pipeline() -> InferencePipeline:
    """获取推理流水线实例"""
    return InferencePipeline.get_instance()
