"""
VLN 任务管理器

管理导航任务的生命周期、历史上下文和经历摘要
"""

import logging
import threading
from typing import Dict, List, Optional

from . import (
    FrameContext,
    NavigationHistory,
    TaskStatus,
    VLNTask,
    VLNTaskConfig,
    Waypoint,
)

logger = logging.getLogger(__name__)


class TaskManager:
    """VLN 任务管理器（线程安全）"""

    _instance: Optional["TaskManager"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._tasks: Dict[str, VLNTask] = {}
        self._task_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "TaskManager":
        """获取单例实例"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def create_task(
        self,
        instruction: str,
        config: Optional[VLNTaskConfig] = None,
    ) -> VLNTask:
        """
        创建新的导航任务

        Args:
            instruction: 自然语言导航指令
            config: 任务配置

        Returns:
            创建的任务对象
        """
        task = VLNTask.create(instruction, config)
        with self._task_lock:
            self._tasks[task.task_id] = task
        logger.info(f"创建任务: {task.task_id}, 指令: {instruction}")
        return task

    def get_task(self, task_id: str) -> Optional[VLNTask]:
        """获取任务"""
        with self._task_lock:
            return self._tasks.get(task_id)

    def list_tasks(self, status: Optional[TaskStatus] = None) -> List[VLNTask]:
        """
        列出任务

        Args:
            status: 可选，按状态过滤

        Returns:
            任务列表
        """
        with self._task_lock:
            tasks = list(self._tasks.values())
        if status:
            tasks = [t for t in tasks if t.status == status]
        return sorted(tasks, key=lambda t: t.created_at, reverse=True)

    def update_task_status(
        self,
        task_id: str,
        status: TaskStatus,
        error: str = "",
    ) -> bool:
        """
        更新任务状态

        Args:
            task_id: 任务 ID
            status: 新状态
            error: 错误信息（可选）

        Returns:
            是否成功
        """
        task = self.get_task(task_id)
        if not task:
            return False
        task.update_status(status, error)
        logger.info(f"任务 {task_id} 状态更新: {status.value}")
        return True

    def add_frame(
        self,
        task_id: str,
        frame: FrameContext,
    ) -> bool:
        """
        添加帧到任务

        Args:
            task_id: 任务 ID
            frame: 帧上下文

        Returns:
            是否成功
        """
        task = self.get_task(task_id)
        if not task:
            return False

        task.frame_count += 1
        task.updated_at = frame.timestamp

        # 如果是关键帧，添加到历史
        if frame.is_keyframe:
            task.history.add_keyframe(frame)

        # 更新当前状态
        if frame.environment_desc:
            task.current_environment = frame.environment_desc
        if frame.action_desc:
            task.current_action = frame.action_desc
        if frame.waypoints:
            task.current_waypoints = frame.waypoints
            task.history.add_waypoints(frame.waypoints)

        return True

    def get_history_context(
        self,
        task_id: str,
        n_frames: int = 5,
    ) -> Optional[Dict]:
        """
        获取任务的历史上下文（用于构建 prompt）

        Args:
            task_id: 任务 ID
            n_frames: 获取的关键帧数量

        Returns:
            历史上下文字典
        """
        task = self.get_task(task_id)
        if not task:
            return None

        keyframes = task.history.get_recent_keyframes(n_frames)
        return {
            "instruction": task.instruction,
            "keyframes": [f.to_dict() for f in keyframes],
            "trajectory_length": len(task.history.trajectory),
            "summary": task.history.summary,
            "current_environment": task.current_environment,
        }

    def update_summary(self, task_id: str, summary: str) -> bool:
        """
        更新任务的经历摘要

        Args:
            task_id: 任务 ID
            summary: 新摘要

        Returns:
            是否成功
        """
        task = self.get_task(task_id)
        if not task:
            return False
        task.history.summary = summary
        logger.debug(f"任务 {task_id} 摘要更新")
        return True

    def delete_task(self, task_id: str) -> bool:
        """删除任务"""
        with self._task_lock:
            if task_id in self._tasks:
                del self._tasks[task_id]
                logger.info(f"删除任务: {task_id}")
                return True
        return False

    def cleanup_completed_tasks(self, max_age_seconds: float = 3600) -> int:
        """
        清理已完成的旧任务

        Args:
            max_age_seconds: 最大保留时间（秒）

        Returns:
            清理的任务数量
        """
        import time
        now = time.time()
        to_delete = []

        with self._task_lock:
            for task_id, task in self._tasks.items():
                if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                    if now - task.updated_at > max_age_seconds:
                        to_delete.append(task_id)

            for task_id in to_delete:
                del self._tasks[task_id]

        if to_delete:
            logger.info(f"清理 {len(to_delete)} 个已完成任务")
        return len(to_delete)


def get_task_manager() -> TaskManager:
    """获取任务管理器实例"""
    return TaskManager.get_instance()
