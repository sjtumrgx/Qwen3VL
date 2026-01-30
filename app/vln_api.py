"""
VLN REST API 端点

提供 VLN 导航服务的 HTTP 和 WebSocket 接口
"""

import asyncio
import base64
import logging
import time
from typing import List, Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from .vln import TaskStatus, VLNTask, VLNTaskConfig, Waypoint
from .vln.inference_pipeline import InferenceResult, get_inference_pipeline
from .vln.task_manager import get_task_manager
from .vln.visualizer import Visualizer
from .vln.waypoint_parser import WaypointParser

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/vln", tags=["VLN Navigation"])

# 全局实例
_visualizer = Visualizer()
_waypoint_parser = WaypointParser()


# ============== 请求/响应模型 ==============

class CreateTaskRequest(BaseModel):
    """创建任务请求"""
    instruction: str = Field(..., description="自然语言导航指令")
    config: Optional[dict] = Field(None, description="任务配置")


class CreateTaskResponse(BaseModel):
    """创建任务响应"""
    task_id: str
    status: str
    instruction: str


class FrameRequest(BaseModel):
    """帧请求"""
    task_id: str = Field(..., description="任务 ID")
    frame: str = Field(..., description="Base64 编码的图像")
    timestamp: Optional[float] = Field(None, description="时间戳")


class WaypointResponse(BaseModel):
    """航点响应"""
    dx: float
    dy: float
    dtheta: float
    confidence: float = 1.0


class FrameResponse(BaseModel):
    """帧响应"""
    task_id: str
    frame_id: int
    waypoints: List[WaypointResponse]
    environment: str
    action: str
    linear_vel: float
    angular_vel: float
    progress: float
    reached_goal: bool
    visualized_frame: Optional[str] = None
    inference_time: float


class TaskStatusResponse(BaseModel):
    """任务状态响应"""
    task_id: str
    status: str
    instruction: str
    frame_count: int
    current_environment: str
    current_action: str
    current_waypoints: List[WaypointResponse]
    progress: float
    error_message: str


class PipelineStatsResponse(BaseModel):
    """流水线统计响应"""
    total_requests: int
    processed: int
    dropped: int
    errors: int
    avg_inference_time: float
    queue_size: int
    running: bool


# ============== REST API 端点 ==============

@router.post("/task/create", response_model=CreateTaskResponse)
async def create_task(request: CreateTaskRequest):
    """
    创建新的导航任务

    Args:
        request: 包含导航指令和可选配置

    Returns:
        创建的任务信息
    """
    try:
        # 解析配置
        config = None
        if request.config:
            config = VLNTaskConfig(
                output_fps=request.config.get("output_fps", 5.0),
                history_frames=request.config.get("history_frames", 10),
                keyframe_threshold=request.config.get("keyframe_threshold", 0.3),
                enable_visualization=request.config.get("enable_visualization", True),
            )

        # 创建任务
        task_manager = get_task_manager()
        task = task_manager.create_task(request.instruction, config)

        # 更新状态为运行中
        task_manager.update_task_status(task.task_id, TaskStatus.RUNNING)

        # 确保推理流水线已启动
        pipeline = get_inference_pipeline()
        if not pipeline.stats["running"]:
            await pipeline.start()

        return CreateTaskResponse(
            task_id=task.task_id,
            status=task.status.value,
            instruction=task.instruction,
        )

    except Exception as e:
        logger.error(f"创建任务失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/frame", response_model=FrameResponse)
async def process_frame(request: FrameRequest):
    """
    处理单帧图像（非阻塞模式）

    提交帧到推理流水线，立即返回最新结果（不等待当前帧推理完成）
    这确保了 3-5Hz 的响应速度
    """
    try:
        task_manager = get_task_manager()
        task = task_manager.get_task(request.task_id)

        if not task:
            raise HTTPException(status_code=404, detail=f"任务不存在: {request.task_id}")

        if task.status not in (TaskStatus.RUNNING, TaskStatus.CREATED):
            raise HTTPException(
                status_code=400,
                detail=f"任务状态不允许处理帧: {task.status.value}",
            )

        timestamp = request.timestamp or time.time()

        # 提交到推理流水线（非阻塞）
        pipeline = get_inference_pipeline()

        # 提交帧，不等待结果
        await pipeline.submit_frame(
            task_id=request.task_id,
            image_base64=request.frame,
            timestamp=timestamp,
        )

        # 立即获取最新结果（可能是上一帧的结果）
        result = pipeline.get_latest_result(request.task_id)

        # 如果没有历史结果，返回默认值（首帧场景）
        if not result:
            # 首帧：返回默认前进动作
            from . import Waypoint
            default_waypoints = [Waypoint(dx=0.3, dy=0.0, dtheta=0.0)]
            v, w = _waypoint_parser.waypoints_to_velocity(default_waypoints)

            # 生成可视化帧（带默认信息）
            visualized_frame = None
            if task.config.enable_visualization:
                frame = _visualizer.base64_to_frame(request.frame)
                if frame is not None:
                    rendered = _visualizer.render_frame(
                        frame=frame,
                        waypoints=default_waypoints,
                        environment="正在分析环境...",
                        action="等待推理",
                        linear_vel=v,
                        angular_vel=w,
                        progress=0.0,
                    )
                    visualized_frame = _visualizer.frame_to_base64(rendered)

            return FrameResponse(
                task_id=request.task_id,
                frame_id=task.frame_count,
                waypoints=[
                    WaypointResponse(dx=0.3, dy=0.0, dtheta=0.0, confidence=0.5)
                ],
                environment="正在分析环境...",
                action="等待推理",
                linear_vel=v,
                angular_vel=w,
                progress=0.0,
                reached_goal=False,
                visualized_frame=visualized_frame,
                inference_time=0.0,
            )

        # 有历史结果，使用最新结果
        v, w = _waypoint_parser.waypoints_to_velocity(result.waypoints)

        # 生成可视化帧
        visualized_frame = None
        if task.config.enable_visualization:
            frame = _visualizer.base64_to_frame(request.frame)
            if frame is not None:
                rendered = _visualizer.render_frame(
                    frame=frame,
                    waypoints=result.waypoints,
                    environment=result.environment,
                    action=result.action,
                    linear_vel=v,
                    angular_vel=w,
                    progress=result.progress,
                )
                visualized_frame = _visualizer.frame_to_base64(rendered)

        return FrameResponse(
            task_id=result.task_id,
            frame_id=result.frame_id,
            waypoints=[
                WaypointResponse(
                    dx=wp.dx,
                    dy=wp.dy,
                    dtheta=wp.dtheta,
                    confidence=wp.confidence,
                )
                for wp in result.waypoints
            ],
            environment=result.environment,
            action=result.action,
            linear_vel=v,
            angular_vel=w,
            progress=result.progress,
            reached_goal=result.reached_goal,
            visualized_frame=visualized_frame,
            inference_time=result.inference_time,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"处理帧失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/frame/sync", response_model=FrameResponse)
async def process_frame_sync(request: FrameRequest):
    """
    处理单帧图像（同步模式）

    提交帧到推理流水线，等待推理完成后返回结果
    适用于测试或需要确保获取当前帧推理结果的场景
    """
    try:
        task_manager = get_task_manager()
        task = task_manager.get_task(request.task_id)

        if not task:
            raise HTTPException(status_code=404, detail=f"任务不存在: {request.task_id}")

        if task.status not in (TaskStatus.RUNNING, TaskStatus.CREATED):
            raise HTTPException(
                status_code=400,
                detail=f"任务状态不允许处理帧: {task.status.value}",
            )

        timestamp = request.timestamp or time.time()
        pipeline = get_inference_pipeline()

        # 创建事件等待结果
        result_event = asyncio.Event()
        result_holder = {"result": None}

        async def on_result(result: InferenceResult):
            result_holder["result"] = result
            result_event.set()

        # 提交帧
        submitted = await pipeline.submit_frame(
            task_id=request.task_id,
            image_base64=request.frame,
            timestamp=timestamp,
            callback=on_result,
        )

        if submitted:
            # 等待推理结果（最多等待 60 秒）
            try:
                await asyncio.wait_for(result_event.wait(), timeout=60.0)
            except asyncio.TimeoutError:
                result_holder["result"] = pipeline.get_latest_result(request.task_id)

        result = result_holder["result"]

        if not result:
            raise HTTPException(status_code=500, detail="推理超时，无结果返回")

        v, w = _waypoint_parser.waypoints_to_velocity(result.waypoints)

        visualized_frame = None
        if task.config.enable_visualization:
            frame = _visualizer.base64_to_frame(request.frame)
            if frame is not None:
                rendered = _visualizer.render_frame(
                    frame=frame,
                    waypoints=result.waypoints,
                    environment=result.environment,
                    action=result.action,
                    linear_vel=v,
                    angular_vel=w,
                    progress=result.progress,
                )
                visualized_frame = _visualizer.frame_to_base64(rendered)

        return FrameResponse(
            task_id=result.task_id,
            frame_id=result.frame_id,
            waypoints=[
                WaypointResponse(
                    dx=wp.dx,
                    dy=wp.dy,
                    dtheta=wp.dtheta,
                    confidence=wp.confidence,
                )
                for wp in result.waypoints
            ],
            environment=result.environment,
            action=result.action,
            linear_vel=v,
            angular_vel=w,
            progress=result.progress,
            reached_goal=result.reached_goal,
            visualized_frame=visualized_frame,
            inference_time=result.inference_time,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"同步处理帧失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/task/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """获取任务状态"""
    task_manager = get_task_manager()
    task = task_manager.get_task(task_id)

    if not task:
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")

    # 获取最新推理结果计算进度
    pipeline = get_inference_pipeline()
    latest_result = pipeline.get_latest_result(task_id)
    progress = latest_result.progress if latest_result else 0.0

    return TaskStatusResponse(
        task_id=task.task_id,
        status=task.status.value,
        instruction=task.instruction,
        frame_count=task.frame_count,
        current_environment=task.current_environment,
        current_action=task.current_action,
        current_waypoints=[
            WaypointResponse(
                dx=wp.dx,
                dy=wp.dy,
                dtheta=wp.dtheta,
                confidence=wp.confidence,
            )
            for wp in task.current_waypoints
        ],
        progress=progress,
        error_message=task.error_message,
    )


@router.post("/task/{task_id}/stop")
async def stop_task(task_id: str):
    """停止任务"""
    task_manager = get_task_manager()
    task = task_manager.get_task(task_id)

    if not task:
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")

    task_manager.update_task_status(task_id, TaskStatus.COMPLETED)

    # 清理流水线资源
    pipeline = get_inference_pipeline()
    pipeline.remove_task(task_id)

    return {"task_id": task_id, "status": "stopped"}


@router.delete("/task/{task_id}")
async def delete_task(task_id: str):
    """删除任务"""
    task_manager = get_task_manager()

    if not task_manager.delete_task(task_id):
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")

    # 清理流水线资源
    pipeline = get_inference_pipeline()
    pipeline.remove_task(task_id)

    return {"task_id": task_id, "deleted": True}


@router.get("/tasks")
async def list_tasks(status: Optional[str] = None):
    """列出所有任务"""
    task_manager = get_task_manager()

    filter_status = None
    if status:
        try:
            filter_status = TaskStatus(status)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"无效的状态: {status}")

    tasks = task_manager.list_tasks(filter_status)

    return {
        "tasks": [
            {
                "task_id": t.task_id,
                "instruction": t.instruction,
                "status": t.status.value,
                "frame_count": t.frame_count,
                "created_at": t.created_at,
            }
            for t in tasks
        ]
    }


@router.get("/pipeline/stats", response_model=PipelineStatsResponse)
async def get_pipeline_stats():
    """获取推理流水线统计"""
    pipeline = get_inference_pipeline()
    stats = pipeline.stats

    return PipelineStatsResponse(**stats)


# ============== WebSocket 端点 ==============

@router.websocket("/stream/{task_id}")
async def websocket_stream(websocket: WebSocket, task_id: str):
    """
    WebSocket 流式接口

    实时接收帧并返回导航结果
    """
    await websocket.accept()

    task_manager = get_task_manager()
    task = task_manager.get_task(task_id)

    if not task:
        await websocket.close(code=4004, reason=f"任务不存在: {task_id}")
        return

    pipeline = get_inference_pipeline()
    if not pipeline.stats["running"]:
        await pipeline.start()

    logger.info(f"WebSocket 连接建立: task={task_id}")

    try:
        while True:
            # 接收帧数据
            data = await websocket.receive_json()

            frame_base64 = data.get("frame")
            timestamp = data.get("timestamp", time.time())

            if not frame_base64:
                await websocket.send_json({"error": "缺少 frame 字段"})
                continue

            # 提交到流水线
            result_event = asyncio.Event()
            result_holder = {"result": None}

            async def on_result(result: InferenceResult):
                result_holder["result"] = result
                result_event.set()

            submitted = await pipeline.submit_frame(
                task_id=task_id,
                image_base64=frame_base64,
                timestamp=timestamp,
                callback=on_result,
            )

            # 等待结果或使用缓存
            if submitted:
                try:
                    await asyncio.wait_for(result_event.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    result_holder["result"] = pipeline.get_latest_result(task_id)

            result = result_holder["result"]

            if result:
                v, w = _waypoint_parser.waypoints_to_velocity(result.waypoints)

                # 生成可视化帧
                visualized_frame = None
                if task.config.enable_visualization:
                    frame = _visualizer.base64_to_frame(frame_base64)
                    if frame is not None:
                        rendered = _visualizer.render_frame(
                            frame=frame,
                            waypoints=result.waypoints,
                            environment=result.environment,
                            action=result.action,
                            linear_vel=v,
                            angular_vel=w,
                            progress=result.progress,
                        )
                        visualized_frame = _visualizer.frame_to_base64(rendered)

                await websocket.send_json({
                    "task_id": result.task_id,
                    "frame_id": result.frame_id,
                    "waypoints": [wp.to_dict() for wp in result.waypoints],
                    "environment": result.environment,
                    "action": result.action,
                    "linear_vel": v,
                    "angular_vel": w,
                    "progress": result.progress,
                    "reached_goal": result.reached_goal,
                    "visualized_frame": visualized_frame,
                    "inference_time": result.inference_time,
                })

                if result.reached_goal:
                    await websocket.send_json({"status": "goal_reached"})
                    break
            else:
                # 返回空结果
                await websocket.send_json({
                    "task_id": task_id,
                    "frame_id": task.frame_count,
                    "waypoints": [],
                    "environment": task.current_environment,
                    "action": "等待推理",
                    "linear_vel": 0.0,
                    "angular_vel": 0.0,
                    "progress": 0.0,
                    "reached_goal": False,
                    "visualized_frame": None,
                    "inference_time": 0.0,
                })

    except WebSocketDisconnect:
        logger.info(f"WebSocket 断开: task={task_id}")
    except Exception as e:
        logger.error(f"WebSocket 错误: {e}")
        await websocket.close(code=4000, reason=str(e))
    finally:
        # 清理
        pipeline.remove_task(task_id)
