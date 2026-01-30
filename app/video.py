"""
视频处理工具

提供视频抽帧功能，将视频转换为多帧图像用于 Qwen3-VL 分析
"""

import base64
import logging
import tempfile
from pathlib import Path
from typing import List, Optional

import cv2

logger = logging.getLogger(__name__)


def extract_frames(
    video_path: str,
    max_frames: int = 8,
    resize_width: Optional[int] = 512,
) -> List[str]:
    """
    从视频中抽取关键帧

    Args:
        video_path: 视频文件路径
        max_frames: 最大抽帧数量
        resize_width: 调整宽度（保持宽高比），None 表示不调整

    Returns:
        Base64 编码的帧图像列表
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise ValueError("视频帧数为 0")

        # 计算抽帧间隔（均匀抽取）
        frame_indices = _calculate_frame_indices(total_frames, max_frames)
        logger.info(f"视频总帧数: {total_frames}, 抽取帧索引: {frame_indices}")

        frames_base64: List[str] = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"读取帧 {idx} 失败，跳过")
                continue

            # 调整尺寸
            if resize_width is not None:
                frame = _resize_frame(frame, resize_width)

            # 编码为 JPEG 并转 Base64
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_base64 = base64.b64encode(buffer).decode("utf-8")
            frames_base64.append(frame_base64)

        return frames_base64

    finally:
        cap.release()


def extract_frames_from_base64(
    video_base64: str,
    max_frames: int = 8,
    resize_width: Optional[int] = 512,
) -> List[str]:
    """
    从 Base64 编码的视频中抽取关键帧

    Args:
        video_base64: Base64 编码的视频数据
        max_frames: 最大抽帧数量
        resize_width: 调整宽度

    Returns:
        Base64 编码的帧图像列表
    """
    # 解码视频数据到临时文件
    video_data = base64.b64decode(video_base64)

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
        tmp_file.write(video_data)
        tmp_path = tmp_file.name

    try:
        return extract_frames(tmp_path, max_frames, resize_width)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _calculate_frame_indices(total_frames: int, max_frames: int) -> List[int]:
    """计算均匀分布的帧索引"""
    if total_frames <= max_frames:
        return list(range(total_frames))

    # 均匀抽取，包含首尾帧
    step = (total_frames - 1) / (max_frames - 1)
    return [int(i * step) for i in range(max_frames)]


def _resize_frame(frame, target_width: int):
    """按宽度等比缩放帧"""
    h, w = frame.shape[:2]
    if w <= target_width:
        return frame

    ratio = target_width / w
    new_h = int(h * ratio)
    return cv2.resize(frame, (target_width, new_h), interpolation=cv2.INTER_AREA)
