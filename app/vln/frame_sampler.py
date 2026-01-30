"""
关键帧采样器

基于帧差和场景变化检测，动态选择关键帧以减少 VLM 推理负载
"""

import base64
import logging
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FrameSampler:
    """关键帧采样器"""

    def __init__(
        self,
        threshold: float = 0.3,
        min_interval: float = 0.2,
        max_interval: float = 2.0,
    ) -> None:
        """
        初始化采样器

        Args:
            threshold: 帧差阈值（0-1），越小越敏感
            min_interval: 最小采样间隔（秒）
            max_interval: 最大采样间隔（秒），超过则强制采样
        """
        self.threshold = threshold
        self.min_interval = min_interval
        self.max_interval = max_interval

        self._last_keyframe: Optional[np.ndarray] = None
        self._last_keyframe_time: float = 0.0
        self._frame_count: int = 0

    def reset(self) -> None:
        """重置采样器状态"""
        self._last_keyframe = None
        self._last_keyframe_time = 0.0
        self._frame_count = 0

    def should_sample(
        self,
        frame: np.ndarray,
        timestamp: float,
    ) -> Tuple[bool, float]:
        """
        判断是否应该采样当前帧

        Args:
            frame: BGR 图像
            timestamp: 时间戳

        Returns:
            (是否采样, 帧差分数)
        """
        self._frame_count += 1
        time_since_last = timestamp - self._last_keyframe_time

        # 首帧必须采样
        if self._last_keyframe is None:
            self._update_keyframe(frame, timestamp)
            return True, 1.0

        # 未达到最小间隔，不采样
        if time_since_last < self.min_interval:
            return False, 0.0

        # 计算帧差
        diff_score = self._compute_frame_diff(frame, self._last_keyframe)

        # 超过最大间隔，强制采样
        if time_since_last >= self.max_interval:
            self._update_keyframe(frame, timestamp)
            logger.debug(f"强制采样（超时）: diff={diff_score:.3f}")
            return True, diff_score

        # 帧差超过阈值，采样
        if diff_score >= self.threshold:
            self._update_keyframe(frame, timestamp)
            logger.debug(f"关键帧采样: diff={diff_score:.3f}")
            return True, diff_score

        return False, diff_score

    def should_sample_base64(
        self,
        image_base64: str,
        timestamp: float,
    ) -> Tuple[bool, float, Optional[np.ndarray]]:
        """
        判断是否应该采样（Base64 输入）

        Args:
            image_base64: Base64 编码的图像
            timestamp: 时间戳

        Returns:
            (是否采样, 帧差分数, 解码后的图像)
        """
        frame = self._decode_base64(image_base64)
        if frame is None:
            return False, 0.0, None

        is_keyframe, diff_score = self.should_sample(frame, timestamp)
        return is_keyframe, diff_score, frame

    def _update_keyframe(self, frame: np.ndarray, timestamp: float) -> None:
        """更新关键帧"""
        # 缩小存储以节省内存
        self._last_keyframe = self._resize_for_comparison(frame)
        self._last_keyframe_time = timestamp

    def _compute_frame_diff(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
    ) -> float:
        """
        计算两帧之间的差异分数

        使用多种特征综合评估：
        1. 直方图差异
        2. 结构相似度（简化版）
        """
        # 统一尺寸
        f1 = self._resize_for_comparison(frame1)
        f2 = frame2  # 已经是缩小的

        # 转灰度
        g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)

        # 方法1：直方图差异
        hist1 = cv2.calcHist([g1], [0], None, [64], [0, 256])
        hist2 = cv2.calcHist([g2], [0], None, [64], [0, 256])
        hist1 = cv2.normalize(hist1, hist1).flatten()
        hist2 = cv2.normalize(hist2, hist2).flatten()
        hist_diff = 1.0 - cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

        # 方法2：像素差异（归一化）
        pixel_diff = np.mean(np.abs(g1.astype(float) - g2.astype(float))) / 255.0

        # 方法3：边缘差异
        edges1 = cv2.Canny(g1, 50, 150)
        edges2 = cv2.Canny(g2, 50, 150)
        edge_diff = np.mean(np.abs(edges1.astype(float) - edges2.astype(float))) / 255.0

        # 综合评分（加权平均）
        score = 0.3 * hist_diff + 0.4 * pixel_diff + 0.3 * edge_diff
        return min(1.0, max(0.0, score))

    def _resize_for_comparison(
        self,
        frame: np.ndarray,
        target_width: int = 160,
    ) -> np.ndarray:
        """缩小图像用于比较"""
        h, w = frame.shape[:2]
        if w <= target_width:
            return frame
        ratio = target_width / w
        new_h = int(h * ratio)
        return cv2.resize(frame, (target_width, new_h), interpolation=cv2.INTER_AREA)

    def _decode_base64(self, image_base64: str) -> Optional[np.ndarray]:
        """解码 Base64 图像"""
        try:
            # 处理 data URL 格式
            if "," in image_base64:
                image_base64 = image_base64.split(",", 1)[1]
            image_data = base64.b64decode(image_base64)
            nparr = np.frombuffer(image_data, np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            logger.error(f"解码图像失败: {e}")
            return None

    @property
    def stats(self) -> dict:
        """获取采样统计"""
        return {
            "frame_count": self._frame_count,
            "last_keyframe_time": self._last_keyframe_time,
            "threshold": self.threshold,
        }
