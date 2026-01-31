"""
可视化渲染器

在视频帧上绘制导航信息：环境描述框、动作描述、路径曲线、速度信息
"""

import base64
import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import Waypoint

logger = logging.getLogger(__name__)

# 中文字体路径候选列表（按优先级）
FONT_PATHS = [
    # Linux 常见路径
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/wqy-zenhei/wqy-zenhei.ttc",
    "/usr/share/fonts/wqy-microhei/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    # macOS
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    # Windows
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
]


def _load_chinese_font(size: int = 16) -> ImageFont.FreeTypeFont:
    """加载中文字体，失败则返回默认字体"""
    for font_path in FONT_PATHS:
        try:
            return ImageFont.truetype(font_path, size)
        except (OSError, IOError):
            continue
    logger.warning("未找到中文字体，使用默认字体（中文可能显示异常）")
    return ImageFont.load_default()


class Visualizer:
    """VLN 可视化渲染器"""

    # 颜色定义 (BGR)
    COLOR_PATH_NEAR = (0, 255, 0)      # 绿色（近）
    COLOR_PATH_MID = (0, 255, 255)     # 黄色（中）
    COLOR_PATH_FAR = (0, 0, 255)       # 红色（远）
    COLOR_ENV_BOX_BG = (0, 80, 0)      # 环境框背景（深绿）
    COLOR_ENV_TEXT = (0, 255, 0)       # 环境框文字（绿色）
    COLOR_ACTION_BOX_BG = (0, 100, 139)  # 动作框背景（深橙）
    COLOR_ACTION_TEXT = (255, 255, 255)  # 动作框文字（白色）
    COLOR_SPEED_TEXT = (255, 255, 255)   # 速度文字（白色）
    COLOR_DETECTION_BOX = (255, 255, 0)  # 检测框（青色）
    COLOR_GESTURE_BOX = (0, 165, 255)    # 手势框（橙色）

    def __init__(
        self,
        font_scale: float = 0.5,
        line_thickness: int = 2,
        path_thickness: int = 3,
        box_alpha: float = 0.7,
    ) -> None:
        """
        初始化渲染器

        Args:
            font_scale: 字体缩放
            line_thickness: 线条粗细
            path_thickness: 路径线粗细
            box_alpha: 信息框透明度
        """
        self.font_scale = font_scale
        self.line_thickness = line_thickness
        self.path_thickness = path_thickness
        self.box_alpha = box_alpha
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        # 加载中文字体
        self._pil_font = _load_chinese_font(size=int(16 * font_scale / 0.5))
        self._pil_font_small = _load_chinese_font(size=int(14 * font_scale / 0.5))

    def render_frame(
        self,
        frame: np.ndarray,
        waypoints: List[Waypoint],
        environment: str = "",
        action: str = "",
        linear_vel: float = 0.0,
        angular_vel: float = 0.0,
        detection_boxes: Optional[List[Tuple[int, int, int, int, str]]] = None,
        gesture: str = "",
        progress: float = 0.0,
    ) -> np.ndarray:
        """
        渲染完整的可视化帧

        Args:
            frame: 输入帧 (BGR)
            waypoints: 航点列表
            environment: 环境描述
            action: 动作描述
            linear_vel: 线速度 (m/s)
            angular_vel: 角速度 (rad/s)
            detection_boxes: 检测框列表 [(x1, y1, x2, y2, label), ...]
            gesture: 手势识别结果
            progress: 导航进度 (0-1)

        Returns:
            渲染后的帧
        """
        output = frame.copy()
        h, w = output.shape[:2]

        # 1. 绘制路径曲线（从底部中央开始）
        if waypoints:
            output = self._draw_path_curve(output, waypoints)

        # 2. 绘制检测框
        if detection_boxes:
            for box in detection_boxes:
                output = self._draw_detection_box(output, box)

        # 3. 绘制手势状态
        if gesture:
            output = self._draw_gesture_status(output, gesture)

        # 4. 绘制环境描述框（右上）
        if environment:
            output = self._draw_info_box(
                output,
                text=environment,
                position="top_right",
                title="Environment:",
                bg_color=self.COLOR_ENV_BOX_BG,
                text_color=self.COLOR_ENV_TEXT,
            )

        # 5. 绘制动作描述框（右中）
        if action:
            output = self._draw_info_box(
                output,
                text=action,
                position="mid_right",
                title="",
                bg_color=self.COLOR_ACTION_BOX_BG,
                text_color=self.COLOR_ACTION_TEXT,
            )

        # 6. 绘制速度信息（左下）
        output = self._draw_speed_info(output, linear_vel, angular_vel)

        # 7. 绘制速度指示器（右下）
        speed = abs(linear_vel)
        output = self._draw_speed_indicator(output, speed)

        # 8. 绘制进度条（可选）
        if progress > 0:
            output = self._draw_progress_bar(output, progress)

        return output

    def _draw_path_curve(
        self,
        frame: np.ndarray,
        waypoints: List[Waypoint],
        scale: float = 100.0,
    ) -> np.ndarray:
        """
        绘制路径曲线

        从图像底部中央开始，根据航点绘制渐变色路径
        绿色（近）→ 黄色（中）→ 红色（远）

        坐标系说明：
        - 机器人坐标系：dx 前进（正向前），dy 横移（正向左），dtheta 旋转（正逆时针）
        - 图像坐标系：x 向右增大，y 向下增大
        - 转换：机器人前进 → 图像向上（y减小），机器人左移 → 图像向左（x减小）
        """
        h, w = frame.shape[:2]

        # 起点：底部中央
        start_x = w // 2
        start_y = h

        # 将航点转换为像素坐标
        points = [(start_x, start_y)]
        current_x, current_y = float(start_x), float(start_y)
        current_theta = 0.0  # 当前朝向（图像坐标系中，0 表示向上）

        for wp in waypoints:
            # 先旋转，再移动（更符合实际机器人运动）
            current_theta += wp.dtheta

            # 机器人坐标系到图像坐标系的转换：
            # - 机器人 dx（前进）→ 沿当前朝向移动
            # - 机器人 dy（左移）→ 垂直于朝向向左移动
            # 图像坐标系中：向上为 -y，向左为 -x
            # current_theta = 0 时朝向图像上方

            # 前进方向在图像中的分量
            forward_x = np.sin(current_theta)   # 朝向的 x 分量
            forward_y = -np.cos(current_theta)  # 朝向的 y 分量（向上为负）

            # 左移方向在图像中的分量（垂直于前进方向，逆时针90度）
            left_x = -np.cos(current_theta)     # 左方向的 x 分量
            left_y = -np.sin(current_theta)     # 左方向的 y 分量

            # 计算像素位移
            dx_pixel = (wp.dx * forward_x + wp.dy * left_x) * scale
            dy_pixel = (wp.dx * forward_y + wp.dy * left_y) * scale

            current_x += dx_pixel
            current_y += dy_pixel

            # 限制在图像范围内
            current_x = max(0, min(w - 1, current_x))
            current_y = max(0, min(h - 1, current_y))

            points.append((int(current_x), int(current_y)))

        # 如果点太少，添加插值点使曲线更平滑
        if len(points) >= 2:
            points = self._smooth_path(points)

        # 绘制渐变色路径
        if len(points) >= 2:
            frame = self._draw_gradient_path(frame, points)

        return frame

    def _smooth_path(
        self,
        points: List[Tuple[int, int]],
        num_interpolate: int = 10,
    ) -> List[Tuple[int, int]]:
        """使用贝塞尔曲线平滑路径"""
        if len(points) < 2:
            return points

        # 简单线性插值（可以升级为贝塞尔曲线）
        smoothed = []
        for i in range(len(points) - 1):
            p1 = points[i]
            p2 = points[i + 1]
            for t in range(num_interpolate):
                ratio = t / num_interpolate
                x = int(p1[0] + (p2[0] - p1[0]) * ratio)
                y = int(p1[1] + (p2[1] - p1[1]) * ratio)
                smoothed.append((x, y))
        smoothed.append(points[-1])
        return smoothed

    def _draw_gradient_path(
        self,
        frame: np.ndarray,
        points: List[Tuple[int, int]],
    ) -> np.ndarray:
        """绘制渐变色路径"""
        if len(points) < 2:
            return frame

        n = len(points)
        for i in range(n - 1):
            # 计算渐变色
            ratio = i / (n - 1)
            color = self._interpolate_color(ratio)

            # 绘制线段
            cv2.line(
                frame,
                points[i],
                points[i + 1],
                color,
                self.path_thickness,
                cv2.LINE_AA,
            )

        # 在起点绘制圆点
        cv2.circle(frame, points[0], 6, self.COLOR_PATH_NEAR, -1, cv2.LINE_AA)

        return frame

    def _interpolate_color(self, ratio: float) -> Tuple[int, int, int]:
        """
        根据比例插值颜色

        0.0 -> 绿色
        0.5 -> 黄色
        1.0 -> 红色
        """
        if ratio < 0.5:
            # 绿色 -> 黄色
            t = ratio * 2
            return (
                int(self.COLOR_PATH_NEAR[0] + (self.COLOR_PATH_MID[0] - self.COLOR_PATH_NEAR[0]) * t),
                int(self.COLOR_PATH_NEAR[1] + (self.COLOR_PATH_MID[1] - self.COLOR_PATH_NEAR[1]) * t),
                int(self.COLOR_PATH_NEAR[2] + (self.COLOR_PATH_MID[2] - self.COLOR_PATH_NEAR[2]) * t),
            )
        else:
            # 黄色 -> 红色
            t = (ratio - 0.5) * 2
            return (
                int(self.COLOR_PATH_MID[0] + (self.COLOR_PATH_FAR[0] - self.COLOR_PATH_MID[0]) * t),
                int(self.COLOR_PATH_MID[1] + (self.COLOR_PATH_FAR[1] - self.COLOR_PATH_MID[1]) * t),
                int(self.COLOR_PATH_MID[2] + (self.COLOR_PATH_FAR[2] - self.COLOR_PATH_MID[2]) * t),
            )

    def _draw_info_box(
        self,
        frame: np.ndarray,
        text: str,
        position: str,
        title: str = "",
        bg_color: Tuple[int, int, int] = (0, 0, 0),
        text_color: Tuple[int, int, int] = (255, 255, 255),
        max_width: int = 250,
        padding: int = 10,
    ) -> np.ndarray:
        """绘制信息框（支持中文）"""
        h, w = frame.shape[:2]

        # 文本换行
        lines = self._wrap_text(text, max_width)
        if title:
            lines = [title] + lines

        # 计算框大小
        line_height = int(20 * self.font_scale / 0.5)
        box_height = len(lines) * line_height + padding * 2
        box_width = max_width + padding * 2

        # 确定位置
        if position == "top_right":
            x = w - box_width - 10
            y = 10
        elif position == "mid_right":
            x = w - box_width - 10
            y = h // 3
        elif position == "bottom_right":
            x = w - box_width - 10
            y = h - box_height - 50
        else:
            x, y = 10, 10

        # 绘制半透明背景
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (x, y),
            (x + box_width, y + box_height),
            bg_color,
            -1,
        )
        frame = cv2.addWeighted(overlay, self.box_alpha, frame, 1 - self.box_alpha, 0)

        # 绘制边框
        cv2.rectangle(
            frame,
            (x, y),
            (x + box_width, y + box_height),
            text_color,
            1,
        )

        # 使用 PIL 绘制中文文本
        frame = self._put_chinese_text(
            frame,
            lines,
            x + padding,
            y + padding,
            line_height,
            text_color,
        )

        return frame

    def _put_chinese_text(
        self,
        frame: np.ndarray,
        lines: List[str],
        x: int,
        y: int,
        line_height: int,
        color: Tuple[int, int, int],
    ) -> np.ndarray:
        """使用 PIL 绘制中文文本"""
        # OpenCV BGR -> PIL RGB
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        # BGR -> RGB 颜色转换
        rgb_color = (color[2], color[1], color[0])

        for i, line in enumerate(lines):
            text_y = y + i * line_height
            draw.text((x, text_y), line, font=self._pil_font, fill=rgb_color)

        # PIL RGB -> OpenCV BGR
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def _wrap_text(self, text: str, max_width: int) -> List[str]:
        """文本换行"""
        # 简单按字符数换行（中文约15字符，英文约30字符）
        chars_per_line = max_width // 8
        lines = []
        current_line = ""

        for char in text:
            current_line += char
            # 中文字符占2个宽度
            width = sum(2 if ord(c) > 127 else 1 for c in current_line)
            if width >= chars_per_line:
                lines.append(current_line)
                current_line = ""

        if current_line:
            lines.append(current_line)

        return lines[:5]  # 最多5行

    def _draw_detection_box(
        self,
        frame: np.ndarray,
        box: Tuple[int, int, int, int, str],
    ) -> np.ndarray:
        """绘制检测框（支持中文标签）"""
        x1, y1, x2, y2, label = box

        # 绘制框
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            self.COLOR_DETECTION_BOX,
            self.line_thickness,
        )

        # 绘制标签
        if label:
            # 使用 PIL 计算文本大小并绘制
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)
            bbox = draw.textbbox((0, 0), label, font=self._pil_font_small)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]

            # 绘制标签背景
            cv2.rectangle(
                frame,
                (x1, y1 - text_h - 10),
                (x1 + text_w + 10, y1),
                self.COLOR_DETECTION_BOX,
                -1,
            )

            # 使用 PIL 绘制文本
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)
            draw.text((x1 + 5, y1 - text_h - 5), label, font=self._pil_font_small, fill=(0, 0, 0))
            frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        return frame

    def _draw_gesture_status(
        self,
        frame: np.ndarray,
        gesture: str,
    ) -> np.ndarray:
        """绘制手势状态（支持中文）"""
        h, w = frame.shape[:2]

        # 顶部中央
        text = f"[{gesture.upper()}]"

        # 使用 PIL 计算文本大小
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        bbox = draw.textbbox((0, 0), text, font=self._pil_font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        x = (w - text_w) // 2
        y = 30

        # 背景
        cv2.rectangle(
            frame,
            (x - 5, y - 5),
            (x + text_w + 5, y + text_h + 5),
            self.COLOR_GESTURE_BOX,
            -1,
        )

        # 使用 PIL 绘制文本
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        draw.text((x, y), text, font=self._pil_font, fill=(255, 255, 255))
        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        return frame

    def _draw_speed_info(
        self,
        frame: np.ndarray,
        linear_vel: float,
        angular_vel: float,
    ) -> np.ndarray:
        """绘制速度信息（左下角）"""
        h, w = frame.shape[:2]

        # 线速度
        text1 = f"Linear: {linear_vel:+.3f} m/s"
        cv2.putText(
            frame,
            text1,
            (10, h - 40),
            self.font,
            self.font_scale,
            self.COLOR_SPEED_TEXT,
            1,
            cv2.LINE_AA,
        )

        # 角速度
        text2 = f"Angular: {angular_vel:+.3f} rad/s"
        cv2.putText(
            frame,
            text2,
            (10, h - 15),
            self.font,
            self.font_scale,
            self.COLOR_SPEED_TEXT,
            1,
            cv2.LINE_AA,
        )

        return frame

    def _draw_speed_indicator(
        self,
        frame: np.ndarray,
        speed: float,
        max_speed: float = 2.0,
    ) -> np.ndarray:
        """绘制速度指示器（右下角）"""
        h, w = frame.shape[:2]

        text = f"Speed: {speed:.2f} m/s"
        text_size = cv2.getTextSize(text, self.font, self.font_scale, 1)[0]
        x = w - text_size[0] - 20
        y = h - 15

        # 背景
        cv2.rectangle(
            frame,
            (x - 5, y - text_size[1] - 5),
            (x + text_size[0] + 5, y + 5),
            (100, 100, 100),
            -1,
        )

        # 文字
        cv2.putText(
            frame,
            text,
            (x, y),
            self.font,
            self.font_scale,
            self.COLOR_SPEED_TEXT,
            1,
            cv2.LINE_AA,
        )

        return frame

    def _draw_progress_bar(
        self,
        frame: np.ndarray,
        progress: float,
    ) -> np.ndarray:
        """绘制进度条"""
        h, w = frame.shape[:2]

        bar_width = 200
        bar_height = 10
        x = (w - bar_width) // 2
        y = h - 60

        # 背景
        cv2.rectangle(
            frame,
            (x, y),
            (x + bar_width, y + bar_height),
            (50, 50, 50),
            -1,
        )

        # 进度
        progress_width = int(bar_width * min(1.0, max(0.0, progress)))
        if progress_width > 0:
            cv2.rectangle(
                frame,
                (x, y),
                (x + progress_width, y + bar_height),
                self.COLOR_PATH_NEAR,
                -1,
            )

        # 边框
        cv2.rectangle(
            frame,
            (x, y),
            (x + bar_width, y + bar_height),
            (200, 200, 200),
            1,
        )

        return frame

    def frame_to_base64(self, frame: np.ndarray, quality: int = 85) -> str:
        """将帧转换为 Base64"""
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return base64.b64encode(buffer).decode("utf-8")

    def base64_to_frame(self, image_base64: str) -> Optional[np.ndarray]:
        """将 Base64 转换为帧"""
        try:
            if "," in image_base64:
                image_base64 = image_base64.split(",", 1)[1]
            image_data = base64.b64decode(image_base64)
            nparr = np.frombuffer(image_data, np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            logger.error(f"解码图像失败: {e}")
            return None
