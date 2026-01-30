"""
航点解析器

解析 VLM 输出为航点序列
"""

import json
import logging
import re
from typing import Dict, List, Optional, Tuple

from . import Waypoint

logger = logging.getLogger(__name__)


class WaypointParser:
    """航点解析器"""

    # 离散动作到航点的映射
    DISCRETE_ACTION_MAP = {
        "forward": Waypoint(dx=0.5, dy=0.0, dtheta=0.0),
        "前进": Waypoint(dx=0.5, dy=0.0, dtheta=0.0),
        "backward": Waypoint(dx=-0.3, dy=0.0, dtheta=0.0),
        "后退": Waypoint(dx=-0.3, dy=0.0, dtheta=0.0),
        "left": Waypoint(dx=0.0, dy=0.0, dtheta=0.523),  # 30度
        "turn_left": Waypoint(dx=0.0, dy=0.0, dtheta=0.523),
        "左转": Waypoint(dx=0.0, dy=0.0, dtheta=0.523),
        "right": Waypoint(dx=0.0, dy=0.0, dtheta=-0.523),
        "turn_right": Waypoint(dx=0.0, dy=0.0, dtheta=-0.523),
        "右转": Waypoint(dx=0.0, dy=0.0, dtheta=-0.523),
        "stop": Waypoint(dx=0.0, dy=0.0, dtheta=0.0),
        "停止": Waypoint(dx=0.0, dy=0.0, dtheta=0.0),
        "slight_left": Waypoint(dx=0.3, dy=0.1, dtheta=0.1),
        "左前": Waypoint(dx=0.3, dy=0.1, dtheta=0.1),
        "slight_right": Waypoint(dx=0.3, dy=-0.1, dtheta=-0.1),
        "右前": Waypoint(dx=0.3, dy=-0.1, dtheta=-0.1),
    }

    def __init__(
        self,
        max_dx: float = 1.0,
        max_dy: float = 0.5,
        max_dtheta: float = 1.0,
    ) -> None:
        """
        初始化解析器

        Args:
            max_dx: 最大前进距离限制
            max_dy: 最大横向距离限制
            max_dtheta: 最大旋转角度限制
        """
        self.max_dx = max_dx
        self.max_dy = max_dy
        self.max_dtheta = max_dtheta

    def parse(self, vlm_output: str) -> Dict:
        """
        解析 VLM 输出

        Args:
            vlm_output: VLM 生成的文本

        Returns:
            解析结果字典，包含:
            - waypoints: 航点列表
            - environment: 环境描述
            - action: 动作描述
            - reasoning: 推理过程
            - progress: 进度
            - reached_goal: 是否到达目标
            - raw_output: 原始输出
        """
        result = {
            "waypoints": [],
            "environment": "",
            "action": "",
            "reasoning": "",
            "progress": 0.0,
            "reached_goal": False,
            "raw_output": vlm_output,
        }

        # 尝试解析 JSON
        json_data = self._extract_json(vlm_output)
        if json_data:
            result.update(self._parse_json_response(json_data))
        else:
            # 回退到文本解析
            result.update(self._parse_text_response(vlm_output))

        # 验证和限制航点
        result["waypoints"] = self._validate_waypoints(result["waypoints"])

        return result

    def _extract_json(self, text: str) -> Optional[dict]:
        """从文本中提取 JSON"""
        # 尝试直接解析
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        # 尝试提取 JSON 块
        patterns = [
            r"```json\s*([\s\S]*?)\s*```",
            r"```\s*([\s\S]*?)\s*```",
            r"\{[\s\S]*\}",
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    json_str = match.group(1) if "```" in pattern else match.group(0)
                    return json.loads(json_str)
                except (json.JSONDecodeError, IndexError):
                    continue

        return None

    def _parse_json_response(self, data: dict) -> Dict:
        """解析 JSON 格式的响应"""
        result = {
            "environment": str(data.get("environment", "")),
            "action": str(data.get("action", "")),
            "reasoning": str(data.get("reasoning", "")),
            "progress": float(data.get("progress", 0.0)),
            "reached_goal": bool(data.get("reached_goal", False)),
            "waypoints": [],
        }

        # 解析航点
        waypoints_data = data.get("waypoints", [])
        if isinstance(waypoints_data, list):
            for wp in waypoints_data:
                if isinstance(wp, dict):
                    waypoint = Waypoint(
                        dx=float(wp.get("dx", 0.0)),
                        dy=float(wp.get("dy", 0.0)),
                        dtheta=float(wp.get("dtheta", 0.0)),
                        confidence=float(wp.get("confidence", 1.0)),
                    )
                    result["waypoints"].append(waypoint)

        return result

    def _parse_text_response(self, text: str) -> Dict:
        """解析纯文本响应（回退方案）"""
        result = {
            "environment": "",
            "action": "",
            "reasoning": text[:200],
            "progress": 0.0,
            "reached_goal": False,
            "waypoints": [],
        }

        text_lower = text.lower()

        # 检测是否到达目标
        goal_keywords = ["到达", "完成", "reached", "arrived", "goal"]
        if any(kw in text_lower for kw in goal_keywords):
            result["reached_goal"] = True
            result["waypoints"] = [Waypoint(dx=0.0, dy=0.0, dtheta=0.0)]
            return result

        # 尝试匹配离散动作
        for action, waypoint in self.DISCRETE_ACTION_MAP.items():
            if action in text_lower:
                result["waypoints"] = [waypoint]
                result["action"] = action
                break

        # 如果没有匹配到动作，默认前进
        if not result["waypoints"]:
            result["waypoints"] = [Waypoint(dx=0.3, dy=0.0, dtheta=0.0)]
            result["action"] = "默认前进"

        # 提取环境描述（简单启发式）
        env_patterns = [
            r"环境[：:]\s*(.+?)(?:\n|$)",
            r"environment[：:]\s*(.+?)(?:\n|$)",
            r"看到(.+?)(?:\n|。|$)",
        ]
        for pattern in env_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result["environment"] = match.group(1).strip()[:100]
                break

        return result

    def _validate_waypoints(self, waypoints: List[Waypoint]) -> List[Waypoint]:
        """验证和限制航点"""
        validated = []
        for wp in waypoints[:5]:  # 最多5个航点
            validated_wp = Waypoint(
                dx=max(-self.max_dx, min(self.max_dx, wp.dx)),
                dy=max(-self.max_dy, min(self.max_dy, wp.dy)),
                dtheta=max(-self.max_dtheta, min(self.max_dtheta, wp.dtheta)),
                confidence=max(0.0, min(1.0, wp.confidence)),
            )
            validated.append(validated_wp)
        return validated

    def waypoints_to_velocity(
        self,
        waypoints: List[Waypoint],
        dt: float = 0.2,
    ) -> Tuple[float, float]:
        """
        将航点转换为速度命令（用于直接控制）

        Args:
            waypoints: 航点列表
            dt: 时间步长

        Returns:
            (linear_velocity, angular_velocity)
        """
        if not waypoints:
            return 0.0, 0.0

        # 使用第一个航点计算速度
        wp = waypoints[0]
        v = wp.dx / dt  # 线速度 m/s
        w = wp.dtheta / dt  # 角速度 rad/s

        # 限制速度范围
        v = max(-1.0, min(1.5, v))
        w = max(-1.0, min(1.0, w))

        return v, w
