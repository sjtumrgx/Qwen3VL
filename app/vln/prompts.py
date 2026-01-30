"""
VLN Prompt 模板

构建导航专用的 prompt，包含历史摘要、当前帧、导航指令
"""

from typing import List, Optional

from . import FrameContext


# 系统提示词
SYSTEM_PROMPT = """You are a vision-language navigation (VLN) agent for a quadruped robot. Your task is to analyze the current visual observation and provide navigation commands to reach the goal described in the instruction.

You must output in the following JSON format:
{
    "environment": "Brief description of the current environment (in Chinese)",
    "action": "Current action description (in Chinese)",
    "waypoints": [
        {"dx": <forward distance in meters>, "dy": <lateral distance in meters, positive=left>, "dtheta": <rotation in radians, positive=counterclockwise>}
    ],
    "reasoning": "Brief reasoning for the navigation decision (in Chinese)",
    "progress": <estimated progress towards goal, 0.0-1.0>,
    "reached_goal": <true if goal is reached, false otherwise>
}

Navigation constraints:
- dx: forward distance, typically 0.0 to 1.0 meters per step
- dy: lateral distance, typically -0.3 to 0.3 meters per step
- dtheta: rotation angle, typically -0.5 to 0.5 radians per step
- Output 1-5 waypoints per inference
- If obstacle detected, output smaller steps or rotation to avoid
- If goal reached, set reached_goal=true and output empty waypoints"""


def build_navigation_prompt(
    instruction: str,
    history_summary: str = "",
    recent_actions: Optional[List[str]] = None,
    current_environment: str = "",
) -> str:
    """
    构建导航 prompt（文本部分）

    Args:
        instruction: 导航指令
        history_summary: 历史经历摘要
        recent_actions: 最近的动作列表
        current_environment: 当前环境描述

    Returns:
        构建的 prompt 文本
    """
    parts = []

    # 导航指令
    parts.append(f"## Navigation Instruction\n{instruction}")

    # 历史摘要
    if history_summary:
        parts.append(f"## Navigation History Summary\n{history_summary}")

    # 最近动作
    if recent_actions:
        actions_text = "\n".join(f"- {a}" for a in recent_actions[-5:])
        parts.append(f"## Recent Actions\n{actions_text}")

    # 当前环境
    if current_environment:
        parts.append(f"## Previous Environment\n{current_environment}")

    # 任务提示
    parts.append(
        "## Task\n"
        "Analyze the current image(s) and provide navigation waypoints to reach the goal. "
        "Output in the specified JSON format."
    )

    return "\n\n".join(parts)


def build_messages_with_images(
    prompt: str,
    image_base64_list: List[str],
    include_system: bool = True,
) -> List[dict]:
    """
    构建包含图像的消息列表

    Args:
        prompt: 文本 prompt
        image_base64_list: Base64 编码的图像列表
        include_system: 是否包含系统提示

    Returns:
        OpenAI 格式的消息列表
    """
    messages = []

    # 系统消息
    if include_system:
        messages.append({
            "role": "system",
            "content": SYSTEM_PROMPT,
        })

    # 用户消息（图像 + 文本）
    content = []

    # 添加图像
    for img_b64 in image_base64_list:
        # 确保是 data URL 格式
        if not img_b64.startswith("data:"):
            img_b64 = f"data:image/jpeg;base64,{img_b64}"
        content.append({
            "type": "image_url",
            "image_url": {"url": img_b64},
        })

    # 添加文本
    content.append({
        "type": "text",
        "text": prompt,
    })

    messages.append({
        "role": "user",
        "content": content,
    })

    return messages


def build_history_context_prompt(
    keyframes: List[FrameContext],
    max_frames: int = 3,
) -> str:
    """
    从关键帧构建历史上下文描述

    Args:
        keyframes: 关键帧列表
        max_frames: 最大使用帧数

    Returns:
        历史上下文描述文本
    """
    if not keyframes:
        return ""

    recent = keyframes[-max_frames:]
    descriptions = []

    for i, frame in enumerate(recent):
        desc_parts = []
        if frame.environment_desc:
            desc_parts.append(f"环境: {frame.environment_desc}")
        if frame.action_desc:
            desc_parts.append(f"动作: {frame.action_desc}")
        if desc_parts:
            descriptions.append(f"[帧{i+1}] " + ", ".join(desc_parts))

    return "\n".join(descriptions) if descriptions else ""


def build_summary_prompt(
    instruction: str,
    history_descriptions: List[str],
    current_progress: float,
) -> str:
    """
    构建经历摘要生成的 prompt

    Args:
        instruction: 原始导航指令
        history_descriptions: 历史环境描述列表
        current_progress: 当前进度

    Returns:
        摘要生成 prompt
    """
    history_text = "\n".join(f"- {d}" for d in history_descriptions[-10:])

    return f"""请根据以下导航历史生成简洁的经历摘要（不超过100字）：

导航目标：{instruction}
当前进度：{current_progress:.0%}

历史记录：
{history_text}

请用中文输出摘要，描述机器人经过的主要区域和遇到的情况。"""
