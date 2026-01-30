"""
VLN 视频测试脚本

读取视频文件，调用 VLN API，输出带可视化标注的结果视频
"""

import argparse
import base64
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class VLNVideoTester:
    """VLN 视频测试器"""

    def __init__(
        self,
        base_url: str = "http://localhost:20000",
        output_fps: float = 5.0,
    ) -> None:
        """
        初始化测试器

        Args:
            base_url: API 基础 URL
            output_fps: 输出视频帧率
        """
        self.base_url = base_url.rstrip("/")
        self.output_fps = output_fps
        self.task_id: Optional[str] = None

    def check_health(self) -> bool:
        """检查服务健康状态"""
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=10)
            resp.raise_for_status()
            data = resp.json()
            logger.info(f"服务状态: {data}")
            return data.get("status") == "healthy"
        except Exception as e:
            logger.error(f"健康检查失败: {e}")
            return False

    def create_task(self, instruction: str) -> str:
        """创建导航任务"""
        resp = requests.post(
            f"{self.base_url}/vln/task/create",
            json={
                "instruction": instruction,
                "config": {
                    "output_fps": self.output_fps,
                    "history_frames": 10,
                    "keyframe_threshold": 0.3,
                    "enable_visualization": True,
                },
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        self.task_id = data["task_id"]
        logger.info(f"创建任务: {self.task_id}")
        return self.task_id

    def process_frame(self, frame: bytes, timestamp: float) -> dict:
        """处理单帧"""
        frame_base64 = base64.b64encode(frame).decode("utf-8")

        resp = requests.post(
            f"{self.base_url}/vln/frame",
            json={
                "task_id": self.task_id,
                "frame": frame_base64,
                "timestamp": timestamp,
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def stop_task(self) -> None:
        """停止任务"""
        if self.task_id:
            try:
                requests.post(
                    f"{self.base_url}/vln/task/{self.task_id}/stop",
                    timeout=10,
                )
                logger.info(f"停止任务: {self.task_id}")
            except Exception as e:
                logger.warning(f"停止任务失败: {e}")

    def process_video(
        self,
        input_path: str,
        output_path: str,
        instruction: str,
        max_frames: Optional[int] = None,
        skip_frames: int = 0,
    ) -> dict:
        """
        处理视频文件

        Args:
            input_path: 输入视频路径
            output_path: 输出视频路径
            instruction: 导航指令
            max_frames: 最大处理帧数（None 表示全部）
            skip_frames: 跳过的帧数（用于降低帧率）

        Returns:
            处理统计信息
        """
        # 打开输入视频
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {input_path}")

        # 获取视频信息
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        input_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(f"输入视频: {total_frames} 帧, {input_fps:.1f} FPS, {width}x{height}")

        # 计算实际处理的帧率
        if skip_frames > 0:
            effective_fps = input_fps / (skip_frames + 1)
        else:
            effective_fps = input_fps

        # 创建输出视频
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, effective_fps, (width, height))

        # 创建任务
        self.create_task(instruction)

        # 统计
        stats = {
            "total_frames": 0,
            "processed_frames": 0,
            "keyframes": 0,
            "total_inference_time": 0.0,
            "waypoints_generated": 0,
            "reached_goal": False,
        }

        try:
            frame_idx = 0
            start_time = time.time()

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                stats["total_frames"] += 1

                # 跳帧处理
                if skip_frames > 0 and frame_idx % (skip_frames + 1) != 0:
                    frame_idx += 1
                    continue

                # 限制最大帧数
                if max_frames and stats["processed_frames"] >= max_frames:
                    break

                # 编码帧
                _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                timestamp = frame_idx / input_fps

                # 调用 API
                try:
                    result = self.process_frame(buffer.tobytes(), timestamp)

                    stats["processed_frames"] += 1
                    stats["total_inference_time"] += result.get("inference_time", 0)
                    stats["waypoints_generated"] += len(result.get("waypoints", []))

                    if result.get("inference_time", 0) > 0:
                        stats["keyframes"] += 1

                    if result.get("reached_goal"):
                        stats["reached_goal"] = True
                        logger.info("到达目标！")

                    # 获取可视化帧
                    vis_frame_b64 = result.get("visualized_frame")
                    if vis_frame_b64:
                        vis_data = base64.b64decode(vis_frame_b64)
                        vis_array = cv2.imdecode(
                            __import__("numpy").frombuffer(vis_data, __import__("numpy").uint8),
                            cv2.IMREAD_COLOR,
                        )
                        out.write(vis_array)
                    else:
                        out.write(frame)

                    # 打印进度
                    if stats["processed_frames"] % 10 == 0:
                        elapsed = time.time() - start_time
                        fps = stats["processed_frames"] / elapsed
                        logger.info(
                            f"进度: {stats['processed_frames']}/{total_frames} "
                            f"({fps:.1f} FPS), "
                            f"环境: {result.get('environment', '')[:30]}..."
                        )

                except Exception as e:
                    logger.warning(f"处理帧 {frame_idx} 失败: {e}")
                    out.write(frame)

                frame_idx += 1

        finally:
            cap.release()
            out.release()
            self.stop_task()

        # 计算统计
        total_time = time.time() - start_time
        stats["total_time"] = total_time
        stats["avg_fps"] = stats["processed_frames"] / total_time if total_time > 0 else 0
        stats["avg_inference_time"] = (
            stats["total_inference_time"] / stats["keyframes"]
            if stats["keyframes"] > 0
            else 0
        )

        logger.info(f"处理完成: {output_path}")
        logger.info(f"统计: {json.dumps(stats, indent=2, ensure_ascii=False)}")

        return stats


def main():
    parser = argparse.ArgumentParser(description="VLN 视频测试")
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="输入视频路径",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="输出视频路径（默认: input_vln.mp4）",
    )
    parser.add_argument(
        "--instruction",
        default="Navigate forward and explore the environment",
        help="导航指令",
    )
    parser.add_argument(
        "--url",
        default="http://localhost:20000",
        help="API 基础 URL",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="最大处理帧数",
    )
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=0,
        help="跳过的帧数（用于降低帧率）",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=5.0,
        help="输出帧率",
    )

    args = parser.parse_args()

    # 设置输出路径
    if args.output is None:
        input_path = Path(args.input)
        args.output = str(input_path.parent / f"{input_path.stem}_vln.mp4")

    # 创建测试器
    tester = VLNVideoTester(base_url=args.url, output_fps=args.fps)

    # 检查服务
    if not tester.check_health():
        logger.error("服务不可用")
        sys.exit(1)

    # 处理视频
    try:
        stats = tester.process_video(
            input_path=args.input,
            output_path=args.output,
            instruction=args.instruction,
            max_frames=args.max_frames,
            skip_frames=args.skip_frames,
        )
        print(f"\n输出视频: {args.output}")
        print(f"处理帧数: {stats['processed_frames']}")
        print(f"关键帧数: {stats['keyframes']}")
        print(f"平均 FPS: {stats['avg_fps']:.2f}")
        print(f"平均推理时间: {stats['avg_inference_time']:.3f}s")

    except Exception as e:
        logger.error(f"处理失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
