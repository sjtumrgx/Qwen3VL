"""
VLN 视频测试脚本 - 支持中途切换导航任务

功能：
1. 从网络下载视频或使用本地视频
2. 逐帧发送到 VLN API
3. 支持中途切换导航任务（按帧号或时间）
4. 输出带可视化标注的结果视频

使用示例：
    # 基本用法
    python test_vln_switch.py --video https://example.com/video.mp4 \
        --instruction "向前走，找到门"

    # 中途切换任务
    python test_vln_switch.py --video input.mp4 \
        --instruction "向前走" \
        --switch 100 "左转，进入房间" \
        --switch 200 "找到桌子"

    # 使用本地视频
    python test_vln_switch.py --video /path/to/video.mp4 \
        --instruction "探索环境"
"""

import argparse
import base64
import logging
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import cv2
import numpy as np
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class TaskSwitch:
    """任务切换点"""
    frame_number: int
    instruction: str
    task_id: Optional[str] = None


class VLNVideoTester:
    """VLN 视频测试器 - 支持任务切换"""

    def __init__(
        self,
        base_url: str = "http://localhost:20000",
        use_sync: bool = False,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.use_sync = use_sync
        self.current_task_id: Optional[str] = None
        self.tasks: Dict[str, str] = {}  # task_id -> instruction

    def check_health(self) -> bool:
        """检查服务健康状态"""
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=10)
            resp.raise_for_status()
            data = resp.json()
            logger.info(f"服务状态: {data.get('status')}, GPU: {data.get('gpu_count')}")
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
                    "output_fps": 5.0,
                    "keyframe_threshold": 0.2,
                    "enable_visualization": True,
                },
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        task_id = data["task_id"]
        self.tasks[task_id] = instruction
        logger.info(f"创建任务: {task_id} - {instruction}")
        return task_id

    def switch_task(self, instruction: str) -> str:
        """切换到新任务"""
        # 停止当前任务
        if self.current_task_id:
            self.stop_task(self.current_task_id)

        # 创建新任务
        self.current_task_id = self.create_task(instruction)
        return self.current_task_id

    def stop_task(self, task_id: str) -> None:
        """停止任务"""
        try:
            requests.post(f"{self.base_url}/vln/task/{task_id}/stop", timeout=10)
            logger.info(f"停止任务: {task_id}")
        except Exception as e:
            logger.warning(f"停止任务失败: {e}")

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: float,
    ) -> Tuple[dict, Optional[np.ndarray]]:
        """
        处理单帧

        Returns:
            (API 响应, 可视化帧)
        """
        # 编码帧
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_base64 = base64.b64encode(buffer).decode()

        # 选择端点
        endpoint = "/vln/frame/sync" if self.use_sync else "/vln/frame"

        resp = requests.post(
            f"{self.base_url}{endpoint}",
            json={
                "task_id": self.current_task_id,
                "frame": frame_base64,
                "timestamp": timestamp,
            },
            timeout=120 if self.use_sync else 10,
        )
        resp.raise_for_status()
        result = resp.json()

        # 解码可视化帧
        vis_frame = None
        if result.get("visualized_frame"):
            vis_data = base64.b64decode(result["visualized_frame"])
            vis_frame = cv2.imdecode(
                np.frombuffer(vis_data, np.uint8),
                cv2.IMREAD_COLOR,
            )

        return result, vis_frame

    def download_video(self, url: str, output_path: str) -> bool:
        """下载视频"""
        logger.info(f"下载视频: {url}")
        try:
            resp = requests.get(url, stream=True, timeout=60)
            resp.raise_for_status()

            total_size = int(resp.headers.get("content-length", 0))
            downloaded = 0

            with open(output_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = downloaded / total_size * 100
                        print(f"\r下载进度: {progress:.1f}%", end="", flush=True)

            print()  # 换行
            logger.info(f"下载完成: {output_path}")
            return True

        except Exception as e:
            logger.error(f"下载失败: {e}")
            return False

    def process_video(
        self,
        video_path: str,
        output_path: str,
        initial_instruction: str,
        switches: List[TaskSwitch],
        max_frames: Optional[int] = None,
        skip_frames: int = 0,
        show_preview: bool = False,
    ) -> dict:
        """
        处理视频

        Args:
            video_path: 输入视频路径
            output_path: 输出视频路径
            initial_instruction: 初始导航指令
            switches: 任务切换点列表
            max_frames: 最大处理帧数
            skip_frames: 跳帧数
            show_preview: 是否显示预览窗口

        Returns:
            处理统计
        """
        # 打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")

        # 视频信息
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(f"视频信息: {total_frames} 帧, {fps:.1f} FPS, {width}x{height}")

        # 排序切换点
        switches = sorted(switches, key=lambda x: x.frame_number)
        switch_idx = 0

        # 创建输出视频
        effective_fps = fps / (skip_frames + 1) if skip_frames > 0 else fps
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, effective_fps, (width, height))

        # 创建初始任务
        self.current_task_id = self.create_task(initial_instruction)

        # 统计
        stats = {
            "total_frames": 0,
            "processed_frames": 0,
            "task_switches": 0,
            "tasks": [{"instruction": initial_instruction, "start_frame": 0}],
        }

        try:
            frame_idx = 0
            start_time = time.time()

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                stats["total_frames"] += 1

                # 跳帧
                if skip_frames > 0 and frame_idx % (skip_frames + 1) != 0:
                    frame_idx += 1
                    continue

                # 最大帧数限制
                if max_frames and stats["processed_frames"] >= max_frames:
                    break

                # 检查是否需要切换任务
                while switch_idx < len(switches) and frame_idx >= switches[switch_idx].frame_number:
                    switch = switches[switch_idx]
                    logger.info(f"帧 {frame_idx}: 切换任务 -> {switch.instruction}")
                    self.switch_task(switch.instruction)
                    stats["task_switches"] += 1
                    stats["tasks"].append({
                        "instruction": switch.instruction,
                        "start_frame": frame_idx,
                    })
                    switch_idx += 1

                # 处理帧
                timestamp = frame_idx / fps
                try:
                    result, vis_frame = self.process_frame(frame, timestamp)

                    # 在帧上添加任务信息
                    if vis_frame is not None:
                        vis_frame = self._add_task_overlay(
                            vis_frame,
                            self.tasks.get(self.current_task_id, ""),
                            frame_idx,
                            total_frames,
                        )
                        out.write(vis_frame)

                        # 预览
                        if show_preview:
                            cv2.imshow("VLN Preview", vis_frame)
                            key = cv2.waitKey(1) & 0xFF
                            if key == ord('q'):
                                logger.info("用户中断")
                                break
                            elif key == ord('s'):
                                # 按 's' 手动切换任务
                                new_instruction = input("输入新指令: ")
                                if new_instruction:
                                    self.switch_task(new_instruction)
                                    stats["task_switches"] += 1
                                    stats["tasks"].append({
                                        "instruction": new_instruction,
                                        "start_frame": frame_idx,
                                    })
                    else:
                        out.write(frame)

                    stats["processed_frames"] += 1

                    # 进度
                    if stats["processed_frames"] % 30 == 0:
                        elapsed = time.time() - start_time
                        proc_fps = stats["processed_frames"] / elapsed
                        logger.info(
                            f"进度: {frame_idx}/{total_frames} "
                            f"({proc_fps:.1f} FPS), "
                            f"任务: {self.tasks.get(self.current_task_id, '')[:30]}"
                        )

                except Exception as e:
                    logger.warning(f"处理帧 {frame_idx} 失败: {e}")
                    out.write(frame)

                frame_idx += 1

        finally:
            cap.release()
            out.release()
            if show_preview:
                cv2.destroyAllWindows()

            # 停止所有任务
            for task_id in self.tasks:
                self.stop_task(task_id)

        # 统计
        total_time = time.time() - start_time
        stats["total_time"] = total_time
        stats["avg_fps"] = stats["processed_frames"] / total_time if total_time > 0 else 0

        logger.info(f"处理完成: {output_path}")
        logger.info(f"总帧数: {stats['processed_frames']}, 任务切换: {stats['task_switches']}")

        return stats

    def _add_task_overlay(
        self,
        frame: np.ndarray,
        instruction: str,
        frame_idx: int,
        total_frames: int,
    ) -> np.ndarray:
        """在帧上添加任务信息覆盖层"""
        h, w = frame.shape[:2]

        # 顶部任务栏
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 40), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        # 任务文字
        task_text = f"Task: {instruction[:50]}..." if len(instruction) > 50 else f"Task: {instruction}"
        cv2.putText(
            frame, task_text, (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA
        )

        # 帧号
        frame_text = f"Frame: {frame_idx}/{total_frames}"
        cv2.putText(
            frame, frame_text, (w - 180, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA
        )

        return frame


def parse_switches(switch_args: List[str]) -> List[TaskSwitch]:
    """解析切换参数"""
    switches = []
    i = 0
    while i < len(switch_args):
        if i + 1 < len(switch_args):
            frame_num = int(switch_args[i])
            instruction = switch_args[i + 1]
            switches.append(TaskSwitch(frame_number=frame_num, instruction=instruction))
            i += 2
        else:
            break
    return switches


def main():
    parser = argparse.ArgumentParser(
        description="VLN 视频测试 - 支持中途切换导航任务",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本用法
  python test_vln_switch.py --video video.mp4 --instruction "向前走"

  # 中途切换任务（在第100帧和第200帧切换）
  python test_vln_switch.py --video video.mp4 \\
      --instruction "向前走" \\
      --switch 100 "左转" \\
      --switch 200 "找到门"

  # 使用网络视频
  python test_vln_switch.py --video https://example.com/video.mp4 \\
      --instruction "探索环境"

  # 显示预览窗口（按 'q' 退出，按 's' 手动切换任务）
  python test_vln_switch.py --video video.mp4 --instruction "向前走" --preview
        """,
    )

    parser.add_argument(
        "--video", "-v",
        required=True,
        help="视频路径或 URL",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="输出视频路径（默认: input_vln.mp4）",
    )
    parser.add_argument(
        "--instruction", "-i",
        default="向前走，探索周围环境",
        help="初始导航指令",
    )
    parser.add_argument(
        "--switch",
        nargs=2,
        action="append",
        metavar=("FRAME", "INSTRUCTION"),
        help="任务切换点: --switch <帧号> <新指令>",
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
        default=2,
        help="跳帧数（默认: 2，即每3帧处理1帧）",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="使用同步模式（等待每帧推理完成）",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="显示预览窗口",
    )

    args = parser.parse_args()

    # 处理视频路径
    video_path = args.video
    is_url = video_path.startswith("http://") or video_path.startswith("https://")

    if is_url:
        # 下载视频到临时文件
        parsed = urlparse(video_path)
        ext = Path(parsed.path).suffix or ".mp4"
        temp_file = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
        temp_path = temp_file.name
        temp_file.close()

        tester = VLNVideoTester(base_url=args.url, use_sync=args.sync)
        if not tester.download_video(video_path, temp_path):
            sys.exit(1)
        video_path = temp_path
    else:
        if not os.path.exists(video_path):
            logger.error(f"视频文件不存在: {video_path}")
            sys.exit(1)
        tester = VLNVideoTester(base_url=args.url, use_sync=args.sync)

    # 设置输出路径
    if args.output is None:
        input_path = Path(video_path)
        args.output = str(input_path.parent / f"{input_path.stem}_vln.mp4")

    # 解析切换点
    switches = []
    if args.switch:
        for frame_str, instruction in args.switch:
            switches.append(TaskSwitch(
                frame_number=int(frame_str),
                instruction=instruction,
            ))

    # 检查服务
    if not tester.check_health():
        logger.error("服务不可用")
        sys.exit(1)

    # 处理视频
    try:
        stats = tester.process_video(
            video_path=video_path,
            output_path=args.output,
            initial_instruction=args.instruction,
            switches=switches,
            max_frames=args.max_frames,
            skip_frames=args.skip_frames,
            show_preview=args.preview,
        )

        print(f"\n{'='*50}")
        print(f"输出视频: {args.output}")
        print(f"处理帧数: {stats['processed_frames']}")
        print(f"任务切换: {stats['task_switches']} 次")
        print(f"平均 FPS: {stats['avg_fps']:.2f}")
        print(f"\n任务历史:")
        for i, task in enumerate(stats['tasks']):
            print(f"  {i+1}. 帧 {task['start_frame']}: {task['instruction']}")
        print(f"{'='*50}")

    except KeyboardInterrupt:
        logger.info("用户中断")
    except Exception as e:
        logger.error(f"处理失败: {e}")
        sys.exit(1)
    finally:
        # 清理临时文件
        if is_url and os.path.exists(temp_path):
            os.unlink(temp_path)


if __name__ == "__main__":
    main()
