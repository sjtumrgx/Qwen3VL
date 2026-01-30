"""
VLN API 快速测试脚本

用于验证 VLN API 端点是否正常工作
"""

import argparse
import base64
import json
import sys
from pathlib import Path

import requests


def test_health(base_url: str) -> bool:
    """测试健康检查"""
    print("=" * 50)
    print("测试: 健康检查")
    print("=" * 50)

    try:
        resp = requests.get(f"{base_url}/health", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        print(f"状态: {data.get('status')}")
        print(f"模型: {data.get('model')}")
        print(f"GPU: {data.get('gpu_count')}")
        return data.get("status") == "healthy"
    except Exception as e:
        print(f"错误: {e}")
        return False


def test_create_task(base_url: str, instruction: str) -> str:
    """测试创建任务"""
    print("\n" + "=" * 50)
    print("测试: 创建 VLN 任务")
    print("=" * 50)

    resp = requests.post(
        f"{base_url}/vln/task/create",
        json={
            "instruction": instruction,
            "config": {
                "output_fps": 5.0,
                "enable_visualization": True,
            },
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    print(f"任务 ID: {data.get('task_id')}")
    print(f"状态: {data.get('status')}")
    print(f"指令: {data.get('instruction')}")
    return data.get("task_id")


def test_process_frame(base_url: str, task_id: str, image_path: str) -> dict:
    """测试处理帧"""
    print("\n" + "=" * 50)
    print("测试: 处理帧")
    print("=" * 50)

    # 读取图像
    with open(image_path, "rb") as f:
        image_data = f.read()
    image_base64 = base64.b64encode(image_data).decode("utf-8")

    resp = requests.post(
        f"{base_url}/vln/frame",
        json={
            "task_id": task_id,
            "frame": image_base64,
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()

    print(f"帧 ID: {data.get('frame_id')}")
    print(f"环境: {data.get('environment')}")
    print(f"动作: {data.get('action')}")
    print(f"航点数: {len(data.get('waypoints', []))}")
    print(f"线速度: {data.get('linear_vel'):.3f} m/s")
    print(f"角速度: {data.get('angular_vel'):.3f} rad/s")
    print(f"推理时间: {data.get('inference_time'):.3f}s")

    # 保存可视化帧
    vis_frame = data.get("visualized_frame")
    if vis_frame:
        output_path = Path(image_path).parent / "vln_output.jpg"
        with open(output_path, "wb") as f:
            f.write(base64.b64decode(vis_frame))
        print(f"可视化帧已保存: {output_path}")

    return data


def test_task_status(base_url: str, task_id: str) -> dict:
    """测试任务状态"""
    print("\n" + "=" * 50)
    print("测试: 任务状态")
    print("=" * 50)

    resp = requests.get(f"{base_url}/vln/task/{task_id}", timeout=10)
    resp.raise_for_status()
    data = resp.json()

    print(f"任务 ID: {data.get('task_id')}")
    print(f"状态: {data.get('status')}")
    print(f"帧数: {data.get('frame_count')}")
    print(f"当前环境: {data.get('current_environment')}")
    print(f"当前动作: {data.get('current_action')}")

    return data


def test_pipeline_stats(base_url: str) -> dict:
    """测试流水线统计"""
    print("\n" + "=" * 50)
    print("测试: 流水线统计")
    print("=" * 50)

    resp = requests.get(f"{base_url}/vln/pipeline/stats", timeout=10)
    resp.raise_for_status()
    data = resp.json()

    print(f"总请求: {data.get('total_requests')}")
    print(f"已处理: {data.get('processed')}")
    print(f"已丢弃: {data.get('dropped')}")
    print(f"错误数: {data.get('errors')}")
    print(f"平均推理时间: {data.get('avg_inference_time'):.3f}s")
    print(f"队列大小: {data.get('queue_size')}")
    print(f"运行中: {data.get('running')}")

    return data


def test_stop_task(base_url: str, task_id: str) -> None:
    """测试停止任务"""
    print("\n" + "=" * 50)
    print("测试: 停止任务")
    print("=" * 50)

    resp = requests.post(f"{base_url}/vln/task/{task_id}/stop", timeout=10)
    resp.raise_for_status()
    data = resp.json()
    print(f"任务 {task_id} 已停止")


def main():
    parser = argparse.ArgumentParser(description="VLN API 快速测试")
    parser.add_argument(
        "--url",
        default="http://localhost:20000",
        help="API 基础 URL",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="测试图像路径",
    )
    parser.add_argument(
        "--instruction",
        default="向前走，探索周围环境",
        help="导航指令",
    )

    args = parser.parse_args()

    print(f"API URL: {args.url}")
    print(f"测试图像: {args.image}")
    print(f"导航指令: {args.instruction}")

    # 1. 健康检查
    if not test_health(args.url):
        print("\n服务不可用，退出")
        sys.exit(1)

    # 2. 创建任务
    task_id = test_create_task(args.url, args.instruction)

    # 3. 处理帧（如果提供了图像）
    if args.image:
        test_process_frame(args.url, task_id, args.image)

    # 4. 查询任务状态
    test_task_status(args.url, task_id)

    # 5. 流水线统计
    test_pipeline_stats(args.url)

    # 6. 停止任务
    test_stop_task(args.url, task_id)

    print("\n" + "=" * 50)
    print("所有测试完成！")
    print("=" * 50)


if __name__ == "__main__":
    main()
