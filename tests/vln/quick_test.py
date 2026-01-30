"""VLN API 测试 - 同步模式"""
import base64
import requests
import numpy as np
import cv2
import time

# 创建一个模拟的走廊图像
def create_test_image(variation=0):
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:, :] = (100 + variation * 10, 100, 100)
    cv2.line(img, (0, 480), (320, 200), (50, 50, 50), 3)
    cv2.line(img, (640, 480), (320, 200), (50, 50, 50), 3)
    cv2.rectangle(img, (280, 180), (360, 220), (80, 80, 80), -1)
    # 添加一些文字模拟场景
    cv2.putText(img, "Corridor", (250, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
    _, buffer = cv2.imencode(".jpg", img)
    return base64.b64encode(buffer).decode()

print("Creating test images...")
img_base64 = create_test_image(0)

# 创建任务
print("Creating task...")
resp = requests.post(
    "http://localhost:8000/vln/task/create",
    json={
        "instruction": "向前走，探索周围环境",
        "config": {
            "keyframe_threshold": 0.01,  # 降低阈值，更容易触发
            "enable_visualization": True
        }
    }
)
task_id = resp.json()["task_id"]
print(f"Task ID: {task_id}")

# 使用同步端点发送帧（等待推理完成）
img = create_test_image(0)
print("Sending frame (sync mode)...")

resp = requests.post(
    "http://localhost:8000/vln/frame/sync",  # 使用同步端点
    json={
        "task_id": task_id,
        "frame": img,
        "timestamp": time.time()
    },
    timeout=120
)

result = resp.json()
print(f"Environment: {result.get('environment', 'N/A')[:80]}")
print(f"Action: {result.get('action', 'N/A')}")
print(f"Waypoints: {len(result.get('waypoints', []))}")
print(f"Inference time: {result.get('inference_time', 0):.3f}s")

if result.get("inference_time", 0) > 0:
    print(f"Linear vel: {result.get('linear_vel'):.3f} m/s")
    print(f"Angular vel: {result.get('angular_vel'):.3f} rad/s")

    if result.get("visualized_frame"):
        vis_data = base64.b64decode(result["visualized_frame"])
        with open("/workspace/vln_test_output.jpg", "wb") as f:
            f.write(vis_data)
        print("Saved: /workspace/vln_test_output.jpg")

# 测试非阻塞模式
print("\n--- Testing non-blocking mode ---")
for i in range(3):
    img = create_test_image(i * 5)
    start = time.time()
    resp = requests.post(
        "http://localhost:8000/vln/frame",  # 非阻塞端点
        json={"task_id": task_id, "frame": img, "timestamp": time.time()},
        timeout=10
    )
    elapsed = time.time() - start
    result = resp.json()
    print(f"Frame {i+1}: response in {elapsed*1000:.1f}ms, action={result.get('action', 'N/A')[:30]}")
    time.sleep(0.2)

# 检查流水线统计
resp = requests.get("http://localhost:8000/vln/pipeline/stats")
stats = resp.json()
print(f"\nPipeline stats:")
print(f"  Total requests: {stats.get('total_requests')}")
print(f"  Processed: {stats.get('processed')}")
print(f"  Avg inference time: {stats.get('avg_inference_time'):.3f}s")
