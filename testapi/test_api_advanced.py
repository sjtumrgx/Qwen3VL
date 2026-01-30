#!/usr/bin/env python3
"""
Qwen3-VL API 增强测试脚本

测试自定义 FastAPI 推理服务的所有功能
"""

import argparse
import base64
import json
import sys
from pathlib import Path

import requests


class Qwen3VLAdvancedTester:
    """Qwen3-VL API 增强测试器"""

    def __init__(self, base_url: str = "http://localhost:20000"):
        self.base_url = base_url.rstrip("/")
        self.headers = {"Content-Type": "application/json"}

    def test_gpu_status(self) -> bool:
        """测试 GPU 状态端点"""
        print("\n" + "=" * 50)
        print("测试: GPU 状态 (/gpu/status)")
        print("=" * 50)

        try:
            response = requests.get(f"{self.base_url}/gpu/status", timeout=5)
            print(f"状态码: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                print(f"GPU 可用: {data.get('available')}")
                print(f"GPU 数量: {data.get('gpu_count')}")
                for gpu in data.get("gpus", []):
                    print(
                        f"  GPU {gpu['index']}: {gpu['name']} "
                        f"({gpu['memory_allocated_gb']:.2f}/{gpu['memory_total_gb']:.2f} GB)"
                    )
                print("✓ GPU 状态获取成功")
                return True
            else:
                print(f"✗ 获取失败: {response.text}")
                return False
        except Exception as e:
            print(f"✗ 请求失败: {e}")
            return False

    def test_simple_infer(self) -> bool:
        """测试简单推理接口"""
        print("\n" + "=" * 50)
        print("测试: 简单推理 (/infer)")
        print("=" * 50)

        payload = {
            "prompt": "请用一句话介绍一下 LMDeploy 推理框架。",
            "max_tokens": 100,
        }

        try:
            response = requests.post(
                f"{self.base_url}/infer",
                headers=self.headers,
                json=payload,
                timeout=60,
            )
            print(f"状态码: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                print(f"\n回复: {data['text']}")
                print(f"Token: {data['total_tokens']}")
                print("✓ 简单推理成功")
                return True
            else:
                print(f"✗ 失败: {response.text}")
                return False
        except Exception as e:
            print(f"✗ 请求失败: {e}")
            return False

    def test_analyze_upload(self, image_path: str = None) -> bool:
        """测试图像分析（文件上传）"""
        print("\n" + "=" * 50)
        print("测试: 图像分析-文件上传 (/analyze/upload)")
        print("=" * 50)

        if image_path is None:
            print("未提供测试图像，跳过此测试")
            return True

        try:
            with open(image_path, "rb") as f:
                files = {"image": (Path(image_path).name, f, "image/jpeg")}
                data = {"instruction": "请分析这张图片中的主要内容。", "max_tokens": 300}

                print(f"上传图像: {image_path}")
                response = requests.post(
                    f"{self.base_url}/analyze/upload",
                    files=files,
                    data=data,
                    timeout=120,
                )

            print(f"状态码: {response.status_code}")

            if response.status_code == 200:
                result = response.json()
                print(f"\n分析结果:\n{result['analysis']}")
                print(f"Token: {result['tokens']['total']}")
                print("✓ 图像分析成功")
                return True
            else:
                print(f"✗ 失败: {response.text}")
                return False
        except Exception as e:
            print(f"✗ 请求失败: {e}")
            return False

    def run_tests(self, image_path: str = None):
        """运行增强测试"""
        print("\n" + "=" * 50)
        print("Qwen3-VL 增强测试套件")
        print(f"服务地址: {self.base_url}")
        print("=" * 50)

        results = {
            "GPU 状态": self.test_gpu_status(),
            "简单推理": self.test_simple_infer(),
            "图像分析": self.test_analyze_upload(image_path),
        }

        print("\n" + "=" * 50)
        print("测试总结")
        print("=" * 50)

        passed = sum(results.values())
        total = len(results)

        for test_name, result in results.items():
            status = "✓ 通过" if result else "✗ 失败"
            print(f"{test_name}: {status}")

        print(f"\n总计: {passed}/{total} 测试通过")
        return 0 if passed == total else 1


def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL API 增强测试脚本")
    parser.add_argument(
        "--url",
        default="http://localhost:20000",
        help="API 服务地址",
    )
    parser.add_argument("--image", help="测试图像路径")

    args = parser.parse_args()

    tester = Qwen3VLAdvancedTester(base_url=args.url)
    exit_code = tester.run_tests(image_path=args.image)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
