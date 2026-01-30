#!/usr/bin/env python3
"""
Qwen3-VL API 测试脚本

测试 vLLM OpenAI 兼容 API 的功能，包括：
1. 健康检查
2. 模型列表
3. 文本推理
4. 图像+文本推理
"""

import base64
import io
import json
import sys
from pathlib import Path

import requests


def _ensure_utf8_stdio() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue

        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
            continue
        except Exception:
            pass

        try:
            buffer = getattr(stream, "buffer", None)
            if buffer is not None:
                wrapped = io.TextIOWrapper(
                    buffer,
                    encoding="utf-8",
                    errors="replace",
                    line_buffering=True,
                )
                setattr(sys, stream_name, wrapped)
        except Exception:
            pass


_ensure_utf8_stdio()


class Qwen3VLTester:
    """Qwen3-VL API 测试器"""

    def __init__(self, base_url: str = "http://localhost:20000"):
        self.base_url = base_url.rstrip("/")
        self.headers = {"Content-Type": "application/json"}

    def test_health(self) -> bool:
        """测试健康检查端点"""
        print("\n" + "=" * 50)
        print("测试 1: 健康检查")
        print("=" * 50)

        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            print(f"状态码: {response.status_code}")
            print(f"响应: {response.text}")

            if response.status_code == 200:
                print("✓ 健康检查通过")
                return True
            else:
                print("✗ 健康检查失败")
                return False
        except Exception as e:
            print(f"✗ 请求失败: {e}")
            return False

    def test_models(self) -> bool:
        """测试模型列表端点"""
        print("\n" + "=" * 50)
        print("测试 2: 模型列表")
        print("=" * 50)

        try:
            response = requests.get(f"{self.base_url}/v1/models", timeout=5)
            print(f"状态码: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                print(f"可用模型数量: {len(data.get('data', []))}")
                for model in data.get("data", []):
                    print(f"  - {model.get('id')}")
                print("✓ 模型列表获取成功")
                return True
            else:
                print(f"✗ 获取失败: {response.text}")
                return False
        except Exception as e:
            print(f"✗ 请求失败: {e}")
            return False

    def test_text_inference(self) -> bool:
        """测试文本推理"""
        print("\n" + "=" * 50)
        print("测试 3: 文本推理")
        print("=" * 50)

        payload = {
            "model": "Qwen3-VL-32B-Instruct",
            "messages": [
                {"role": "user", "content": "你好，请用一句话介绍一下你自己。"}
            ],
            "max_tokens": 100,
            "temperature": 0.7,
        }

        print(f"请求: {json.dumps(payload, ensure_ascii=False, indent=2)}")

        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=30,
            )
            print(f"\n状态码: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                print(f"\n模型回复:\n{content}")
                print(f"\nToken 使用: {data.get('usage', {})}")
                print("✓ 文本推理成功")
                return True
            else:
                print(f"✗ 推理失败: {response.text}")
                return False
        except Exception as e:
            print(f"✗ 请求失败: {e}")
            return False

    def test_image_inference(self, image_path: str = None) -> bool:
        """测试图像+文本推理"""
        print("\n" + "=" * 50)
        print("测试 4: 图像+文本推理")
        print("=" * 50)

        # 如果没有提供图像，创建一个简单的测试图像（1x1 红色像素）
        if image_path is None:
            print("未提供测试图像，跳过此测试")
            print("提示: 运行 python testapi/test_api.py --image <图像路径> 来测试图像推理")
            return True

        # 读取并编码图像
        try:
            with open(image_path, "rb") as f:
                image_data = f.read()
            image_base64 = base64.b64encode(image_data).decode("utf-8")
            print(f"图像路径: {image_path}")
            print(f"图像大小: {len(image_data)} 字节")
        except Exception as e:
            print(f"✗ 读取图像失败: {e}")
            return False

        payload = {
            "model": "Qwen3-VL-32B-Instruct",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            },
                        },
                        {"type": "text", "text": "请详细描述这张图片的内容。"},
                    ],
                }
            ],
            "max_tokens": 500,
            "temperature": 0.7,
        }

        print("发送图像推理请求...")

        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=60,
            )
            print(f"\n状态码: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                print(f"\n模型回复:\n{content}")
                print(f"\nToken 使用: {data.get('usage', {})}")
                print("✓ 图像推理成功")
                return True
            else:
                print(f"✗ 推理失败: {response.text}")
                return False
        except Exception as e:
            print(f"✗ 请求失败: {e}")
            return False

    def run_all_tests(self, image_path: str = None):
        """运行所有测试"""
        print("\n" + "=" * 50)
        print("Qwen3-VL API 测试套件")
        print(f"服务地址: {self.base_url}")
        print("=" * 50)

        results = {
            "健康检查": self.test_health(),
            "模型列表": self.test_models(),
            "文本推理": self.test_text_inference(),
            "图像推理": self.test_image_inference(image_path),
        }

        # 打印测试总结
        print("\n" + "=" * 50)
        print("测试总结")
        print("=" * 50)

        passed = sum(results.values())
        total = len(results)

        for test_name, result in results.items():
            status = "✓ 通过" if result else "✗ 失败"
            print(f"{test_name}: {status}")

        print(f"\n总计: {passed}/{total} 测试通过")

        if passed == total:
            print("\n🎉 所有测试通过！")
            return 0
        else:
            print(f"\n⚠️  {total - passed} 个测试失败")
            return 1


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="Qwen3-VL API 测试脚本")
    parser.add_argument(
        "--url",
        default="http://localhost:20000",
        help="API 服务地址 (默认: http://localhost:20000)",
    )
    parser.add_argument("--image", help="测试图像路径（用于图像推理测试）")

    args = parser.parse_args()

    tester = Qwen3VLTester(base_url=args.url)
    exit_code = tester.run_all_tests(image_path=args.image)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
