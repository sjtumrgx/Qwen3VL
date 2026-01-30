#!/usr/bin/env python3
"""
Qwen3-VL 媒体分析测试脚本

自动下载示例图片/视频，测试 Qwen3-VL 的多模态分析能力
支持资源缓存，避免重复下载

使用方法:
    python testapi/test_media.py --url http://localhost:20000
"""

import argparse
import io
import json
import sys
from pathlib import Path
from typing import List, Tuple

import requests


def _ensure_utf8_stdio() -> None:
    """确保标准输出使用 UTF-8 编码"""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            try:
                buffer = getattr(stream, "buffer", None)
                if buffer is not None:
                    wrapped = io.TextIOWrapper(
                        buffer, encoding="utf-8", errors="replace", line_buffering=True
                    )
                    setattr(sys, stream_name, wrapped)
            except Exception:
                pass


_ensure_utf8_stdio()


# 示例资源配置
SAMPLE_RESOURCES = {
    "image": {
        "url": "https://images.unsplash.com/photo-1574158622682-e40e69881006?w=800",
        "filename": "sample_cat.jpg",
        "description": "一只猫的照片",
        "expected_keywords": ["cat", "猫", "animal", "动物", "毛", "眼睛", "耳朵"],
    },
    "video": {
        "url": "https://vjs.zencdn.net/v/oceans.mp4",
        "filename": "sample_oceans.mp4",
        "description": "海洋风景视频",
        "expected_keywords": [
            "ocean", "海", "water", "水", "sea", "洋",
            "wave", "浪", "blue", "蓝", "fish", "鱼",
            "underwater", "水下", "marine", "海洋",
        ],
    },
}


class MediaDownloader:
    """媒体资源下载器（支持缓存）"""

    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, filename: str) -> Path:
        """获取缓存文件路径"""
        return self.cache_dir / filename

    def _is_cached(self, filename: str) -> bool:
        """检查文件是否已缓存"""
        cache_path = self._get_cache_path(filename)
        return cache_path.exists() and cache_path.stat().st_size > 0

    def download(self, url: str, filename: str, timeout: int = 60) -> Tuple[bytes, str]:
        """
        下载资源（优先使用缓存）

        Returns:
            (文件内容, 缓存路径)
        """
        cache_path = self._get_cache_path(filename)

        if self._is_cached(filename):
            print(f"  使用缓存: {cache_path}")
            return cache_path.read_bytes(), str(cache_path)

        print(f"  下载中: {url}")
        response = requests.get(url, timeout=timeout, stream=True)
        response.raise_for_status()

        content = response.content
        cache_path.write_bytes(content)
        print(f"  已缓存到: {cache_path}")

        return content, str(cache_path)


class ResponseValidator:
    """响应内容验证器"""

    @staticmethod
    def validate_keywords(
        response_text: str, expected_keywords: List[str], min_matches: int = 1
    ) -> Tuple[bool, List[str]]:
        """
        验证响应是否包含预期关键词

        Args:
            response_text: 模型响应文本
            expected_keywords: 预期关键词列表
            min_matches: 最少匹配数量

        Returns:
            (是否通过, 匹配到的关键词列表)
        """
        response_lower = response_text.lower()
        matched = [kw for kw in expected_keywords if kw.lower() in response_lower]
        return len(matched) >= min_matches, matched

    @staticmethod
    def validate_non_empty(response_text: str, min_length: int = 10) -> bool:
        """验证响应非空且有意义"""
        return len(response_text.strip()) >= min_length


class Qwen3VLMediaTester:
    """Qwen3-VL 媒体分析测试器"""

    def __init__(self, base_url: str = "http://localhost:20000", cache_dir: str = "cache"):
        self.base_url = base_url.rstrip("/")
        self.headers = {"Content-Type": "application/json"}
        self.downloader = MediaDownloader(cache_dir)
        self.validator = ResponseValidator()

    def test_health(self) -> bool:
        """测试服务健康状态"""
        print("\n" + "=" * 60)
        print("前置检查: 服务健康状态")
        print("=" * 60)

        try:
            response = requests.get(f"{self.base_url}/health", timeout=30)
            if response.status_code == 200:
                data = response.json()
                print(f"  状态: {data.get('status')}")
                print(f"  模型: {data.get('model')}")
                print(f"  GPU: {data.get('gpu_count')} 个")
                print("✓ 服务正常")
                return True
            else:
                print(f"✗ 服务异常: {response.status_code}")
                return False
        except Exception as e:
            print(f"✗ 连接失败: {e}")
            return False

    def test_image_analysis(self) -> Tuple[bool, str]:
        """测试图像分析"""
        print("\n" + "=" * 60)
        print("测试 1: 图像分析")
        print("=" * 60)

        resource = SAMPLE_RESOURCES["image"]
        print(f"  资源描述: {resource['description']}")

        try:
            # 下载图片
            image_data, cache_path = self.downloader.download(
                resource["url"], resource["filename"]
            )
            print(f"  图片大小: {len(image_data) / 1024:.1f} KB")

            # 调用分析接口
            print("  发送分析请求...")
            response = requests.post(
                f"{self.base_url}/analyze/upload",
                files={"image": (resource["filename"], image_data, "image/jpeg")},
                data={
                    "instruction": "请详细描述这张图片中的内容，包括主体、颜色、场景等。",
                    "max_tokens": 500,
                },
                timeout=120,
            )

            if response.status_code != 200:
                print(f"✗ 请求失败: {response.status_code} - {response.text}")
                return False, ""

            result = response.json()
            analysis = result.get("analysis", "")
            tokens = result.get("tokens", {})

            print(f"\n  分析结果:\n  {'-' * 50}")
            print(f"  {analysis[:500]}{'...' if len(analysis) > 500 else ''}")
            print(f"  {'-' * 50}")
            print(f"  Token 使用: {tokens}")

            # 验证响应
            is_valid_length = self.validator.validate_non_empty(analysis, 20)
            is_valid_content, matched_kw = self.validator.validate_keywords(
                analysis, resource["expected_keywords"], min_matches=1
            )

            print(f"\n  验证结果:")
            print(f"    长度检查: {'✓' if is_valid_length else '✗'} (长度: {len(analysis)})")
            print(f"    内容检查: {'✓' if is_valid_content else '✗'} (匹配关键词: {matched_kw})")

            if is_valid_length and is_valid_content:
                print("\n✓ 图像分析测试通过")
                return True, analysis
            else:
                print("\n✗ 图像分析测试失败（响应内容不符合预期）")
                return False, analysis

        except Exception as e:
            print(f"✗ 测试异常: {e}")
            return False, ""

    def test_video_analysis(self) -> Tuple[bool, str]:
        """测试视频分析"""
        print("\n" + "=" * 60)
        print("测试 2: 视频分析")
        print("=" * 60)

        resource = SAMPLE_RESOURCES["video"]
        print(f"  资源描述: {resource['description']}")

        try:
            # 下载视频
            video_data, cache_path = self.downloader.download(
                resource["url"], resource["filename"], timeout=120
            )
            print(f"  视频大小: {len(video_data) / 1024 / 1024:.1f} MB")

            # 调用视频分析接口
            print("  发送视频分析请求（抽帧 + 多图分析）...")
            response = requests.post(
                f"{self.base_url}/analyze/video/upload",
                files={"video": (resource["filename"], video_data, "video/mp4")},
                data={
                    "instruction": "这是一段视频的多个帧。请分析视频的内容，描述场景、主体和发生的事情。",
                    "max_frames": 8,
                    "max_tokens": 800,
                },
                timeout=180,
            )

            if response.status_code != 200:
                print(f"✗ 请求失败: {response.status_code} - {response.text}")
                return False, ""

            result = response.json()
            analysis = result.get("analysis", "")
            frames_extracted = result.get("frames_extracted", 0)
            tokens = result.get("tokens", {})

            print(f"\n  抽取帧数: {frames_extracted}")
            print(f"  分析结果:\n  {'-' * 50}")
            print(f"  {analysis[:600]}{'...' if len(analysis) > 600 else ''}")
            print(f"  {'-' * 50}")
            print(f"  Token 使用: {tokens}")

            # 验证响应
            is_valid_length = self.validator.validate_non_empty(analysis, 30)
            is_valid_content, matched_kw = self.validator.validate_keywords(
                analysis, resource["expected_keywords"], min_matches=1
            )

            print(f"\n  验证结果:")
            print(f"    长度检查: {'✓' if is_valid_length else '✗'} (长度: {len(analysis)})")
            print(f"    内容检查: {'✓' if is_valid_content else '✗'} (匹配关键词: {matched_kw})")

            if is_valid_length and is_valid_content:
                print("\n✓ 视频分析测试通过")
                return True, analysis
            else:
                print("\n✗ 视频分析测试失败（响应内容不符合预期）")
                return False, analysis

        except Exception as e:
            print(f"✗ 测试异常: {e}")
            return False, ""

    def test_streaming_chat(self) -> Tuple[bool, str]:
        """测试流式聊天输出"""
        print("\n" + "=" * 60)
        print("测试 3: 流式聊天输出 (SSE)")
        print("=" * 60)

        try:
            print("  发送流式请求...")
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers=self.headers,
                json={
                    "messages": [{"role": "user", "content": "用一句话介绍人工智能"}],
                    "stream": True,
                    "max_tokens": 100,
                },
                stream=True,
                timeout=60,
            )

            if response.status_code != 200:
                print(f"✗ 请求失败: {response.status_code}")
                return False, ""

            # 解析 SSE 流
            full_content = ""
            chunk_count = 0
            print(f"\n  流式输出:\n  {'-' * 50}")
            print("  ", end="", flush=True)

            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue

                if line.startswith("data:"):
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break

                    try:
                        data = json.loads(data_str)
                        choices = data.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                full_content += content
                                chunk_count += 1
                                print(content, end="", flush=True)
                    except json.JSONDecodeError:
                        continue

            print(f"\n  {'-' * 50}")
            print(f"  接收到 {chunk_count} 个数据块")
            print(f"  完整内容长度: {len(full_content)} 字符")

            # 验证
            is_valid = len(full_content) >= 10 and chunk_count >= 2

            print(f"\n  验证结果:")
            print(f"    流式传输: {'✓' if chunk_count >= 2 else '✗'} (数据块: {chunk_count})")
            print(f"    内容完整: {'✓' if len(full_content) >= 10 else '✗'} (长度: {len(full_content)})")

            if is_valid:
                print("\n✓ 流式聊天测试通过")
                return True, full_content
            else:
                print("\n✗ 流式聊天测试失败")
                return False, full_content

        except Exception as e:
            print(f"✗ 测试异常: {e}")
            return False, ""

    def test_streaming_infer(self) -> Tuple[bool, str]:
        """测试流式推理输出"""
        print("\n" + "=" * 60)
        print("测试 4: 流式推理输出 (SSE)")
        print("=" * 60)

        try:
            print("  发送流式推理请求...")
            response = requests.post(
                f"{self.base_url}/infer",
                headers=self.headers,
                json={
                    "prompt": "什么是深度学习？用一句话回答",
                    "stream": True,
                    "max_tokens": 80,
                },
                stream=True,
                timeout=60,
            )

            if response.status_code != 200:
                print(f"✗ 请求失败: {response.status_code}")
                return False, ""

            # 解析 SSE 流
            full_content = ""
            chunk_count = 0
            print(f"\n  流式输出:\n  {'-' * 50}")
            print("  ", end="", flush=True)

            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue

                if line.startswith("data:"):
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break

                    try:
                        data = json.loads(data_str)
                        choices = data.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                full_content += content
                                chunk_count += 1
                                print(content, end="", flush=True)
                    except json.JSONDecodeError:
                        continue

            print(f"\n  {'-' * 50}")
            print(f"  接收到 {chunk_count} 个数据块")
            print(f"  完整内容长度: {len(full_content)} 字符")

            # 验证
            is_valid = len(full_content) >= 10 and chunk_count >= 2

            print(f"\n  验证结果:")
            print(f"    流式传输: {'✓' if chunk_count >= 2 else '✗'} (数据块: {chunk_count})")
            print(f"    内容完整: {'✓' if len(full_content) >= 10 else '✗'} (长度: {len(full_content)})")

            if is_valid:
                print("\n✓ 流式推理测试通过")
                return True, full_content
            else:
                print("\n✗ 流式推理测试失败")
                return False, full_content

        except Exception as e:
            print(f"✗ 测试异常: {e}")
            return False, ""

    def run_all_tests(self) -> int:
        """运行所有测试"""
        print("\n" + "=" * 60)
        print("Qwen3-VL 媒体分析测试套件")
        print(f"服务地址: {self.base_url}")
        print("=" * 60)

        # 前置检查
        if not self.test_health():
            print("\n⚠️ 服务不可用，测试终止")
            return 1

        # 运行测试
        results = {}

        image_passed, _ = self.test_image_analysis()
        results["图像分析"] = image_passed

        video_passed, _ = self.test_video_analysis()
        results["视频分析"] = video_passed

        streaming_chat_passed, _ = self.test_streaming_chat()
        results["流式聊天"] = streaming_chat_passed

        streaming_infer_passed, _ = self.test_streaming_infer()
        results["流式推理"] = streaming_infer_passed

        # 打印总结
        print("\n" + "=" * 60)
        print("测试总结")
        print("=" * 60)

        passed = sum(results.values())
        total = len(results)

        for test_name, result in results.items():
            status = "✓ 通过" if result else "✗ 失败"
            print(f"  {test_name}: {status}")

        print(f"\n总计: {passed}/{total} 测试通过")

        if passed == total:
            print("\n🎉 所有测试通过！")
            return 0
        else:
            print(f"\n⚠️ {total - passed} 个测试失败")
            return 1


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Qwen3-VL 媒体分析测试脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python testapi/test_media.py
    python testapi/test_media.py --url http://localhost:20000
    python testapi/test_media.py --cache-dir ./my_cache
        """,
    )
    parser.add_argument(
        "--url",
        default="http://localhost:20000",
        help="API 服务地址 (默认: http://localhost:20000)",
    )
    parser.add_argument(
        "--cache-dir",
        default="cache",
        help="资源缓存目录 (默认: cache)",
    )

    args = parser.parse_args()

    tester = Qwen3VLMediaTester(base_url=args.url, cache_dir=args.cache_dir)
    exit_code = tester.run_all_tests()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
