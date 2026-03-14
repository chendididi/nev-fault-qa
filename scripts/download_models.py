"""
一键下载所有模型

用法：
    # 只下载 BGE 模型（约 1.5GB，无需 GPU）
    python scripts/download_models.py

    # 同时下载 Qwen2-VL（约 16GB，需要 GPU）
    python scripts/download_models.py --qwen

    # 指定 Qwen 下载目录
    python scripts/download_models.py --qwen --qwen-dir /your/path/models/Qwen2-VL-7B-Instruct

注意：
    国内用户建议先设置 HuggingFace 镜像：
    export HF_ENDPOINT=https://hf-mirror.com  (Linux/Mac)
    $env:HF_ENDPOINT="https://hf-mirror.com"  (Windows PowerShell)
"""

import argparse
import os
import sys
from pathlib import Path

# 将项目根目录加入路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def download_bge_models():
    """下载 BGE 嵌入模型和重排序模型（通过 HuggingFace / 镜像自动下载）。"""
    print("\n" + "=" * 50)
    print("下载 BGE 嵌入模型：BAAI/bge-base-zh-v1.5（约 400MB）")
    print("=" * 50)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("BAAI/bge-base-zh-v1.5")
    print("✅ BGE 嵌入模型下载完成")

    print("\n" + "=" * 50)
    print("下载 BGE 重排序模型：BAAI/bge-reranker-v2-m3（约 1.1GB）")
    print("=" * 50)
    from sentence_transformers import CrossEncoder
    model = CrossEncoder("BAAI/bge-reranker-v2-m3")
    print("✅ BGE 重排序模型下载完成")


def download_qwen(local_dir: str):
    """通过 ModelScope 下载 Qwen2-VL-7B-Instruct（约 16GB）。"""
    print("\n" + "=" * 50)
    print(f"下载 Qwen2-VL-7B-Instruct（约 16GB）")
    print(f"目标目录：{local_dir}")
    print("=" * 50)
    print("使用 ModelScope 下载（国内速度快）...")

    try:
        from modelscope import snapshot_download
    except ImportError:
        print("正在安装 modelscope...")
        os.system(f"{sys.executable} -m pip install modelscope")
        from modelscope import snapshot_download

    snapshot_download(
        model_id="Qwen/Qwen2-VL-7B-Instruct",
        local_dir=local_dir,
    )
    print(f"✅ Qwen2-VL-7B-Instruct 下载完成，路径：{local_dir}")
    print(f"\n请确认 config/config.yaml 中的 model_path 为：\n  {local_dir}")


def main():
    parser = argparse.ArgumentParser(description="一键下载项目所需模型")
    parser.add_argument(
        "--qwen",
        action="store_true",
        help="同时下载 Qwen2-VL-7B-Instruct（约 16GB，需要 GPU）",
    )
    parser.add_argument(
        "--qwen-dir",
        default="/root/ubuntuchen_file/models/Qwen2-VL-7B-Instruct",
        help="Qwen 模型下载目录（默认：/root/ubuntuchen_file/models/Qwen2-VL-7B-Instruct）",
    )
    parser.add_argument(
        "--skip-bge",
        action="store_true",
        help="跳过 BGE 模型下载",
    )
    args = parser.parse_args()

    print("NEV Fault QA — 模型下载脚本")
    print(f"HF_ENDPOINT: {os.environ.get('HF_ENDPOINT', '未设置（建议设置镜像）')}")

    if not args.skip_bge:
        download_bge_models()

    if args.qwen:
        download_qwen(args.qwen_dir)
    else:
        print("\n提示：Qwen2-VL 未下载。如需本地部署，运行：")
        print("  python scripts/download_models.py --qwen")

    print("\n🎉 完成！")


if __name__ == "__main__":
    main()
