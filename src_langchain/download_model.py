"""
BGE 模型下载脚本

将 BAAI/bge-m3 下载到项目的 models/ 目录，而非 HuggingFace 默认的用户缓存
路径（~/.cache/huggingface/），使项目自包含、模型随项目迁移。

模型：BAAI/bge-m3（多语言，中英双语，1024 维，约 2.2GB），下载到 models/bge-m3/
"""

import sys
from pathlib import Path

# 项目根目录
BASE_DIR   = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

MODEL_NAME = "BAAI/bge-m3"
LOCAL_DIR  = MODELS_DIR / "bge-m3"


def download():
    print(f"目标目录: {LOCAL_DIR}")

    if LOCAL_DIR.exists() and any(LOCAL_DIR.iterdir()):
        # 检查关键文件是否存在
        key_files = ["config.json", "tokenizer.json"]
        if all((LOCAL_DIR / f).exists() for f in key_files):
            print(f"模型已存在，跳过下载: {LOCAL_DIR}")
            return str(LOCAL_DIR)

    print(f"开始下载 {MODEL_NAME} → {LOCAL_DIR}")
    print("首次下载约 2.2GB，请耐心等待...\n")

    try:
        from huggingface_hub import snapshot_download
        path = snapshot_download(
            repo_id=MODEL_NAME,
            local_dir=str(LOCAL_DIR),
            local_dir_use_symlinks=False,   # 真实文件，不用软链接
            ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*", "onnx/*"],  # 只要 PyTorch 格式
        )
        print(f"\n下载完成！模型路径: {path}")
        return path
    except Exception as e:
        print(f"\n下载失败: {e}")
        print("\n可以尝试：")
        print("  1. 设置镜像: set HF_ENDPOINT=https://hf-mirror.com")
        print("  2. 手动从 https://hf-mirror.com/BAAI/bge-m3 下载")
        print(f"     解压到: {LOCAL_DIR}")
        sys.exit(1)


def verify(model_path: str):
    """验证模型可以正常加载。"""
    print("\n验证模型可用性...")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_path)
        test_vec = model.encode(["测试句子"], normalize_embeddings=True)
        print(f"验证通过！embedding 维度: {test_vec.shape[1]}")
    except Exception as e:
        print(f"验证失败: {e}")
        print("模型文件可能不完整，请删除后重新下载")


if __name__ == "__main__":
    path = download()
    verify(path)
