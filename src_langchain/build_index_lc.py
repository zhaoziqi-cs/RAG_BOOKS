"""
向量索引构建（LangChain 版）：读 data/chunks/all_semantic.json，用 LangChain 构建并保存 FAISS 向量库

数据源：直接复用上游语义分块结果 data/chunks/all_semantic.json（parse_books.py + chunk_documents.py
产出）。书籍解析依赖 MinerU 的结构化输出，LangChain 通用 loader 无法还原，故分块由上游语义分块器完成。

Embedding：本地 BAAI/bge-m3（1024 维，中英多语言），模型目录 models/bge-m3/
向量库：LangChain FAISS 封装，保存路径 vectorstore/faiss_lc/
"""

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR        = Path(__file__).parent.parent
CHUNKS_FILE     = BASE_DIR / "data" / "chunks" / "all_semantic.json"
VECTORSTORE_DIR = BASE_DIR / "vectorstore" / "faiss_lc"
MODELS_DIR      = BASE_DIR / "models"
BGE_MODEL_PATH  = MODELS_DIR / "bge-m3"


# ── 加载文档 ──

def load_documents():
    """
    读上游语义分块结果，转成 LangChain Document 列表。
    metadata 保留 book / title / page_num / section / ... 供过滤与展示。
    """
    from langchain_core.documents import Document

    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"找不到 {CHUNKS_FILE}\n请先运行: python src/chunk_documents.py"
        )

    with open(CHUNKS_FILE, encoding="utf-8") as f:
        chunks = json.load(f)

    docs = []
    for c in chunks:
        m = c["metadata"]
        docs.append(Document(
            page_content=c["content"],
            metadata={
                "chunk_id":    c["chunk_id"],
                "book":        m.get("book", ""),
                "title":       m.get("title", ""),
                "page_num":    m.get("page_num"),
                "page_end":    m.get("page_end"),
                "section":     m.get("section", ""),
                "source_file": m.get("source_file", ""),
            },
        ))

    logger.info(f"加载 {len(docs)} 个 chunk（来自 {CHUNKS_FILE.name}）")
    return docs


# ── Embedding 模型 ──

def _device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def get_embeddings():
    """
    HuggingFaceEmbeddings 封装本地 bge-m3。
    模型须与建索引侧一致，否则查询向量与库向量不在同一向量空间。
    """
    from langchain_huggingface import HuggingFaceEmbeddings

    model_path = str(BGE_MODEL_PATH) if BGE_MODEL_PATH.exists() else "BAAI/bge-m3"
    if not BGE_MODEL_PATH.exists():
        logger.warning(
            f"本地模型不存在: {BGE_MODEL_PATH}\n"
            "  将从 HuggingFace 下载（约 2.2GB，建议先运行 python src_langchain/download_model.py）"
        )

    embeddings = HuggingFaceEmbeddings(
        model_name=model_path,
        cache_folder=str(MODELS_DIR),
        model_kwargs={"device": _device()},
        encode_kwargs={"normalize_embeddings": True},
    )
    logger.info(f"Embedding 模型加载完成: {model_path}（device={_device()}）")
    return embeddings


# ── 构建并保存 FAISS 向量库 ──

def build_vectorstore(docs, embeddings):
    """
    FAISS.from_documents() 对全部 Document 计算 embedding 并建索引，
    保存到 VECTORSTORE_DIR（index.faiss + index.pkl）。
    """
    from langchain_community.vectorstores import FAISS

    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"构建向量库（{len(docs)} 个 chunk）...")
    vectorstore = FAISS.from_documents(docs, embeddings)

    vectorstore.save_local(str(VECTORSTORE_DIR))
    logger.info(f"向量库已保存 → {VECTORSTORE_DIR}")
    logger.info(f"  index.faiss: {(VECTORSTORE_DIR / 'index.faiss').stat().st_size // 1024} KB")
    return vectorstore


# ── 主流程 ──

def main():
    docs       = load_documents()
    embeddings = get_embeddings()
    build_vectorstore(docs, embeddings)

    print(f"LangChain 向量库构建完成：{VECTORSTORE_DIR}")


if __name__ == "__main__":
    main()
