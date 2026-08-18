"""
向量索引构建：读 data/chunks/all_semantic.json，向量化并建 FAISS 索引

Embedding：BAAI/bge-m3（本地，1024 维，中英多语言）
向量库：FAISS IndexFlatIP（向量已 L2 归一化，内积即余弦相似度）
"""

import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR        = Path(__file__).parent.parent
CHUNKS_DIR      = BASE_DIR / "data" / "chunks"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

STRATEGY    = "semantic"          # 与 chunk_documents.py 保持一致
CHUNKS_FILE = CHUNKS_DIR / f"all_{STRATEGY}.json"

EMBED_MODEL = str(BASE_DIR / "models" / "bge-m3")   # 本地模型目录（绝对路径，避免依赖 CWD）
EMBED_DIM   = 1024
BATCH_SIZE  = 32


_embed_model = None


def _get_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        logger.info(f"加载 embedding 模型 {EMBED_MODEL} ...")
        _embed_model = SentenceTransformer(EMBED_MODEL)
    return _embed_model


def embed_texts(texts: list[str]) -> np.ndarray:
    """批量计算 embedding，返回 shape=(N, EMBED_DIM) 的 float32 数组，已 L2 归一化。"""
    model = _get_model()
    vecs = model.encode(texts, batch_size=BATCH_SIZE, normalize_embeddings=True, show_progress_bar=True)
    return np.asarray(vecs, dtype="float32")


def build_faiss_index(chunks: list[dict]):
    import faiss

    logger.info(f"开始计算 {len(chunks)} 条 chunk 的 embedding ...")
    embeddings = embed_texts([c["content"] for c in chunks])

    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(embeddings)

    index_path = VECTORSTORE_DIR / "faiss_index.bin"
    faiss.write_index(index, str(index_path))
    logger.info(f"FAISS 索引已保存 → {index_path}  ({index_path.stat().st_size // 1024} KB)")

    # 元数据与向量分开存，检索时按位置对齐（faiss_index.bin 第 i 条 ↔ 本列表第 i 条）
    meta_list = [
        {
            "chunk_id":    c["chunk_id"],
            "content":     c["content"],
            "book":        c["metadata"].get("book", ""),
            "title":       c["metadata"].get("title", ""),
            "page_num":    c["metadata"].get("page_num"),
            "page_end":    c["metadata"].get("page_end"),
            "section":     c["metadata"].get("section", ""),
            "block_types": c["metadata"].get("block_types", []),
            "source_file": c["metadata"].get("source_file", ""),
            "strategy":    c["metadata"].get("strategy", ""),
            "images":      c["metadata"].get("images", []),
            "parent_id":      c["metadata"].get("parent_id", ""),
            "parent_content": c["metadata"].get("parent_content", ""),
        }
        for c in chunks
    ]
    meta_path = VECTORSTORE_DIR / "faiss_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_list, f, ensure_ascii=False, indent=2)
    logger.info(f"元数据已保存 → {meta_path}")

    return index, meta_list


def main():
    if not CHUNKS_FILE.exists():
        logger.error(f"找不到 {CHUNKS_FILE}，请先运行 chunk_documents.py")
        return

    with open(CHUNKS_FILE, encoding="utf-8") as f:
        chunks = json.load(f)
    logger.info(f"加载 {len(chunks)} 个 chunks（策略={STRATEGY}）")

    build_faiss_index(chunks)
    logger.info("\n索引构建完成！")


if __name__ == "__main__":
    main()
