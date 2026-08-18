"""
RAG 问答流水线：查询改写（可选）→ 向量检索（bge-m3 + FAISS）+ BM25（jieba + rank_bm25）
→ RRF 融合 → CrossEncoder Rerank（可选）→ 相关性阈值过滤 → LLM 生成（DashScope qwen-plus）。

查询侧 embedding 模型须与建索引一致（本地 BAAI/bge-m3，1024 维，L2 归一化，
IndexFlatIP 内积即余弦相似度），否则查询向量与库向量不在同一空间。

上游产物：data/manifest.json（download_books.py）→ data/parsed/manifest.json（parse_books.py）
→ data/chunks/all_semantic.json（chunk_documents.py）→ vectorstore/faiss_index.bin + faiss_meta.json（build_index.py）
"""

import os
import re
import json
import logging
import argparse
import numpy as np
from pathlib import Path
from typing import Optional
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR        = Path(__file__).parent.parent
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
INDEX_PATH      = VECTORSTORE_DIR / "faiss_index.bin"
META_PATH       = VECTORSTORE_DIR / "faiss_meta.json"

# embedding：本地 bge-m3，与 build_index.py 保持一致
EMBED_MODEL_DIR = str(BASE_DIR / "models" / "bge-m3")   # 本地模型目录（绝对路径，避免依赖 CWD）
EMBED_DIM       = 1024

# LLM 生成：DashScope（OpenAI 兼容接口）
DASHSCOPE_URL   = "https://dashscope.aliyuncs.com/compatible-mode/v1"
LLM_MODEL       = "qwen-plus"        # 可换 qwen-turbo（更快）/ qwen-max（更强）

TOP_K_RETRIEVE  = 10    # 初始召回数
TOP_K_RERANK    = 4     # Rerank 后保留数（给 LLM）
SCORE_THRESHOLD = 0.25  # 最高余弦相似度低于此值时触发拒绝回答（bge-m3 归一化后为余弦）

SYSTEM_PROMPT = """你是一个专业的技术书籍问答助手，根据给定的机器学习 / 深度学习 / 数据科学书籍资料回答问题。

回答规则：
1. 只根据【参考资料】中的内容回答，不得引用或编造资料外的内容
2. 若参考资料不足以支撑回答，直接说"根据提供的资料无法回答此问题"
3. 引用具体内容时，在句末标注来源编号，如：梯度下降通过迭代更新参数来最小化损失函数[1]
4. 涉及公式、代码、定义时要忠实于原文，不要凭记忆改写
5. 用与用户问题相同的语言回答（中文问题用中文回答）
6. 回答简洁，重点突出，避免无关废话"""


# ── 本地 embedding 模型（懒加载，与 build_index.py 一致）──────────────────────

_embed_model = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        logger.info(f"加载 embedding 模型 {EMBED_MODEL_DIR} ...")
        _embed_model = SentenceTransformer(EMBED_MODEL_DIR)
    return _embed_model


def embed_query(query: str) -> np.ndarray:
    """计算查询向量，返回 shape=(1, EMBED_DIM) 的 float32 数组，已 L2 归一化。"""
    model = _get_embed_model()
    vecs = model.encode([query], normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vecs, dtype="float32")


# ── DashScope 客户端（仅 LLM 生成用）──────────────────────────────────────────

def get_client() -> OpenAI:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise EnvironmentError("请设置环境变量 DASHSCOPE_API_KEY（仅 LLM 生成需要）")
    return OpenAI(api_key=api_key, base_url=DASHSCOPE_URL)


# ── 向量检索 ──────────────────────────────────────────────────────────────────

def _match_filter(item: dict, filter_meta: dict) -> bool:
    """元数据过滤：book 同时匹配 name / title 字段（大小写不敏感、子串匹配），section 子串匹配。"""
    if "book" in filter_meta:
        q = filter_meta["book"].lower()
        book  = str(item.get("book", "")).lower()
        title = str(item.get("title", "")).lower()
        if q not in book and q not in title:
            return False
    if "section" in filter_meta:
        q = filter_meta["section"].lower()
        if q not in str(item.get("section", "")).lower():
            return False
    return True


class VectorStore:
    def __init__(self):
        import faiss
        self.index = faiss.read_index(str(INDEX_PATH))
        with open(META_PATH, encoding="utf-8") as f:
            self.meta_list = json.load(f)
        logger.info(f"FAISS 索引加载完成，共 {self.index.ntotal} 条向量")

    def search(
        self,
        query: str,
        top_k: int = TOP_K_RETRIEVE,
        filter_meta: Optional[dict] = None,
    ) -> list[dict]:
        """
        向量检索，可选元数据过滤（按 book / section）。
        filter_meta 示例：{"book": "dive_into_deep_learning", "section": "softmax"}
        """
        query_vec = embed_query(query)
        # 多取一些再过滤，避免过滤后不足 top_k
        scores, indices = self.index.search(query_vec, top_k * 4)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.meta_list):
                continue
            item = dict(self.meta_list[idx])
            item["vec_score"] = float(score)

            if filter_meta and not _match_filter(item, filter_meta):
                continue

            results.append(item)
            if len(results) >= top_k:
                break
        return results

    def scores_for(self, query: str, chunk_ids: list[str]) -> dict[str, float]:
        """
        批量查询指定 chunk 的向量余弦分数（RRF 融合后统一刷新 vec_score 用）。
        一次全量检索（top_k=ntotal），按位置对齐 meta_list 建立 chunk_id → vec_score 映射，
        返回请求的 chunk_ids 的分数；分数为索引中真实向量的余弦相似度。
        """
        query_vec = embed_query(query)
        scores, indices = self.index.search(query_vec, self.index.ntotal)

        score_map: dict[str, float] = {}
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.meta_list):
                continue
            score_map[self.meta_list[idx]["chunk_id"]] = float(score)

        return {cid: score_map[cid] for cid in chunk_ids if cid in score_map}


# ── BM25 关键词检索 ───────────────────────────────────────────────────────────

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """英文分词：小写后提取英文单词/数字序列，供语料与查询两侧复用，保证分词一致。"""
    return _TOKEN_RE.findall(text.lower())


class BM25Store:
    """
    基于 rank_bm25 的关键词检索，语料为英文，用英文分词（小写 + 提取单词/数字）。
    首次初始化会分词整个语料库，约需数秒。
    注意：纯中文查询会被切成空 token（返回空结果），需配合 --query-rewrite 翻译成英文。
    """
    def __init__(self):
        from rank_bm25 import BM25Okapi

        with open(META_PATH, encoding="utf-8") as f:
            self.meta_list = json.load(f)

        logger.info("构建 BM25 索引（分词中，请稍候）...")
        tokenized   = [_tokenize(item["content"]) for item in self.meta_list]
        self.bm25   = BM25Okapi(tokenized)
        logger.info("BM25 索引完成")

    def search(
        self,
        query: str,
        top_k: int = TOP_K_RETRIEVE,
        filter_meta: Optional[dict] = None,
    ) -> list[dict]:
        """
        BM25 关键词检索，可选元数据过滤（按 book / section）。
        filter_meta 示例：{"book": "dive_into_deep_learning", "section": "softmax"}
        BM25 已对全量文档打分，直接在降序结果上过滤即可，过滤条件严格时仍能取到靠后的符合条件 chunk。
        """
        tokens = _tokenize(query)
        if not tokens:  # 查询无有效英文 token（如纯中文未翻译），跳过 BM25
            return []

        scores = self.bm25.get_scores(tokens)
        order = np.argsort(scores)[::-1]   # 全量降序（不切片，保证过滤后仍能取满 top_k）

        results = []
        for idx in order:
            if scores[idx] <= 1e-9:  # 零分（无任何词面重叠）跳过
                continue
            item = dict(self.meta_list[idx])
            item["bm25_score"] = float(scores[idx])

            if filter_meta and not _match_filter(item, filter_meta):
                continue

            results.append(item)
            if len(results) >= top_k:
                break
        return results


# ── RRF 融合 ──────────────────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    vec_results: list[dict],
    bm25_results: list[dict],
    k: int = 60,
) -> list[dict]:
    """
    Reciprocal Rank Fusion（RRF）。
    公式：score(d) = Σ 1/(k + rank_i(d))，k=60 为经验值。
    将向量召回和 BM25 召回的排名合并，互补各自的盲区。
    """
    rrf_scores: dict[str, float] = {}
    chunk_map:  dict[str, dict]  = {}

    for rank, item in enumerate(vec_results, 1):
        cid = item["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (k + rank)
        chunk_map[cid]  = item

    for rank, item in enumerate(bm25_results, 1):
        cid = item["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (k + rank)
        chunk_map[cid]  = item

    sorted_cids = sorted(rrf_scores, key=lambda x: -rrf_scores[x])
    results = []
    for cid in sorted_cids:
        item = dict(chunk_map[cid])
        item["rrf_score"] = rrf_scores[cid]
        results.append(item)
    return results


# ── CrossEncoder Rerank（可选）────────────────────────────────────────────────

def rerank(query: str, candidates: list[dict], top_k: int = TOP_K_RERANK) -> list[dict]:
    """
    用 CrossEncoder 对候选集二次精排。
    模型：BAAI/bge-reranker-base（优先本地 models/bge-reranker-base）

    若模型不可用（未下载 / 未安装 sentence-transformers），自动降级为直接截断。
    """
    try:
        from sentence_transformers import CrossEncoder
        # 如果模型已下载到项目 models/ 目录，优先用本地路径
        model_path = Path(__file__).parent.parent / "models" / "bge-reranker-base"
        model_name = str(model_path) if model_path.exists() else "BAAI/bge-reranker-base"
        reranker = CrossEncoder(model_name)
        pairs    = [(query, c["content"]) for c in candidates]
        scores   = reranker.predict(pairs)
        for item, score in zip(candidates, scores):
            item["rerank_score"] = float(score)
        candidates.sort(key=lambda x: -x.get("rerank_score", 0))
    except ImportError:
        logger.warning("sentence-transformers 未安装，跳过 Rerank（pip install sentence-transformers）")
    except Exception as e:
        logger.warning(f"Rerank 失败，使用 RRF 原始排序: {e}")

    return candidates[:top_k]


# ── 查询改写 ──────────────────────────────────────────────────────────────────

def rewrite_query(query: str, client: OpenAI) -> str:
    """
    用 LLM 把用户问题改写成适合检索的英文查询（语料为英文，中文问题翻译后
    BM25 词法检索才能命中）。模型用 qwen-turbo。
    """
    resp = client.chat.completions.create(
        model="qwen-turbo",
        messages=[
            {
                "role": "system",
                "content": (
                    "你是检索查询优化专家。语料库为英文技术书籍，请把用户问题改写成英文检索查询。"
                    "若问题含中文，先准确翻译成英文，专业术语要精确（如「梯度下降」→ gradient descent，书名用英文名）；"
                    "保留关键实体（书名、章节、术语、公式符号），可扩展 1-2 个近义/上位词。"
                    "直接输出英文查询语句，不要解释，不要超过 20 个词。"
                ),
            },
            {"role": "user", "content": query},
        ],
        temperature=0,
    )
    rewritten = resp.choices[0].message.content.strip()
    logger.info(f"查询改写: {query!r} → {rewritten!r}")
    return rewritten


# ── LLM 生成 ──────────────────────────────────────────────────────────────────

def build_context(retrieved: list[dict]) -> tuple[str, list[dict]]:
    """将检索结果组装为 Prompt 上下文，返回上下文字符串和引用列表。"""
    parts     = []
    citations = []

    for i, item in enumerate(retrieved, 1):
        title   = item.get("title", "") or item.get("book", "")
        section = item.get("section", "")
        page    = item.get("page_num")

        label = f"[{i}] 《{title}》"
        if section:
            label += f" · {section}"
        if page:
            label += f" · 第{page}页"

        # 层级分块时优先用父块内容（信息更完整）；semantic 策略下 parent_content 为空则退回 content
        content = item.get("parent_content") or item.get("content", "")
        parts.append(f"{label}\n{content}")
        citations.append({"index": i, "source": label, "chunk_id": item.get("chunk_id", "")})

    return "\n\n---\n\n".join(parts), citations


def call_llm(query: str, context: str, client: OpenAI) -> str:
    user_msg = (
        f"【参考资料】\n{context}\n\n"
        f"【问题】\n{query}\n\n"
        "请根据参考资料回答，并在引用数据处标注来源编号（如[1]）。"
    )
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0.1,
    )
    return resp.choices[0].message.content


# ── 完整流水线 ────────────────────────────────────────────────────────────────

class RAGPipeline:
    def __init__(
        self,
        use_bm25:         bool = True,
        use_rerank:       bool = True,
        use_query_rewrite: bool = True,
    ):
        self.client     = get_client()   # 仅用于 LLM 生成
        self.vec_store  = VectorStore()
        self.use_bm25   = use_bm25
        self.use_rerank = use_rerank
        self.use_query_rewrite = use_query_rewrite
        self.bm25_store = BM25Store() if use_bm25 else None

    def query(
        self,
        question: str,
        filter_meta: Optional[dict] = None,
        verbose: bool = True,
    ) -> dict:
        # ① 查询改写（可选）
        retrieval_query = rewrite_query(question, self.client) if self.use_query_rewrite else question

        # ② 向量检索
        vec_results = self.vec_store.search(retrieval_query, TOP_K_RETRIEVE, filter_meta)
        if verbose:
            logger.info(f"向量召回: {len(vec_results)} 条，最高分={vec_results[0]['vec_score']:.3f}" if vec_results else "向量召回: 0 条")

        # ③ BM25 + RRF 融合
        if self.use_bm25 and self.bm25_store:
            bm25_results = self.bm25_store.search(retrieval_query, TOP_K_RETRIEVE, filter_meta)
            candidates   = reciprocal_rank_fusion(vec_results, bm25_results)
            # RRF 后对所有候选重查一次向量分数并回填 vec_score：
            # 消除 BM25 独有 chunk 无分、以及 RRF 中 bm25 覆盖 vec 造成的分数缺口，
            # 保证阈值侧每个候选都有真实余弦相似度。
            new_scores = self.vec_store.scores_for(retrieval_query, [c["chunk_id"] for c in candidates])
            for c in candidates:
                c["vec_score"] = new_scores.get(c["chunk_id"])
            if verbose:
                logger.info(f"BM25 召回: {len(bm25_results)} 条，RRF 后: {len(candidates)} 条")
        else:
            candidates = vec_results

        # ④ Rerank
        if self.use_rerank:
            final = rerank(question, candidates, TOP_K_RERANK)
        else:
            final = candidates[:TOP_K_RERANK]

        if verbose:
            logger.info(f"最终使用 {len(final)} 条上下文")

        # ⑤ 相关性阈值检查
        if not final:
            return {
                "answer": "未找到相关内容，无法回答此问题。",
                "citations": [], "retrieved": [],
            }
        # 阈值只用 vec_score：RRF 后已统一回填，final[0] 必有真实余弦值；
        # rerank_score（未 sigmoid 的 logit）与 rrf_score 量纲不同，均不参与阈值。
        # filter_meta 非空时跳过阈值。
        top_score = final[0].get("vec_score")
        if top_score is not None and top_score < SCORE_THRESHOLD and filter_meta is None:
            return {
                "answer": "根据书籍知识库未能找到与该问题相关的内容，建议直接查阅原书。",
                "citations": [], "retrieved": final,
            }

        # ⑥ LLM 生成
        context, citations = build_context(final)
        answer = call_llm(question, context, self.client)

        return {"answer": answer, "citations": citations, "retrieved": final}


# ── 入口 ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="书籍 RAG 问答（原生版）")
    parser.add_argument("--query",         type=str,  default=None)
    parser.add_argument("--book",          type=str,  default=None, help="限定书籍（书名或名称，模糊匹配）")
    parser.add_argument("--section",       type=str,  default=None, help="限定章节（模糊匹配）")
    parser.add_argument("--query-rewrite", action="store_true", help="开启查询改写（增加一次 LLM 调用）")
    parser.add_argument("--no-bm25",       action="store_true", help="关闭 BM25（消融实验用）")
    parser.add_argument("--no-rerank",     action="store_true", help="关闭 Rerank（消融实验用）")
    args = parser.parse_args()

    pipeline = RAGPipeline(
        use_bm25          = not args.no_bm25,
        use_rerank        = not args.no_rerank,
        use_query_rewrite = args.query_rewrite,
    )

    filter_meta = {}
    if args.book:    filter_meta["book"]    = args.book
    if args.section: filter_meta["section"] = args.section
    if not filter_meta: filter_meta = None

    def print_result(q: str, result: dict):
        print(f"\n{'='*60}")
        print(f"问题：{q}")
        print(f"{'='*60}")
        print(f"\n{result['answer']}")
        if result["citations"]:
            print("\n── 来源 ──")
            for c in result["citations"]:
                print(f"  {c['source']}")

    if args.query:
        result = pipeline.query(args.query, filter_meta=filter_meta, verbose=True)
        print_result(args.query, result)
    else:
        print("书籍 RAG 问答系统（原生版）")
        print(f"LLM：{LLM_MODEL}  |  向量库：{INDEX_PATH}")
        print("输入 'exit' 退出，'mode' 查看当前配置\n")
        while True:
            try:
                q = input("问题：").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q:
                continue
            if q.lower() == "exit":
                break
            if q.lower() == "mode":
                print(f"BM25={'on' if pipeline.use_bm25 else 'off'}  "
                      f"Rerank={'on' if pipeline.use_rerank else 'off'}  "
                      f"QueryRewrite={'on' if pipeline.use_query_rewrite else 'off'}")
                continue
            result = pipeline.query(q, filter_meta=filter_meta, verbose=True)
            print_result(q, result)


if __name__ == "__main__":
    main()
