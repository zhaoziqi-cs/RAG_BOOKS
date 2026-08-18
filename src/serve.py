"""
书籍 RAG 问答 HTTP 服务（FastAPI）

接口：
  /            — 本地问答 Web 页面（static/index.html）
  /query       — 标准问答，返回答案 + 引用
  /query/debug — 调试接口，逐步返回每个检索阶段的中间结果
  /health      — 健康检查
"""

import os
import importlib.util
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_PIPELINE_PATH = Path(__file__).parent / "rag_pipeline.py"
_STATIC_DIR    = Path(__file__).parent / "static"


# ── 动态加载 rag_pipeline（importlib 方式，保持与启动目录无关）────────────────

def _load_pipeline_module():
    spec   = importlib.util.spec_from_file_location("rag_pipeline", _PIPELINE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ── 全局状态 ──────────────────────────────────────────────────────────────────

_module  = None   # 持有模块引用，供 debug 端点访问模块级函数
pipeline = None   # RAGPipeline 实例


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _module, pipeline
    logger.info("服务启动，初始化 RAG Pipeline...")
    _module  = _load_pipeline_module()
    pipeline = _module.RAGPipeline(
        use_bm25          = True,
        use_rerank        = True,    # 本地 bge-reranker-base，重排可用
        use_query_rewrite = True,
    )
    logger.info(f"Pipeline 初始化完成（query_rewrite={pipeline.use_query_rewrite}），开始接受请求")
    yield
    logger.info("服务关闭")


# ── FastAPI 应用 ──────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "书籍 RAG 问答服务",
    description = "本地 bge-m3 + FAISS + BM25 混合检索 + DashScope qwen-plus，含调试接口",
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


# ── 数据模型 ──────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question:   str           = Field(...,  example="What is gradient descent?")
    book:       Optional[str] = Field(None, example="dive_into_deep_learning")
    section:    Optional[str] = Field(None, example="Softmax")

class Citation(BaseModel):
    index:    int
    source:   str
    chunk_id: str

class QueryResponse(BaseModel):
    answer:    str
    citations: list[Citation]

# ── Debug 专用模型 ────────────────────────────────────────────────────────────

class VecChunk(BaseModel):
    rank:            int
    vec_score:       float
    source:          str
    content_preview: str

class BM25Chunk(BaseModel):
    rank:            int
    bm25_score:      float
    source:          str
    content_preview: str

class RRFChunk(BaseModel):
    rank:            int
    rrf_score:       float
    vec_rank:        Optional[int]
    bm25_rank:       Optional[int]
    source:          str
    content_preview: str

class RerankChunk(BaseModel):
    rank:            int
    rerank_score:    float
    rrf_rank:        Optional[int]   # 重排前在 RRF 中的名次，用于展示名次变化
    source:          str
    content_preview: str

class ContextChunk(BaseModel):
    index:   int
    source:  str
    content: str          # 完整内容，用于展示"输入 LLM 的原文"

class DebugResponse(BaseModel):
    question:       str
    vec_results:    list[VecChunk]
    bm25_results:   list[BM25Chunk]
    rrf_results:    list[RRFChunk]
    rerank_results: list[RerankChunk]
    context_chunks: list[ContextChunk]
    answer:         str
    citations:      list[Citation]


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def _build_source(item: dict) -> str:
    """把 chunk metadata 格式化为可读来源字符串（与 rag_pipeline.build_context 一致）。"""
    title = item.get("title", "") or item.get("book", "")
    s = f"《{title}》"
    section = item.get("section", "")
    if section:
        # 只保留最后两级，避免过长
        parts = section.split(" > ")
        s += " · " + " > ".join(parts[-2:])
    page = item.get("page_num")
    if page:
        s += f" · 第{page}页"
    return s

def _preview(text: str, n: int = 150) -> str:
    text = text.strip()
    return text[:n] + "…" if len(text) > n else text

def _filter_meta(req: QueryRequest) -> Optional[dict]:
    fm = {}
    if req.book:    fm["book"]    = req.book
    if req.section: fm["section"] = req.section
    return fm or None


# ── 接口 ──────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def index():
    """返回静态问答页面。"""
    return FileResponse(_STATIC_DIR / "index.html")


@app.get("/health", summary="健康检查")
def health():
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline 尚未初始化")
    return {"status": "ok", "pipeline_ready": True}


@app.post("/query", response_model=QueryResponse, summary="标准问答")
def query(req: QueryRequest):
    """标准问答接口，返回最终答案和来源引用。"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline 尚未初始化")
    try:
        result = pipeline.query(req.question, filter_meta=_filter_meta(req), verbose=True)
    except Exception as e:
        logger.error(f"Pipeline 异常: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    return QueryResponse(
        answer    = result["answer"],
        citations = [Citation(**c) for c in result["citations"]],
    )


@app.post("/query/debug", response_model=DebugResponse, summary="调试：逐步返回中间结果")
def query_debug(req: QueryRequest):
    """
    调试接口，逐步执行 RAG 流水线并返回每步的中间结果：

    1. 向量检索（vec_results）：FAISS 召回 Top-10，含余弦相似度分数
    2. BM25 检索（bm25_results）：关键词召回 Top-10，含 BM25 分数
    3. RRF 融合（rrf_results）：两路合并后 Top-5，标注每条来自哪一路
    4. Rerank 精排（rerank_results）：CrossEncoder 对候选重排，标注名次变化
    5. 输入 LLM 的上下文（context_chunks）：重排后 Top-4 的完整文本
    6. 生成答案（answer + citations）
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline 尚未初始化")

    TOP_K       = _module.TOP_K_RETRIEVE   # 10
    TOP_K_FINAL = _module.TOP_K_RERANK     # 4
    fm          = _filter_meta(req)

    try:
        # 查询改写（与 pipeline.query 保持一致）
        retrieval_query = _module.rewrite_query(req.question, pipeline.client) if pipeline.use_query_rewrite else req.question

        # ① 向量检索
        vec_results  = pipeline.vec_store.search(retrieval_query, TOP_K, fm)
        vec_rank_map = {item["chunk_id"]: rank for rank, item in enumerate(vec_results, 1)}

        # ② BM25 检索（同样受 filter_meta 限制，两路一致过滤后融合）
        bm25_results  = pipeline.bm25_store.search(retrieval_query, TOP_K, fm) if pipeline.bm25_store else []
        bm25_rank_map = {item["chunk_id"]: rank for rank, item in enumerate(bm25_results, 1)}

        # ③ RRF 融合
        if bm25_results:
            candidates = _module.reciprocal_rank_fusion(vec_results, bm25_results)
        else:
            candidates = vec_results

        # ④ Rerank 精排（用原始问题打分，与 pipeline.query 保持一致）
        rrf_rank_map = {item["chunk_id"]: rank for rank, item in enumerate(candidates, 1)}
        if pipeline.use_rerank:
            reranked = _module.rerank(req.question, list(candidates), max(len(candidates), 1))
            final    = reranked[:TOP_K_FINAL]
        else:
            reranked = candidates
            final    = candidates[:TOP_K_FINAL]

        # ⑤ 组装上下文
        context, cits = _module.build_context(final)

        # ⑥ LLM 生成
        answer = _module.call_llm(req.question, context, pipeline.client)

    except Exception as e:
        logger.error(f"Debug Pipeline 异常: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    return DebugResponse(
        question = req.question,
        vec_results = [
            VecChunk(
                rank=i+1,
                vec_score=item.get("vec_score", 0.0),
                source=_build_source(item),
                content_preview=_preview(item["content"]),
            ) for i, item in enumerate(vec_results[:5])
        ],
        bm25_results = [
            BM25Chunk(
                rank=i+1,
                bm25_score=item.get("bm25_score", 0.0),
                source=_build_source(item),
                content_preview=_preview(item["content"]),
            ) for i, item in enumerate(bm25_results[:5])
        ],
        rrf_results = [
            RRFChunk(
                rank=i+1,
                rrf_score=item.get("rrf_score", 0.0),
                vec_rank=vec_rank_map.get(item["chunk_id"]),
                bm25_rank=bm25_rank_map.get(item["chunk_id"]),
                source=_build_source(item),
                content_preview=_preview(item["content"]),
            ) for i, item in enumerate(candidates[:5])
        ],
        rerank_results = [
            RerankChunk(
                rank=i+1,
                rerank_score=item.get("rerank_score", 0.0),
                rrf_rank=rrf_rank_map.get(item["chunk_id"]),
                source=_build_source(item),
                content_preview=_preview(item["content"]),
            ) for i, item in enumerate(reranked[:8])
        ],
        context_chunks = [
            ContextChunk(
                index=i+1,
                source=_build_source(item),
                content=item.get("parent_content") or item["content"],
            ) for i, item in enumerate(final)
        ],
        answer    = answer,
        citations = [Citation(**c) for c in cits],
    )
