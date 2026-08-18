"""
RAG 问答链（LangChain LCEL 版）

加载 vectorstore/faiss_lc 向量库，构建 LCEL 问答链（检索 → 上下文格式化 → Prompt → LLM）。
支持 --book / --section 元数据过滤；--with-sources 附带来源文档片段。

LLM：DashScope qwen-plus（OpenAI 兼容接口）
Embedding：本地 BAAI/bge-m3（与 build_index_lc.py 一致）
"""

import os
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR        = Path(__file__).parent.parent
VECTORSTORE_DIR = BASE_DIR / "vectorstore" / "faiss_lc"
MODELS_DIR      = BASE_DIR / "models"
BGE_MODEL_PATH  = MODELS_DIR / "bge-m3"

DASHSCOPE_URL   = "https://dashscope.aliyuncs.com/compatible-mode/v1"
LLM_MODEL       = "qwen-plus"

SYSTEM_PROMPT = """你是一个专业的技术书籍问答助手，根据给定的机器学习 / 深度学习 / 数据科学书籍资料回答问题。

回答规则：
1. 只根据【参考资料】中的内容回答，不得引用或编造资料外的内容
2. 若参考资料不足以支撑回答，直接说"根据提供的资料无法回答此问题"
3. 引用具体内容时，在句末标注来源编号，如：梯度下降通过迭代更新参数来最小化损失函数[1]
4. 涉及公式、代码、定义时要忠实于原文，不要凭记忆改写
5. 用与用户问题相同的语言回答（中文问题用中文回答）
6. 回答简洁，重点突出，避免无关废话"""


# ── 组件初始化 ────────────────────────────────────────────────────────────────

def _device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def get_llm():
    """
    ChatOpenAI 指向 DashScope 的 OpenAI 兼容接口。
    """
    from langchain_openai import ChatOpenAI

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise EnvironmentError("请设置环境变量 DASHSCOPE_API_KEY")

    return ChatOpenAI(
        model=LLM_MODEL,
        openai_api_key=api_key,
        openai_api_base=DASHSCOPE_URL,
        temperature=0.1,
    )


def get_embeddings():
    """用与 build_index_lc.py 相同的 embedding 模型（须与建索引一致，否则向量空间不匹配）。"""
    from langchain_huggingface import HuggingFaceEmbeddings

    model_path = str(BGE_MODEL_PATH) if BGE_MODEL_PATH.exists() else "BAAI/bge-m3"
    return HuggingFaceEmbeddings(
        model_name=model_path,
        cache_folder=str(MODELS_DIR),
        model_kwargs={"device": _device()},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_vectorstore(embeddings):
    """加载已构建的 FAISS 向量库。"""
    from langchain_community.vectorstores import FAISS

    if not VECTORSTORE_DIR.exists():
        raise FileNotFoundError(
            f"向量库不存在: {VECTORSTORE_DIR}\n"
            "请先运行: python src_langchain/build_index_lc.py"
        )
    return FAISS.load_local(
        str(VECTORSTORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True,   # FAISS 本地加载需要此标志
    )


# ── 元数据过滤（book/section 子串匹配）──

def _match_filter(meta: dict, filter_meta: dict) -> bool:
    """book 同时匹配 book / title 字段（大小写不敏感、子串匹配），section 子串匹配。"""
    if "book" in filter_meta:
        q = filter_meta["book"].lower()
        book  = str(meta.get("book", "")).lower()
        title = str(meta.get("title", "")).lower()
        if q not in book and q not in title:
            return False
    if "section" in filter_meta:
        q = filter_meta["section"].lower()
        if q not in str(meta.get("section", "")).lower():
            return False
    return True


# ── LCEL 链构建 ───────────────────────────────────────────────────────────────

def make_retriever(vectorstore, filter_meta=None, k=4):
    """
    自定义检索器：先多召回再按 book/section 子串过滤，最后取 Top-k。
    LangChain FAISS 自带 metadata 过滤为精确匹配，此处改在 Python 侧做子串过滤
    （多召回 k*5 条，过滤后截断到 k）。
    """
    from langchain_core.runnables import RunnableLambda

    def retrieve(query: str):
        kk = k if filter_meta is None else k * 5
        docs = vectorstore.similarity_search(query, k=kk)
        if filter_meta:
            docs = [d for d in docs if _match_filter(d.metadata, filter_meta)]
        return docs[:k]

    return RunnableLambda(retrieve)


def format_docs(docs) -> str:
    """把 Document 列表格式化为带编号 + 书籍来源的上下文字符串。"""
    parts = []
    for i, doc in enumerate(docs, 1):
        meta    = doc.metadata
        title   = meta.get("title", "") or meta.get("book", "")
        label   = f"[{i}] 《{title}》"
        section = meta.get("section", "")
        if section:
            label += f" · {section}"
        page = meta.get("page_num")
        if page:
            label += f" · 第{page}页"
        parts.append(f"{label}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def build_chain(vectorstore, filter_meta=None):
    """
    构建标准 RAG 链：并行执行 retriever 与 question 透传，
    组装 {context, question} → prompt → llm → StrOutputParser。
    """
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser

    llm       = get_llm()
    retriever = make_retriever(vectorstore, filter_meta, k=4)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "【参考资料】\n{context}\n\n【问题】\n{question}\n\n请根据参考资料回答，并标注来源编号。"),
    ])

    # 用 | 串联各组件
    chain = (
        {
            "context":  retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever


def build_chain_with_sources(vectorstore, filter_meta=None):
    """
    返回答案 + 来源文档的版本：用 RunnableParallel 并行执行检索与问题透传。
    """
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough, RunnableParallel
    from langchain_core.output_parsers import StrOutputParser

    llm       = get_llm()
    retriever = make_retriever(vectorstore, filter_meta, k=4)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "【参考资料】\n{context}\n\n【问题】\n{question}"),
    ])

    # 并行：一路检索原始 docs，另一路用格式化 docs 生成答案
    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=lambda x: format_docs(x["context"]))
        | prompt | llm | StrOutputParser()
    )

    chain_with_sources = RunnableParallel(
        {
            "context":  retriever,
            "question": RunnablePassthrough(),
        }
    ).assign(answer=rag_chain_from_docs)

    return chain_with_sources


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="书籍 RAG 问答（LangChain LCEL 版）")
    parser.add_argument("--query",        type=str, default=None)
    parser.add_argument("--book",         type=str, default=None, help="限定书籍（书名或 name，模糊匹配）")
    parser.add_argument("--section",      type=str, default=None, help="限定章节（模糊匹配）")
    parser.add_argument("--with-sources", action="store_true", help="输出结果时附带来源文档片段")
    args = parser.parse_args()

    filter_meta = {}
    if args.book:    filter_meta["book"]    = args.book
    if args.section: filter_meta["section"] = args.section
    if not filter_meta: filter_meta = None

    logger.info("加载 embedding 模型...")
    embeddings  = get_embeddings()
    logger.info("加载向量库...")
    vectorstore = get_vectorstore(embeddings)

    if args.with_sources:
        chain = build_chain_with_sources(vectorstore, filter_meta)
    else:
        chain, _ = build_chain(vectorstore, filter_meta)

    def run_query(question: str):
        print(f"\n{'='*60}")
        print(f"问题：{question}")
        print(f"{'='*60}")

        if args.with_sources:
            result  = chain.invoke(question)
            answer  = result["answer"]
            sources = result["context"]
            print(f"\n{answer}")
            print("\n── 来源文档片段 ──")
            for i, doc in enumerate(sources, 1):
                meta  = doc.metadata
                title = meta.get("title", "") or meta.get("book", "")
                label = f"[{i}] 《{title}》"
                if meta.get("section"):    label += f" · {meta['section']}"
                if meta.get("page_num"):   label += f" · 第{meta['page_num']}页"
                print(label)
                print(f"    {doc.page_content[:120]}...")
        else:
            answer = chain.invoke(question)
            print(f"\n{answer}")

    if args.query:
        run_query(args.query)
    else:
        print("书籍 RAG 问答系统（LangChain LCEL 版）")
        print(f"模型：{LLM_MODEL}  |  向量库：{VECTORSTORE_DIR}")
        print("输入 'exit' 退出\n")
        while True:
            try:
                q = input("问题：").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q or q.lower() == "exit":
                break
            run_query(q)


if __name__ == "__main__":
    main()
