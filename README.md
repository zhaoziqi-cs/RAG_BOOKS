# 书籍 RAG 问答系统（Books RAG）

一个基于 5 本英文机器学习 / 数据科学经典书籍的检索增强生成（RAG）问答系统。支持中文提问、混合检索（向量 + BM25）、CrossEncoder 重排与相关性阈值过滤；书中图表经视觉语言模型生成文字描述后一并纳入检索。提供命令行、HTTP 服务与本地 Web 问答界面（含检索过程可视化调试）。

## 知识库书籍

| `name`（限定书籍用） | 书名 | 类型 |
|---|---|---|
| `introduction_to_statistical_learning_python` | An Introduction to Statistical Learning (ISLP) | PDF |
| `python_data_science_handbook` | Python Data Science Handbook | HTML（多页） |
| `dive_into_deep_learning` | Dive into Deep Learning (d2l) | PDF |
| `mathematics_for_machine_learning` | Mathematics for Machine Learning | PDF |
| `mathematical_foundations_of_rl` | Mathematical Foundations of Reinforcement Learning | PDF |

书籍来源：*Compendium of free ML reading resources*（见 `data/manifest.json`）。

## 系统架构

**数据侧（离线流水线，每次改动数据源后重跑）：**

```
download_books ──► parse_books ──► chunk_documents ──► build_index
  (PDF / HTML)      (MinerU / BS4)   (语义分块 + 图片描述)   (FAISS 索引)
```

**查询侧（在线流水线）：**

```
                         ┌─────────────────────────────────────────┐
                         │            查询（中文/英文）             │
                         └────────────────────┬────────────────────┘
                                              │ (可选) 查询改写 → 英文
                 ┌────────────────────────────┼────────────────────────────┐
                 ▼                            ▼                            │
        ① 向量检索                    ② BM25 关键词检索                     │
        bge-m3 → FAISS                英文分词 → BM25Okapi                  │
        (1024 维, 余弦)                (词法精确匹配)                       │
                 │                            │                            │
                 └────────────┬───────────────┘                            │
                              ▼                                            │
                        ③ RRF 融合排名                                     │
                        score(d) = Σ 1/(60 + rankᵢ)                        │
                              │                                            │
                              ▼                                            │
                        ④ CrossEncoder Rerank                             │
                        bge-reranker-base 二次精排                         │
                              │                                            │
                              ▼                                            │
                        ⑤ 相关性阈值过滤（低于阈值拒绝回答）                  │
                              │                                            │
                              ▼                                            │
                        ⑥ LLM 生成（DashScope qwen-plus）+ 引用标注           │
```

**核心模型：**

| 用途 | 模型 | 说明 |
|---|---|---|
| Embedding | `BAAI/bge-m3` | 本地，多语言，1024 维，L2 归一化，内积 = 余弦相似度 |
| Rerank | `BAAI/bge-reranker-base` | 本地，CrossEncoder 交叉注意力 |
| 图片描述 | DashScope `qwen3-vl-flash` | OpenAI 兼容接口，为书中图表生成中文技术描述 |
| LLM | DashScope `qwen-plus` | OpenAI 兼容接口，生成答案并标注引用 |

## 目录结构

```
rag_books/
├── src/                        # 原生流水线（手写实现）
│   ├── download_books.py       # ① 下载书籍 → data/raw/ + data/manifest.json
│   ├── parse_books.py          # ② 解析 → data/parsed/（MinerU + BeautifulSoup）
│   ├── chunk_documents.py      # ③ 语义分块 + VLM 图片描述 → data/chunks/all_semantic.json
│   ├── build_index.py          # ④ 建向量索引 → vectorstore/faiss_index.bin + faiss_meta.json
│   ├── rag_pipeline.py         # ⑤ 检索 + 生成（命令行 / 可 import）
│   ├── serve.py                # ⑥ FastAPI 服务 + 流水线可视化调试
│   └── static/index.html       #    Web 问答页面（6 步流水线可视化）
├── src_langchain/              # LangChain LCEL 版（框架封装）
│   ├── download_model.py       # 下载 bge-m3
│   ├── build_index_lc.py       # 建 LangChain FAISS 向量库 → vectorstore/faiss_lc/
│   └── rag_chain_lc.py         # LCEL 链（FAISS 单路检索）
├── data/
│   ├── raw/                    # 原始书籍（PDF / HTML）
│   ├── parsed/                 # 解析产物（Markdown + content_list.json + images/）
│   ├── chunks/                 # 分块结果（all_semantic.json，11,445 个 chunk，含图片描述）
│   └── manifest.json           # 下载清单
├── vectorstore/
│   ├── faiss_index.bin         # FAISS 索引（11,445 条）
│   └── faiss_meta.json         # 向量对应的元数据
└── models/
    ├── bge-m3/                 # Embedding 模型
    └── bge-reranker-base/      # Rerank 模型
```

## 环境准备

```powershell
# 1. 激活 conda 环境（Python 3.12，torch + CUDA）
conda activate nlp

# 2. 安装依赖
pip install faiss-cpu rank_bm25 numpy sentence-transformers openai requests beautifulsoup4 lxml
pip install fastapi uvicorn pydantic
# LangChain 版本额外需要：
pip install langchain langchain-openai langchain-community langchain-huggingface

# 3. 设置 DashScope 密钥（LLM 生成与图片描述需要；embedding / rerank 走本地模型）
$env:DASHSCOPE_API_KEY = "sk-xxx"
```

> PDF 解析依赖 **MinerU**（命令行 `mineru`），需单独安装（首次运行会自动下载 pipeline 模型约 1–2GB）。HTML 书（PDSH）用 BeautifulSoup，无需 MinerU。

## 快速开始（原生流水线）

按顺序执行：

```powershell
conda activate nlp

# ① 下载书籍
python src/download_books.py

# ② 解析（MinerU 解析 PDF / BeautifulSoup 解析 HTML）
python src/parse_books.py

# ③ 语义分块 + VLM 图片描述（图片描述需 DASHSCOPE_API_KEY）
python src/chunk_documents.py

# ④ 建向量索引（bge-m3 + FAISS）
python src/build_index.py

# ⑤ 命令行问答
python src/rag_pipeline.py --query "What is gradient descent?"
python src/rag_pipeline.py --query "什么是梯度下降" --query-rewrite       # 中文 → 英文翻译后检索
python src/rag_pipeline.py --query "How does softmax work?" --book dive_into_deep_learning
```

> 仓库已包含 ①–④ 的完整产物：`data/raw` → `data/parsed` → `data/chunks/all_semantic.json`（11,445 个 chunk）→ `vectorstore/faiss_index.bin`（11,445 条向量）。可直接从 ⑤ 开始。

### 图片描述（`chunk_documents.py`）

`ENABLE_IMAGE_CAPTION = True` 时（默认开启），每个图表 / 图片块经 `ImageCaptioner`（DashScope `qwen3-vl-flash`，OpenAI 兼容接口）生成中文技术描述并并入正文；图片路径记录在对应 chunk 元数据的 `images` 字段。当前知识库共 8,954 条图片引用、覆盖 4,124 个 chunk。

- 关闭该开关：只记录图片路径，不生成描述（不依赖 `DASHSCOPE_API_KEY`）。
- 开关为 False 时，重新运行 ③④ 即可重建无描述的索引。

### `rag_pipeline.py` 参数

| 参数 | 说明 |
|---|---|
| `--query "..."` | 单次问答；省略则进入交互式 |
| `--book <片段>` | 限定书籍（书名或 name，子串匹配） |
| `--section <片段>` | 限定章节（子串匹配） |
| `--query-rewrite` | 开启查询改写（中文问题翻译成英文，多一次 LLM 调用） |
| `--no-bm25` / `--no-rerank` | 关闭 BM25 / Rerank（消融对比） |

## Web 服务

```powershell
$env:DASHSCOPE_API_KEY = "sk-xxx"
cd src
uvicorn serve:app --host 0.0.0.0 --port 8000
# 修改代码后想热重载，加 --reload
```

启动后访问：

| 地址 | 说明 |
|---|---|
| `http://localhost:8000/` | Web 问答页面（6 步流水线可视化） |
| `http://localhost:8000/docs` | Swagger 接口文档（可在线 Try it out） |
| `http://localhost:8000/health` | 健康检查 |

### 接口

| 接口 | 方法 | 说明 |
|---|---|---|
| `/query` | POST | 标准问答，返回 `answer` + `citations` |
| `/query/debug` | POST | 调试，逐步返回向量 / BM25 / RRF / Rerank / 上下文 / 答案 |

请求体（两个接口相同）：

```json
{
  "question": "What is gradient descent?",
  "book":     "dive_into_deep_learning",   // 可选
  "section":  "Softmax"                     // 可选
}
```

> **注意**：`/query`、`/query/debug` 是 **POST** 接口，在浏览器地址栏直接打开（GET）会返回 405。请用网页、`/docs` 或命令行调用：

```powershell
Invoke-RestMethod -Method Post http://localhost:8000/query `
  -ContentType 'application/json' `
  -Body '{"question":"What is gradient descent?"}'
```

## LangChain 版本（`src_langchain/`）

同一知识库的 LangChain LCEL 实现，与原生版共用 `data/chunks/all_semantic.json` 与本地 `bge-m3` embedding：

```powershell
conda activate nlp
# 首次需下载模型（本机 models/bge-m3 已存在，会自动跳过）
python src_langchain/download_model.py

# 建 LangChain 向量库（复用上游语义分块，生成 vectorstore/faiss_lc/）
python src_langchain/build_index_lc.py

# LCEL 链问答
python src_langchain/rag_chain_lc.py --query "What is gradient descent?"
```

与原生版差异：LangChain 版为 **FAISS 单路检索**（无 BM25 / RRF / Rerank），链路用 `|` 操作符声明式串联，扩展 Agent / Memory 更方便；原生版为混合检索，检索质量更高、每步可打印调试。

## 检索与生成原理要点

- **向量检索**：`bge-m3` 多语言模型，查询与语料在同一向量空间，中文查询也能跨语言召回；`IndexFlatIP` + L2 归一化，内积即余弦相似度。
- **图片检索**：分块阶段由 `qwen3-vl-flash` 将图表转为中文描述并入正文，图片内容通过普通文本检索即可命中。
- **BM25**：英文分词（小写 + `[a-z0-9]+`），对精确术语 / 公式符号命中更好；纯中文查询分词为空，需配合 `--query-rewrite` 翻译成英文。
- **RRF**：`score = Σ 1/(60 + rank)` 融合两路排名，互补各自盲区。
- **Rerank**：CrossEncoder 对候选集二次精排，取 Top-4 送入 LLM。
- **阈值过滤**：最高余弦相似度低于 0.25 时拒绝回答（避免编造）。
- **回答语言**：Prompt 要求用与问题相同的语言回答（中文提问返回中文）。

## 可能的改进方向

| 方向 | 说明 | 切入点 |
|---|---|---|
| 增量索引更新 | 新增书籍 / 章节时只对增量部分建索引，避免整库重建 | 目前 `build_index.py` 每次全量重建 11,445 条向量 |
| 检索质量评测 | 引入 RAGAS 等框架，用标注问题集量化检索 / 生成质量 | 现有 `--no-bm25` / `--no-rerank` 支持消融对比 |
| 多轮对话 / 记忆 | 支持追问与上下文记忆 | LangChain 版更易接入 Agent / Memory |
| 流式输出 | Web 端改为 SSE 流式生成，降低等待感 | `serve.py` 当前整段返回 |
| 公式 / 图表专门展示 | 公式 LaTeX 渲染、表格结构化呈现 | 分块时 `equation` / `table` 已单独成块（见 `block_types`） |
| 混合检索调参 | RRF 的 k 值、BM25 与向量权重、阈值 0.25 的自动寻优 | 常量集中在 `rag_pipeline.py` 顶部 |
