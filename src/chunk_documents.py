"""
文档分块：把解析结果按语义边界切块，供后续向量化检索。

数据源（data/parsed/manifest.json）：
  PDF  → <book>/auto/<stem>_content_list.json（结构化块，含 page_idx）
  HTML → <book>/<book>.md（含 <!-- source: ... --> 章节标记）

策略（STRATEGY 切换）：
  fixed         固定大小分块（会切断句子/表格，丢失页码章节）
  semantic      语义分块：标题强制切块并作下一块前缀；表格单独成块；正文累积到上限
  hierarchical  层级分块：父块提供上下文，子块用于精确检索

ENABLE_IMAGE_CAPTION 开启时，图表/图片块经 ImageCaptioner（qwen3-vl-flash）
生成中文描述并入正文；图片路径仍记入所在块的 images 元数据。

输出：
  data/chunks/<book>_<strategy>.json   每本书的分块
  data/chunks/all_<strategy>.json      全部分块的合并（供 build_index.py 建索引）
"""

import re
import json
import uuid
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR   = Path(__file__).parent.parent
PARSED_DIR = BASE_DIR / "data" / "parsed"
CHUNKS_DIR = BASE_DIR / "data" / "chunks"
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

STRATEGY       = "semantic"   # "fixed" | "semantic" | "hierarchical"
MAX_CHUNK_SIZE = 800          # 语义分块
MIN_CHUNK_SIZE = 100
FIXED_SIZE     = 500          # 固定大小分块
FIXED_OVERLAP  = 50
PARENT_SIZE    = 2000         # 层级分块
CHILD_SIZE     = 400
HIER_OVERLAP   = 50

# 图片 caption 开关：开启后 chart/image 块会调用 ImageCaptioner 生成描述并入正文
ENABLE_IMAGE_CAPTION = True

_IMAGE_CAPTION_MODEL = "qwen3-vl-flash"
_DASHSCOPE_URL       = "https://dashscope.aliyuncs.com/compatible-mode/v1"   # 与 rag_pipeline.py 一致
_CAPTION_CTX_LIMIT   = 200    # 图片前/后文各截断长度（字符）

_MIME_BY_SUFFIX = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".png": "image/png",  ".webp": "image/webp",
}

IMAGE_CAPTION_PROMPT = """你是机器学习教材的图表解析器。

请分析这张教材中的图片，并生成适合知识库检索的技术描述。

要求：
1. 说明图片展示的机器学习/统计学概念。
2. 描述图中的关键结构、变量、数据流或算法流程。
3. 如果包含公式、符号、坐标轴，请尽可能准确地说明。
4. 如果是模型结构图，说明各层/模块之间的连接关系。
5. 如果是统计图，说明横轴、纵轴、主要趋势和比较关系。
6. 不要描述无关的颜色、装饰等视觉信息。
7. 输出一段简洁但信息密度较高的中文技术描述。"""

_NOISE_TYPES = {"header", "footer", "page_number", "page_footnote"}
_IMAGE_TYPES = {"chart", "image"}


# ── 图片 caption（DashScope qwen3-vl-flash，OpenAI 兼容接口）─────────────────

def _image_to_data_url(img_path: str) -> str:
    """本地图片 → data URL（DashScope VL 本地图只能走 data URL）。失败返回 ""。"""
    try:
        path = Path(img_path)
        if not path.exists():
            logger.warning(f"  图片不存在，跳过 caption: {img_path}")
            return ""
        import base64
        mime = _MIME_BY_SUFFIX.get(path.suffix.lower(), "image/jpeg")
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{b64}"
    except Exception as e:
        logger.warning(f"  读取图片失败: {img_path}: {e}")
        return ""


def _image_caption_context(prev_text: str, after_text: str) -> str:
    """把图片前/后文拼成给 VLM 的分节上下文（空段省略）。"""
    parts = []
    if prev_text:
        parts.append(f"【图片前文】\n{prev_text}")
    if after_text:
        parts.append(f"【图片后文】\n{after_text}")
    return "\n\n".join(parts)


class ImageCaptioner:
    """用 DashScope qwen3-vl-flash（OpenAI 兼容接口）为教材图片生成中文技术描述。

    本地图片转 base64 data URL；非流式、关闭 thinking。
    失败（无 key / 网络错误 / 空返回）返回 "" 并记 warning，由调用方跳过该图。
    """

    def __init__(self, model: str = _IMAGE_CAPTION_MODEL):
        self.model = model
        self._client = None            # 懒加载：不设 key 时构造不报错

    def _get_client(self):
        if self._client is None:
            api_key = os.getenv("DASHSCOPE_API_KEY")
            if not api_key:
                raise EnvironmentError("未设置 DASHSCOPE_API_KEY，无法生成图片描述")
            from openai import OpenAI
            self._client = OpenAI(api_key=api_key, base_url=_DASHSCOPE_URL)
        return self._client

    def caption(self, img_path: str, context: str = "") -> str:
        """给图片生成中文技术描述；失败返回 ""（由调用方跳过该图）。"""
        data_url = _image_to_data_url(img_path)
        if not data_url:
            return ""
        try:
            client = self._get_client()
            text = IMAGE_CAPTION_PROMPT + (f"\n\n{context}" if context else "")
            resp = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text",      "text": text},
                ]}],
                temperature=0,
                extra_body={"enable_thinking": False},
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            logger.warning(f"  图片 caption 失败，跳过: {img_path}: {e}")
            return ""


# ── 块内容提取 ────────────────────────────────────────────────────────────────

def _table_body_to_md(html: str) -> str:
    """MinerU 的 table_body 是 HTML 片段，转成 markdown 表格。"""
    if not html or "<table" not in html:
        return ""
    from bs4 import BeautifulSoup
    table = BeautifulSoup(html, "lxml").find("table")
    if not table:
        return ""
    rows = []
    for tr in table.find_all("tr"):
        cells = [td.get_text(" ", strip=True) for td in tr.find_all(["td", "th"])]
        if cells:
            rows.append(cells)
    if not rows:
        return ""
    header = rows[0]
    lines = ["| " + " | ".join(header) + " |",
             "| " + " | ".join(["---"] * len(header)) + " |"]
    for row in rows[1:]:
        while len(row) < len(header):
            row.append("")
        lines.append("| " + " | ".join(row[:len(header)]) + " |")
    return "\n".join(lines)


def _block_content(block: dict) -> str:
    """按块类型取正文（噪声类型返回空串）。"""
    btype = block.get("type")
    if btype in ("text", "aside_text", "equation"):
        return block.get("text", "")
    if btype == "table":
        return _table_body_to_md(block.get("table_body", ""))
    if btype == "list":
        return "\n".join("- " + item for item in block.get("list_items", []))
    return ""


# ── PDF → 块 ──────────────────────────────────────────────────────────────────

def _load_pdf_blocks(book: dict, captioner: ImageCaptioner):
    """读 content_list.json，归一化为块列表，并返回每页图片路径。"""
    cl_rel = book["outputs"]["content_list"][0]
    cl_path = PARSED_DIR / cl_rel
    if not cl_path.exists():
        logger.warning(f"  content_list 不存在: {cl_path}")
        return [], {}, None

    method = Path(cl_rel).parent.name  # auto / txt / ocr
    blocks = json.loads(cl_path.read_text(encoding="utf-8"))

    out = []
    images_by_page: dict[int, list[str]] = {}
    section_stack: dict[int, str] = {}

    prev_text = ""                            # 最近一个有内容的非图片块正文（截断，作 caption 前文）
    pending_imgs: list[tuple[str, int]] = []  # 连续图片组 [(绝对路径, 页码)]

    def _flush_imgs(after: str) -> None:
        """把累计的连续图片组一次性出 caption；后文=after（可为空串）。"""
        nonlocal pending_imgs
        if not pending_imgs:
            return
        for img_abs, img_page in pending_imgs:
            cap = captioner.caption(img_abs, _image_caption_context(prev_text, after))
            if cap:
                out.append({"block_type": "text", "content": cap,
                            "page_num": img_page,
                            "section_path": _section_path(section_stack)})
        pending_imgs = []

    for b in blocks:
        btype = b.get("type")
        page = (b.get("page_idx") or 0) + 1

        if btype in _NOISE_TYPES:
            continue

        if btype in _IMAGE_TYPES:
            img = b.get("img_path", "")
            if img:
                images_by_page.setdefault(page, []).append(img)   # 元数据始终记录，与开关无关
                if ENABLE_IMAGE_CAPTION:
                    pending_imgs.append((str(cl_path.parent / img), page))   # 连续图片只进组，不立即出 caption
            continue

        content = _block_content(b)
        if not content.strip():
            continue

        # 遇到下一个有内容的非图片块：先给之前累计的图片组出 caption，后文=当前块开头
        _flush_imgs(content.strip()[:_CAPTION_CTX_LIMIT])

        # 标题：更新章节路径
        if btype == "text" and b.get("text_level") is not None:
            level = int(b["text_level"])
            section_stack = {k: v for k, v in section_stack.items() if k <= level}
            section_stack[level] = content.strip()
            block_type = "title"
        elif btype == "equation":
            block_type = "equation"
        elif btype == "table":
            block_type = "table"
        else:
            block_type = "text"

        out.append({
            "block_type": block_type,
            "content": content,
            "page_num": page,
            "section_path": _section_path(section_stack),
        })
        prev_text = content.strip()[-_CAPTION_CTX_LIMIT:]   # 取末尾，最贴近下一张图

    # 循环结束：flush 尾部图片组（后文为空）
    _flush_imgs("")

    return out, images_by_page, method


def _section_path(stack: dict[int, str]) -> list[str]:
    return [stack[i] for i in sorted(stack)]


# ── HTML → 块 ─────────────────────────────────────────────────────────────────

def _load_html_blocks(book: dict):
    """读合并后的 .md，按标题/代码块/表格/段落切分成块。"""
    md_rel = book["outputs"]["markdown"][0]
    md_path = PARSED_DIR / md_rel
    if not md_path.exists():
        logger.warning(f"  markdown 不存在: {md_path}")
        return [], {}

    text = md_path.read_text(encoding="utf-8", errors="ignore")
    parts = re.split(r"<!-- source: (.+?) -->", text)

    out = []
    for i in range(1, len(parts), 2):
        body = parts[i + 1] if i + 1 < len(parts) else ""
        out.extend(_md_to_blocks(body))
    return out, {}


def _md_to_blocks(md_text: str) -> list[dict]:
    """把一段 markdown 转成块（标题 / 代码 / 表格 / 段落）。"""
    blocks = []
    section_stack: list[str] = []
    lines = md_text.splitlines()
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
            continue

        # 标题
        m = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if m:
            level = len(m.group(1))
            title = m.group(2).strip()
            section_stack = section_stack[:level - 1] + [title]
            blocks.append({"block_type": "title", "content": title,
                           "page_num": None, "section_path": list(section_stack)})
            i += 1
            continue

        # 代码块（含内部空行，整体保留）
        if stripped.startswith("```"):
            buf = [line]
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                buf.append(lines[i])
                i += 1
            if i < len(lines):
                buf.append(lines[i])
                i += 1
            blocks.append({"block_type": "text", "content": "\n".join(buf),
                           "page_num": None, "section_path": list(section_stack)})
            continue

        # 表格（| 开头且含分隔行）
        if stripped.startswith("|"):
            buf = [line]
            i += 1
            while i < len(lines) and lines[i].strip().startswith("|"):
                buf.append(lines[i])
                i += 1
            content = "\n".join(buf)
            bt = "table" if "---" in content else "text"
            blocks.append({"block_type": bt, "content": content,
                           "page_num": None, "section_path": list(section_stack)})
            continue

        # 段落（累计到空行）
        buf = [line]
        i += 1
        while i < len(lines) and lines[i].strip():
            buf.append(lines[i])
            i += 1
        blocks.append({"block_type": "text", "content": "\n".join(buf).strip(),
                       "page_num": None, "section_path": list(section_stack)})

    return blocks


# ── 分块策略 ──────────────────────────────────────────────────────────────────

def chunk_fixed(blocks: list[dict]):
    """固定大小分块：全文拼接后按 FIXED_SIZE 滑动切，带重叠，丢失页码/章节信息。"""
    full_text = "\n\n".join(b["content"].strip() for b in blocks if b["content"].strip())
    start = 0
    while start < len(full_text):
        yield {
            "content": full_text[start:start + FIXED_SIZE],
            "metadata": {"page_num": None, "page_end": None, "section": "", "block_types": ["text"]},
        }
        start += FIXED_SIZE - FIXED_OVERLAP


def chunk_hierarchical(blocks: list[dict]):
    """层级分块：先切父块（大上下文），父块内再切子块（精确检索），子块带 parent_id。"""
    full_text = "\n\n".join(b["content"].strip() for b in blocks if b["content"].strip())
    start = 0
    while start < len(full_text):
        parent_id = str(uuid.uuid4())[:8]
        parent_content = full_text[start:start + PARENT_SIZE]

        c_start = 0
        while c_start < len(parent_content):
            yield {
                "content": parent_content[c_start:c_start + CHILD_SIZE],
                "metadata": {
                    "page_num": None, "page_end": None, "section": "",
                    "block_types": ["text"],
                    "parent_id": parent_id,
                    "parent_content": parent_content,
                },
            }
            c_start += CHILD_SIZE - HIER_OVERLAP

        start += PARENT_SIZE - HIER_OVERLAP


def _split_long_text(text: str, max_size: int):
    """把超长文本按行边界拆成约 max_size 的子块（单行超长则硬切）。"""
    out = []
    buf = ""
    for line in re.split(r"\n+", text):
        line = line.strip()
        if not line:
            continue
        if buf and len(buf) + len(line) + 1 > max_size:
            out.append(buf)
            buf = line
        else:
            buf = f"{buf}\n{line}" if buf else line
        while len(buf) > max_size:
            out.append(buf[:max_size])
            buf = buf[max_size:]
    if buf:
        # 尾部过短就并入上一块，避免产生无意义的碎块
        if out and len(buf) < max_size // 4:
            out[-1] = out[-1] + "\n" + buf
        else:
            out.append(buf)
    return out


def chunk_semantic(blocks: list[dict]):
    """按结构切块：标题强制切块并作下一块前缀；表格单独成块；正文累积到上限。"""
    buffer: list[dict] = []
    buf_len = 0

    def flush_with_meta():
        nonlocal buffer, buf_len
        if not buffer:
            return None
        content = "\n\n".join(b["content"].strip() for b in buffer if b["content"].strip())
        if not content:
            buffer = []
            buf_len = 0
            return None
        meta = {
            "page_num":   buffer[0]["page_num"],
            "page_end":   buffer[-1]["page_num"],
            "section":    " > ".join(buffer[0]["section_path"]) if buffer[0]["section_path"] else "",
            "block_types": sorted({b["block_type"] for b in buffer}),
        }
        buffer = []
        buf_len = 0
        return {"content": content, "metadata": meta}

    def block_chunk(block: dict, content: str = None) -> dict:
        return {
            "content": content if content is not None else block["content"],
            "metadata": {
                "page_num":   block["page_num"],
                "page_end":   block["page_num"],
                "section":    " > ".join(block["section_path"]),
                "block_types": [block["block_type"]],
            },
        }

    for block in blocks:
        btype = block["block_type"]
        blen = len(block["content"])

        # 标题：先把前面的正文 flush，标题本身作为下一块的前缀
        if btype == "title":
            if buffer:
                r = flush_with_meta()
                if r and len(r["content"]) >= MIN_CHUNK_SIZE:
                    yield r

        # 表格：单独成块，不与文字混合；超大表格按行拆
        if btype == "table":
            if buffer:
                r = flush_with_meta()
                if r and len(r["content"]) >= MIN_CHUNK_SIZE:
                    yield r
            for sub in _split_long_text(block["content"], MAX_CHUNK_SIZE):
                yield block_chunk(block, sub)
            continue

        # 正文累积，超过上限先 flush
        if buf_len + blen > MAX_CHUNK_SIZE and buffer:
            r = flush_with_meta()
            if r and len(r["content"]) >= MIN_CHUNK_SIZE:
                yield r

        # 单个块就超过上限：按行拆开
        if blen > MAX_CHUNK_SIZE:
            for sub in _split_long_text(block["content"], MAX_CHUNK_SIZE):
                yield block_chunk(block, sub)
            continue

        buffer.append(block)
        buf_len += blen

    r = flush_with_meta()
    if r and len(r["content"]) >= MIN_CHUNK_SIZE:
        yield r


# ── 主流程 ────────────────────────────────────────────────────────────────────

def _page_images(page_start, page_end, images_by_page) -> list[str]:
    if not images_by_page or not page_start or not page_end:
        return []
    imgs = []
    for pg in range(page_start, page_end + 1):
        imgs.extend(images_by_page.get(pg, []))
    return sorted(set(imgs))


def process_book(book: dict) -> list[dict]:
    """把一本书解析成块并语义分块，补齐元信息。"""
    if book["type"] == "pdf":
        blocks, images_by_page, method = _load_pdf_blocks(book, ImageCaptioner())
        source_file = book["outputs"]["content_list"][0]
    elif book["type"] == "html":
        blocks, images_by_page = _load_html_blocks(book)
        method = None
        source_file = book["outputs"]["markdown"][0]
    else:
        logger.warning(f"未知类型，跳过: {book['type']}")
        return []

    generators = {
        "fixed":         chunk_fixed,
        "semantic":      chunk_semantic,
        "hierarchical":  chunk_hierarchical,
    }
    if STRATEGY not in generators:
        raise ValueError(f"未知策略: {STRATEGY}")

    chunks = []
    for idx, chunk in enumerate(generators[STRATEGY](blocks)):
        meta = chunk["metadata"]
        meta.update({
            "book":        book["name"],
            "title":       book["title"],
            "method":      method,
            "strategy":    STRATEGY,
            "source_file": source_file,
            "images":      _page_images(meta.get("page_num"), meta.get("page_end"), images_by_page),
        })
        chunk["chunk_id"] = f"{book['name']}-{idx:06d}"
        chunks.append(chunk)
    return chunks


def main():
    manifest_path = PARSED_DIR / "manifest.json"
    if not manifest_path.exists():
        logger.error("未找到 data/parsed/manifest.json，请先运行 parse_books.py")
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    all_chunks = []
    for book in manifest:
        chunks = process_book(book)
        logger.info(f"{book['title']}: {len(chunks)} 个 chunk")
        all_chunks.extend(chunks)

        out_path = CHUNKS_DIR / f"{book['name']}_{STRATEGY}.json"
        out_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")

    combined = CHUNKS_DIR / f"all_{STRATEGY}.json"
    combined.write_text(json.dumps(all_chunks, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info(f"\n合并完成：共 {len(all_chunks)} 个 chunk → {combined}")
    avg = sum(len(c["content"]) for c in all_chunks) / max(len(all_chunks), 1)
    logger.info(f"平均 chunk 长度：{avg:.0f} 字符")
    for t in ("table", "equation", "title"):
        n = sum(1 for c in all_chunks if t in c["metadata"]["block_types"])
        logger.info(f"含 {t} 的块：{n}")


if __name__ == "__main__":
    main()
