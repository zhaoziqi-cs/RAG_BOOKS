"""
书籍解析：把 data/raw 下的书籍解析为结构化 Markdown + JSON（供 RAG 检索与溯源）

PDF 用 MinerU（pipeline 后端）：布局识别 + 公式 LaTeX 化 + 表格识别，
输出 .md 与 _content_list.json（含页码）；原生数字 PDF（非扫描件）走文本层提取。
MinerU 不支持 HTML 输入，HTML 书（Python Data Science Handbook）改用 BeautifulSoup 提取，
标题/段落/列表/代码块/表格 → Markdown，每个章节页打上 <!-- source: ... --> 标记便于溯源。

输出统一到 data/parsed/，并生成 manifest.json 记录每本书的解析产物路径。
"""

import re
import json
import logging
import shutil
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR    = Path(__file__).parent.parent / "data" / "raw"
PARSED_DIR = Path(__file__).parent.parent / "data" / "parsed"
PARSED_DIR.mkdir(parents=True, exist_ok=True)

# MinerU 后端: [pipeline|vlm-engine|hybrid-engine|vlm-http-client|hybrid-http-client]解析后端（默认为 hybrid-engine）
MINERU_BACKEND = "pipeline"
# 测试用：只解析第 [start, end] 页（None 表示全本）。例如 PAGE_RANGE = (0, 20)
PAGE_RANGE = None


# ── PDF 解析（MinerU）─────────────────────────────────────────────────────────

def run_mineru(pdf_path: Path) -> dict:
    """用 MinerU 解析单个 PDF，返回发现的产物路径（相对 PARSED_DIR）。"""
    if not shutil.which("mineru"):
        logger.error("  未找到 mineru 命令，请先确认项目虚拟环境后再运行")
        return {}

    cmd = ["mineru", "-p", str(pdf_path), "-o", str(PARSED_DIR),
           "-b", MINERU_BACKEND, "-m", "auto"]
    if PAGE_RANGE:
        cmd += ["-s", str(PAGE_RANGE[0]), "-e", str(PAGE_RANGE[1])]

    logger.info(f"  执行: {' '.join(cmd)}")
    result = subprocess.run(cmd)   # 不捕获输出，让 MinerU 的进度条直接打到终端

    if result.returncode != 0:
        logger.error(f"  MinerU 解析失败 (exit {result.returncode}): {pdf_path.name}")
        return {}

    # 产物在 <PARSED_DIR>/<stem>/ 下（pipeline 后端会再按 method 分一层子目录）
    out_root = PARSED_DIR / pdf_path.stem
    md_files = sorted(out_root.rglob("*.md"))
    cl_files = sorted(out_root.rglob("*_content_list.json"))
    logger.info(f"  MinerU 完成: {len(md_files)} 个 md, {len(cl_files)} 个 content_list")
    return {
        "markdown":     [str(p.relative_to(PARSED_DIR)) for p in md_files],
        "content_list": [str(p.relative_to(PARSED_DIR)) for p in cl_files],
    }


# ── HTML 解析（BeautifulSoup）────────────────────────────────────────────────

def table_to_markdown(table_tag) -> str:
    """把 HTML 表格转成 markdown 表格，方便 LLM 理解。"""
    rows = []
    for tr in table_tag.find_all("tr"):
        cells = [td.get_text(" ", strip=True) for td in tr.find_all(["td", "th"])]
        if cells:
            rows.append(cells)

    if not rows:
        return ""

    header = rows[0]
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for row in rows[1:]:
        while len(row) < len(header):
            row.append("")
        lines.append("| " + " | ".join(row[:len(header)]) + " |")

    return "\n".join(lines)


def html_to_markdown(html_text: str) -> str:
    """把一段 HTML 正文转成 Markdown 文本（标题/段落/列表/代码块/表格）。"""
    from bs4 import BeautifulSoup, NavigableString, Tag

    soup = BeautifulSoup(html_text, "lxml")

    # 去掉脚本/样式/导航等噪声
    for sel in ["script", "style", "noscript", "nav", "footer", "header"]:
        for tag in soup.find_all(sel):
            tag.decompose()

    # 去掉标题锚点 ¶、面包屑导航（| Contents | ...）、页首摘录提示条等噪声
    for a in soup.find_all("a", class_=re.compile(r"anchor-link|headerlink|permalink", re.I)):
        a.decompose()
    for p in soup.find_all("p"):
        txt = p.get_text(" ", strip=True)
        if "| Contents |" in txt or txt.startswith("| Contents"):
            p.decompose()

    # 优先定位正文容器（PDSH 用 div.post），找不到就退回 body
    root = (soup.select_one("div.post") or soup.select_one("article")
            or soup.select_one("main") or soup.body or soup)

    lines: list[str] = []

    def walk(node):
        for child in node.children:
            if isinstance(child, NavigableString):
                text = str(child).strip()
                if text:
                    lines.append(text)
                continue
            if not isinstance(child, Tag):
                continue

            name = child.name.lower()
            cls = " ".join(child.get("class") or [])

            # Jupyter 的输入/输出提示符（In [1]:）属于噪声
            if "prompt" in cls:
                continue

            if name in ("h1", "h2", "h3", "h4", "h5", "h6"):
                level = int(name[1])
                lines.append(f"\n{'#' * level} {child.get_text(' ', strip=True)}\n")
            elif name == "p":
                lines.append(child.get_text(" ", strip=True))
                lines.append("")
            elif name == "pre":
                code = child.get_text()
                lang = ""
                for c in child.get("class") or []:
                    if c.startswith("language-"):
                        lang = c[len("language-"):]
                        break
                lines.append(f"```{lang}\n{code}\n```\n")
            elif name == "li":
                lines.append("- " + child.get_text(" ", strip=True))
            elif name == "table":
                lines.append(table_to_markdown(child))
                lines.append("")
            elif name == "blockquote":
                for ln in child.get_text(" ", strip=True).splitlines():
                    lines.append("> " + ln)
                lines.append("")
            elif name in ("ul", "ol", "div", "span", "section", "article", "main"):
                walk(child)   # 容器标签：递归
            else:
                text = child.get_text(" ", strip=True)
                if text:
                    lines.append(text)

    walk(root)
    md = "\n".join(lines)
    # 归一化：HTML 的 &nbsp;（\xa0）转普通空格，去掉行尾空白，压缩多余空行
    md = md.replace("\xa0", " ")
    md = re.sub(r"[ \t]+\n", "\n", md)
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()


def _page_title(html_text: str) -> str:
    m = re.search(r"<title[^>]*>(.*?)</title>", html_text, flags=re.S | re.I)
    return m.group(1).strip() if m else ""


def parse_html_book(book: dict) -> dict:
    """用 BeautifulSoup 解析整本 HTML 书，合并为单个 .md，并写 sections.json。"""
    html_dir = RAW_DIR / book["rel_path"]
    files = [html_dir / f for f in book.get("files", []) if (html_dir / f).exists()]
    # 排除首页目录，只留正文章节（按文件名自然排序，PDSH 文件名自带章节编号）
    chapter_files = sorted(f for f in files if f.name != "index.html")

    out_dir = PARSED_DIR / book["name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    sections = []
    md_parts = []
    for f in chapter_files:
        html_text = f.read_text(encoding="utf-8", errors="ignore")
        md = html_to_markdown(html_text)
        rel = str(f.relative_to(RAW_DIR))
        md_parts.append(f"\n<!-- source: {rel} -->\n{md}")
        sections.append({
            "source": rel,
            "title":  _page_title(html_text) or f.stem,
            "chars":  len(md),
        })

    md_path = out_dir / f"{book['name']}.md"
    md_path.write_text("\n".join(md_parts), encoding="utf-8")
    sections_path = out_dir / "sections.json"
    sections_path.write_text(json.dumps(sections, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info(f"  HTML 解析完成: {len(chapter_files)} 个章节 → {md_path.name}")
    return {
        "markdown": [str(md_path.relative_to(PARSED_DIR))],
        "sections": str(sections_path.relative_to(PARSED_DIR)),
    }


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    manifest_path = RAW_DIR.parent / "manifest.json"
    if not manifest_path.exists():
        logger.error("未找到 data/manifest.json，请先运行 download_books.py")
        return
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    parsed_manifest = []
    for book in manifest:
        logger.info(f"── {book['title']} ({book['type']}) ──")
        entry = {"name": book["name"], "title": book["title"], "type": book["type"]}

        if book["type"] == "pdf":
            pdf_path = RAW_DIR / book["rel_path"]
            if not pdf_path.exists():
                logger.warning(f"  文件不存在，跳过: {pdf_path}")
                continue
            entry["outputs"] = run_mineru(pdf_path)

        elif book["type"] == "html":
            entry["outputs"] = parse_html_book(book)

        else:
            logger.warning(f"  未知类型，跳过: {book['type']}")
            continue

        parsed_manifest.append(entry)

    out_path = PARSED_DIR / "manifest.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(parsed_manifest, f, ensure_ascii=False, indent=2)
    logger.info(f"\n全部解析完成，索引见 {out_path}")


if __name__ == "__main__":
    main()
