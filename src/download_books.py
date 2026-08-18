"""
数据下载脚本：下载 RAG 项目所需的机器学习书籍（PDF / HTML）

书籍来源：Compendium of free ML reading resources
    https://github.com/Carl-McBride-Ellis/Compendium-of-free-ML-reading-resources

说明：
  * 五本书里四本为单文件 PDF，一本（Python Data Science Handbook）为多页面 HTML 站点，
    脚本按 type 区分处理：pdf 直接下载单文件，html 抓取首页里所有章节页面。
  * 全部文件下载到 data/raw 下，HTML 书单独放在同名子目录。
  * 下载完成后在 data/manifest.json 记录每本书的来源 URL，用于最终「答案来源」溯源。
"""

import re
import time
import json
import random
import logging
import requests
from pathlib import Path
from urllib.parse import urljoin, urlparse

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
}

# 书籍清单
#   type="pdf"  : 单文件，url 为直链，filename 为保存名
#   type="html" : 多页面站点，url 为首页，脚本会抓取首页里的所有章节页面
BOOKS = [
    {
        "name": "introduction_to_statistical_learning_python",
        "title": "An Introduction to Statistical Learning with Applications in Python",
        "type": "pdf",
        # 官方站直链跳转到 Google Drive，此 Drive 直链已验证可返回 PDF
        "url": "https://drive.google.com/uc?export=download&id=1ajFkHO6zjrdGNqhqW1jKBZdiNGh_8YQ1",
        "filename": "ISLP_website.pdf",
    },
    {
        "name": "python_data_science_handbook",
        "title": "Python Data Science Handbook",
        "type": "html",
        "url": "https://jakevdp.github.io/PythonDataScienceHandbook/",
    },
    {
        "name": "dive_into_deep_learning",
        "title": "Dive into Deep Learning",
        "type": "pdf",
        "url": "https://d2l.ai/d2l-en.pdf",
        "filename": "d2l-en.pdf",
    },
    {
        "name": "mathematics_for_machine_learning",
        "title": "Mathematics for Machine Learning",
        "type": "pdf",
        "url": "https://mml-book.github.io/book/mml-book.pdf",
        "filename": "mml-book.pdf",
    },
    {
        "name": "mathematical_foundations_of_rl",
        "title": "Mathematical Foundations of Reinforcement Learning",
        "type": "pdf",
        "url": "https://raw.githubusercontent.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning/main/Book-all-in-one.pdf",
        "filename": "mathematical-foundations-of-rl.pdf",
    },
]


def _basename(url: str) -> str:
    """取 URL 路径里的文件名（不含查询参数）。"""
    return urlparse(url).path.rsplit("/", 1)[-1]


def download_file(url: str, save_path: Path) -> bool:
    """下载单个文件到 save_path，已存在则跳过；失败自动重试 3 次。"""
    if save_path.exists():
        logger.info(f"已存在，跳过: {save_path.name}")
        return True

    for attempt in range(3):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=120, stream=True)
            resp.raise_for_status()
            with open(save_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    f.write(chunk)
            size_kb = save_path.stat().st_size // 1024
            logger.info(f"下载成功: {save_path.name}  ({size_kb} KB)")
            return True
        except Exception as e:
            logger.warning(f"第{attempt+1}次失败: {e}")
            if attempt < 2:
                time.sleep(2 ** attempt)

    logger.error(f"下载失败: {url}")
    return False


def fetch_chapter_links(index_url: str) -> list[str]:
    """抓取 HTML 书首页里的章节页面链接（形如 05.02-xxx.html），去重排序后返回绝对 URL。"""
    resp = requests.get(index_url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    hrefs = re.findall(r'href="([^"]+?\.html)"', resp.text, flags=re.IGNORECASE)
    chapters = {
        urljoin(index_url, h)
        for h in hrefs
        if re.match(r"\d{2}\.\d{2}-", _basename(urljoin(index_url, h)))
    }
    return sorted(chapters)


def download_html_book(book: dict, save_dir: Path) -> list[str]:
    """下载整本 HTML 书：首页 + 所有章节页面，返回保存的文件名列表。"""
    save_dir.mkdir(parents=True, exist_ok=True)
    index_url = book["url"]
    files: list[str] = []

    # 首页（含目录，便于后续溯源）
    if download_file(index_url, save_dir / "index.html"):
        files.append("index.html")

    chapters = fetch_chapter_links(index_url)
    logger.info(f"  解析到 {len(chapters)} 个章节页面")
    for url in chapters:
        name = _basename(url)
        if download_file(url, save_dir / name):
            files.append(name)
        time.sleep(random.uniform(0.3, 0.8))

    return files


def main():
    manifest = []

    for book in BOOKS:
        logger.info(f"── {book['title']} ──")

        if book["type"] == "pdf":
            save_path = RAW_DIR / book["filename"]
            ok = download_file(book["url"], save_path)
            rel_path = book["filename"]
            files = [book["filename"]] if ok else []
        else:  # html
            save_dir = RAW_DIR / book["name"]
            files = download_html_book(book, save_dir)
            rel_path = book["name"]

        manifest.append({
            "name": book["name"],
            "title": book["title"],
            "type": book["type"],
            "source_url": book["url"],
            "rel_path": rel_path,
            "num_files": len(files),
            "files": files,
        })

        time.sleep(random.uniform(1.0, 2.0))

    manifest_path = RAW_DIR.parent / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    logger.info(f"\n完成！共处理 {len(manifest)} 本书")
    for item in manifest:
        logger.info(f"  {item['title']} ({item['type']}): {item['num_files']} 个文件 -> data/raw/{item['rel_path']}")


if __name__ == "__main__":
    main()
