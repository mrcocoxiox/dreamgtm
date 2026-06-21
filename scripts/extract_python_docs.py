"""
Extract Python official documentation from HTML files.
Convert to clean text sections for training.
"""
import os
import re
import json
import hashlib
from html.parser import HTMLParser
from pathlib import Path
from datetime import datetime

RAW_DIR = Path("/home/z/my-project/dreamgtm/data/raw/python-3.12.0-docs-html")
OUT_FILE = Path("/home/z/my-project/dreamgtm/data/verified/python_docs.jsonl")
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)


def make_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class HTMLTextExtractor(HTMLParser):
    """Convert HTML to clean text - preserve code blocks specially."""

    def __init__(self):
        super().__init__()
        self.parts = []
        self.in_code = False
        self.in_pre = False
        self.skip_tags = {"script", "style", "nav", "header", "footer"}

    def handle_starttag(self, tag, attrs):
        if tag in self.skip_tags:
            self.in_skip = True
        if tag in ("pre", "code"):
            if tag == "pre":
                self.in_pre = True
                self.parts.append("\n```\n")
            else:
                self.in_code = True
                if not self.in_pre:
                    self.parts.append("`")
        elif tag in ("h1", "h2", "h3", "h4"):
            self.parts.append("\n\n## ")
        elif tag == "p":
            self.parts.append("\n\n")
        elif tag == "li":
            self.parts.append("\n- ")
        elif tag == "a":
            pass  # ignore links

    def handle_endtag(self, tag):
        if tag in self.skip_tags:
            self.in_skip = False
        if tag == "pre":
            self.in_pre = False
            self.parts.append("\n```\n")
        elif tag == "code" and not self.in_pre:
            self.in_code = False
            self.parts.append("`")

    def handle_data(self, data):
        if hasattr(self, "in_skip") and self.in_skip:
            return
        self.parts.append(data)

    def get_text(self):
        text = "".join(self.parts)
        # Cleanup
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


def html_to_text(html: str) -> str:
    parser = HTMLTextExtractor()
    parser.in_skip = False
    try:
        parser.feed(html)
    except Exception:
        return ""
    return parser.get_text()


def extract_title(html: str, fallback: str) -> str:
    m = re.search(r"<title>(.*?)</title>", html, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip().split("—")[0].strip()
    m = re.search(r"<h1[^>]*>(.*?)</h1>", html, re.DOTALL | re.IGNORECASE)
    if m:
        return re.sub(r"<[^>]+>", "", m.group(1)).strip()
    return fallback


def main():
    print(f"[{datetime.now():%H:%M:%S}] Extracting Python docs...")
    html_files = list(RAW_DIR.rglob("*.html"))
    print(f"Found {len(html_files)} HTML files")

    kept = 0
    bytes_written = 0

    with OUT_FILE.open("w", encoding="utf-8") as f:
        for html_path in html_files:
            # Skip index, search, download pages
            name = html_path.name.lower()
            if name in {"index.html", "search.html", "genindex.html", "py-modindex.html",
                       "contents.html", "about.html", "bugs.html", "copyright.html",
                       "license.html", "searchindex.js"}:
                continue
            if "_sources" in str(html_path) or "_static" in str(html_path) or "_images" in str(html_path):
                continue
            try:
                html = html_path.read_text(encoding="utf-8", errors="strict")
            except (UnicodeDecodeError, OSError):
                continue
            text = html_to_text(html)
            if len(text) < 500:
                continue
            title = extract_title(html, html_path.stem)
            rel_path = html_path.relative_to(RAW_DIR)
            record = {
                "id": make_id(text[:1000]),
                "type": "python_doc",
                "source": str(rel_path),
                "title": title,
                "content": text,
                "chars": len(text),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1
            bytes_written += len(text) + 200

    print(f"\nDONE: {kept} Python docs ({bytes_written/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
