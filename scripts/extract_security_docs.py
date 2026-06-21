"""
Extract security documentation from OWASP cheat sheets + ASVS.
Convert markdown docs to training examples.
"""
import os
import re
import json
import hashlib
from pathlib import Path
from datetime import datetime

RAW_DIR = Path("/home/z/my-project/dreamgtm/data/raw/security_data/owasp")
OUT_FILE = Path("/home/z/my-project/dreamgtm/data/verified/security_docs.jsonl")
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)


def make_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def extract_title(md: str, filename: str) -> str:
    """Get the first H1 title."""
    for line in md.split("\n"):
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
    # Fallback to filename
    return filename.replace("_", " ").replace(".md", "").strip()


def split_into_sections(md: str) -> list:
    """Split markdown into sections by H2 headers."""
    sections = []
    current_title = None
    current_lines = []
    for line in md.split("\n"):
        stripped = line.strip()
        if stripped.startswith("## "):
            if current_title and current_lines:
                sections.append((current_title, "\n".join(current_lines)))
            current_title = stripped[3:].strip()
            current_lines = []
        elif stripped.startswith("# "):
            # Top-level header - save existing section
            if current_title and current_lines:
                sections.append((current_title, "\n".join(current_lines)))
            current_title = stripped[2:].strip()
            current_lines = []
        else:
            current_lines.append(line)
    if current_title and current_lines:
        sections.append((current_title, "\n".join(current_lines)))
    return sections


def clean_md(md: str) -> str:
    """Remove HTML comments, fix whitespace."""
    md = re.sub(r"<!--.*?-->", "", md, flags=re.DOTALL)
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()


def main():
    print(f"[{datetime.now():%H:%M:%S}] Extracting OWASP security docs...")
    cheat_sheets = list((RAW_DIR / "CheatSheetSeries-master" / "cheatsheets").glob("*.md"))
    asvs = list((RAW_DIR / "ASVS-master").rglob("*.md"))
    print(f"Found {len(cheat_sheets)} cheat sheets, {len(asvs)} ASVS docs")

    kept = 0
    bytes_written = 0

    with OUT_FILE.open("w", encoding="utf-8") as f:
        for md_path in cheat_sheets + asvs:
            try:
                content = md_path.read_text(encoding="utf-8", errors="strict")
            except (UnicodeDecodeError, OSError):
                continue
            content = clean_md(content)
            if len(content) < 500:
                continue
            title = extract_title(content, md_path.name)
            # Split into sections if the doc is large
            if len(content) > 8000:
                sections = split_into_sections(content)
                for sec_title, sec_body in sections:
                    sec_body = clean_md(sec_body)
                    if len(sec_body) < 300:
                        continue
                    record = {
                        "id": make_id(title + sec_title + sec_body[:500]),
                        "type": "security_doc",
                        "source": "owasp_cheatsheet" if "cheatsheets" in str(md_path) else "owasp_asvs",
                        "title": title,
                        "section": sec_title,
                        "content": sec_body,
                        "chars": len(sec_body),
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    kept += 1
                    bytes_written += len(record["content"]) + 200
            else:
                record = {
                    "id": make_id(title + content[:500]),
                    "type": "security_doc",
                    "source": "owasp_cheatsheet" if "cheatsheets" in str(md_path) else "owasp_asvs",
                    "title": title,
                    "section": "",
                    "content": content,
                    "chars": len(content),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept += 1
                bytes_written += len(content) + 200

    print(f"\nDONE: {kept} security docs ({bytes_written/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
