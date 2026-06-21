"""
Multi-language verification pass:
- CodeSearchNet JavaScript, Go, Ruby, PHP
- github-code shards: extract Rust, C, C++, TypeScript, Shell, etc.
- Delete shards as we go to free disk
"""
import os
import ast
import json
import time
import hashlib
import pyarrow.parquet as pq
from pathlib import Path
from datetime import datetime

VERIFIED_DIR = Path("/home/z/my-project/dreamgtm/data/verified")
CSN_RAW = Path("/home/z/my-project/dreamgtm/data/raw/codesearchnet")
GC_RAW = Path("/home/z/my-project/dreamgtm/data/raw/github_code_multi")

# Languages we want (multi-language AI!)
TARGET_LANGS = {"python", "javascript", "typescript", "go", "rust", "ruby",
                "php", "c", "c++", "c#", "java", "kotlin", "swift", "scala",
                "shell", "bash", "html", "css", "sql", "lua", "perl", "r", "dart"}

MIN_LINES = 10
MAX_LINES = 500
MIN_CHARS = 200
MAX_CHARS = 15000


def make_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def is_valid_python(code: str) -> bool:
    if not code or len(code.strip()) < 50:
        return False
    try:
        tree = ast.parse(code)
        return any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                   for n in ast.walk(tree))
    except (SyntaxError, ValueError, MemoryError):
        return False


def normalize_code(code: str) -> str:
    return "\n".join(line.rstrip() for line in code.replace("\r\n", "\n").split("\n")).strip()


def load_existing_ids():
    ids = set()
    for f in VERIFIED_DIR.glob("*.jsonl"):
        with f.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                    if "id" in rec:
                        ids.add(rec["id"])
                    if "code" in rec:
                        ids.add("c:" + make_id(rec["code"]))
                except json.JSONDecodeError:
                    continue
    print(f"Loaded {len(ids)} existing IDs", flush=True)
    return ids


def process_csn_lang(parquet_path: Path, lang: str, out, seen: set):
    """Process a per-language CodeSearchNet parquet."""
    print(f"\n--- CodeSearchNet {lang} ---", flush=True)
    if not parquet_path.exists():
        print("  SKIP", flush=True)
        return 0, 0
    table = pq.read_table(parquet_path)
    print(f"  Rows: {table.num_rows}", flush=True)
    kept = 0
    skipped = 0
    bytes_written = 0
    for i in range(table.num_rows):
        row = table.slice(i, 1).to_pylist()[0]
        code = row.get("func_code_string") or row.get("whole_func_string") or ""
        doc = row.get("func_documentation_string") or ""
        if not code or not doc:
            skipped += 1
            continue
        code = normalize_code(code)
        if len(code) < MIN_CHARS or len(code) > MAX_CHARS:
            skipped += 1
            continue
        line_count = code.count("\n") + 1
        if line_count < MIN_LINES or line_count > MAX_LINES:
            skipped += 1
            continue
        # For Python specifically, AST validate
        if lang == "python":
            if not is_valid_python(code):
                skipped += 1
                continue
        rec_id = make_id(code)
        if rec_id in seen:
            skipped += 1
            continue
        seen.add(rec_id)
        record = {
            "id": rec_id,
            "type": "codesearchnet_function",
            "language": lang,
            "source_repo": row.get("repository_name", ""),
            "func_name": row.get("func_name", ""),
            "docstring": doc.strip(),
            "code": code,
            "lines": line_count,
            "chars": len(code),
            "url": row.get("func_code_url", ""),
        }
        line = json.dumps(record, ensure_ascii=False)
        out.write(line + "\n")
        kept += 1
        bytes_written += len(line) + 1
        if kept % 10000 == 0:
            print(f"  kept={kept} skipped={skipped} ({bytes_written/1e6:.1f}MB)", flush=True)
    print(f"  FINAL kept={kept} skipped={skipped} ({bytes_written/1e6:.1f}MB)", flush=True)
    # Delete parquet after
    try:
        parquet_path.unlink()
    except:
        pass
    return kept, bytes_written


def process_github_shard(shard_path: Path, out, seen: set):
    """Process github-code shard - extract multi-language files."""
    print(f"\n--- github-code {shard_path.name} ---", flush=True)
    try:
        table = pq.read_table(shard_path)
    except Exception as e:
        print(f"  ERROR: {e}", flush=True)
        try:
            shard_path.unlink()
        except:
            pass
        return 0, 0
    kept = 0
    skipped = 0
    bytes_written = 0
    lang_counter = {}
    for i in range(table.num_rows):
        row = table.slice(i, 1).to_pylist()[0]
        lang = (row.get("language") or "").lower()
        if lang not in TARGET_LANGS:
            skipped += 1
            continue
        # Skip Python - already have it
        if lang == "python":
            skipped += 1
            continue
        code = row.get("code", "")
        if not code or len(code) < MIN_CHARS or len(code) > MAX_CHARS:
            skipped += 1
            continue
        code = normalize_code(code)
        line_count = code.count("\n") + 1
        if line_count < MIN_LINES or line_count > MAX_LINES:
            skipped += 1
            continue
        rec_id = make_id(code)
        if rec_id in seen:
            skipped += 1
            continue
        seen.add(rec_id)
        lang_counter[lang] = lang_counter.get(lang, 0) + 1
        record = {
            "id": rec_id,
            "type": "github_code_file",
            "language": lang,
            "source_repo": row.get("repo_name", ""),
            "path": row.get("path", ""),
            "license": row.get("license", ""),
            "code": code,
            "lines": line_count,
            "chars": len(code),
        }
        line = json.dumps(record, ensure_ascii=False)
        out.write(line + "\n")
        kept += 1
        bytes_written += len(line) + 1
    print(f"  kept={kept} skipped={skipped} ({bytes_written/1e6:.1f}MB) langs={lang_counter}", flush=True)
    # Delete shard after
    try:
        shard_path.unlink()
    except:
        pass
    return kept, bytes_written


def main():
    print(f"[{datetime.now():%H:%M:%S}] Multi-language verification...", flush=True)
    seen = load_existing_ids()
    OUT_FILE = VERIFIED_DIR / "multilang_code.jsonl"
    total_kept = 0
    total_bytes = 0
    with OUT_FILE.open("w", encoding="utf-8") as out:
        # 1. CodeSearchNet per-language
        for lang in ["javascript", "go", "ruby", "php"]:
            p = CSN_RAW / f"{lang}_train.parquet"
            if p.exists():
                k, b = process_csn_lang(p, lang, out, seen)
                total_kept += k
                total_bytes += b

        # 2. github-code multi-language shards
        for shard in sorted(GC_RAW.glob("shard_*.parquet")):
            k, b = process_github_shard(shard, out, seen)
            total_kept += k
            total_bytes += b

    print(f"\n=== TOTAL ADDED: {total_kept} ({total_bytes/1e6:.1f} MB) ===", flush=True)


if __name__ == "__main__":
    main()
