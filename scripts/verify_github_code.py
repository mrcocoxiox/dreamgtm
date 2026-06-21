"""
Verify github-code-clean shards - keep only Python files, AST-validated.
"""
import os
import ast
import json
import time
import hashlib
import pyarrow.parquet as pq
from pathlib import Path
from datetime import datetime

RAW_DIR = Path("/home/z/my-project/dreamgtm/data/raw/github_code")
OUT_FILE = Path("/home/z/my-project/dreamgtm/data/verified/github_code_python.jsonl")
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

MIN_LINES = 10
MAX_LINES = 500
MIN_CHARS = 200
MAX_CHARS = 12000


def make_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def is_valid_python(code: str) -> bool:
    if not code or len(code.strip()) < 50:
        return False
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, MemoryError):
        return False
    # Must have at least one def or class
    has_def = any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                  for n in ast.walk(tree))
    return has_def


def normalize_code(code: str) -> str:
    lines = [line.rstrip() for line in code.replace("\r\n", "\n").split("\n")]
    return "\n".join(lines).strip()


def main():
    print(f"[{datetime.now():%H:%M:%S}] Verifying github-code-clean Python...")
    shards = sorted(RAW_DIR.glob("shard_*.parquet"))
    print(f"Found {len(shards)} shards")

    kept = 0
    skipped = 0
    seen_ids = set()
    bytes_written = 0
    start = time.time()

    with OUT_FILE.open("w", encoding="utf-8") as f:
        for shard in shards:
            print(f"\n  Processing {shard.name}...")
            t = pq.read_table(shard)
            cols = t.column_names
            for i in range(t.num_rows):
                row = t.slice(i, 1).to_pylist()[0]
                lang = (row.get("language") or "").lower()
                if lang != "python":
                    skipped += 1
                    continue
                code = row.get("code", "")
                if not code or len(code) < MIN_CHARS or len(code) > MAX_CHARS:
                    skipped += 1
                    continue
                code = normalize_code(code)
                if not is_valid_python(code):
                    skipped += 1
                    continue
                line_count = code.count("\n") + 1
                if line_count < MIN_LINES or line_count > MAX_LINES:
                    skipped += 1
                    continue
                rec_id = make_id(code)
                if rec_id in seen_ids:
                    skipped += 1
                    continue
                seen_ids.add(rec_id)

                record = {
                    "id": rec_id,
                    "type": "github_code_file",
                    "source_repo": row.get("repo_name", ""),
                    "path": row.get("path", ""),
                    "license": row.get("license", ""),
                    "code": code,
                    "lines": line_count,
                    "chars": len(code),
                }
                line = json.dumps(record, ensure_ascii=False)
                f.write(line + "\n")
                kept += 1
                bytes_written += len(line) + 1

                if kept % 25000 == 0:
                    elapsed = time.time() - start
                    rate = kept / elapsed if elapsed > 0 else 0
                    print(f"    kept={kept} skipped={skipped} | "
                          f"{bytes_written/1e6:.1f} MB | "
                          f"{rate:.0f} ex/s", flush=True)

            # Print per-shard summary
            print(f"  Cumulative: kept={kept} skipped={skipped} "
                  f"bytes={bytes_written/1e6:.1f}MB")

    elapsed = time.time() - start
    print(f"\nDONE: {kept} verified Python files ({bytes_written/1e6:.1f} MB)")
    print(f"Skipped: {skipped}")
    print(f"Time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
