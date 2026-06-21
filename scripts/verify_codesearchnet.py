"""
Verify CodeSearchNet Python parquet:
- AST syntax check (compiles cleanly)
- Docstring presence (≥20 chars)
- Length filter (5-200 lines, not trivial, not巨型)
- Exact hash deduplication
Output: verified/codesearchnet_python.jsonl
"""
import os
import ast
import json
import time
import hashlib
import pyarrow.parquet as pq
from pathlib import Path
from datetime import datetime

IN_FILE = Path("/home/z/my-project/dreamgtm/data/raw/codesearchnet/python_train.parquet")
OUT_FILE = Path("/home/z/my-project/dreamgtm/data/verified/codesearchnet_python.jsonl")
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

MIN_CODE_LINES = 5
MAX_CODE_LINES = 250
MIN_DOC_CHARS = 20
MAX_DOC_CHARS = 5000


def make_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def is_valid_python(code: str) -> bool:
    if not code or len(code.strip()) < 50:
        return False
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, MemoryError):
        return False
    has_func = any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in ast.walk(tree))
    return has_func


def normalize_code(code: str) -> str:
    lines = [line.rstrip() for line in code.replace("\r\n", "\n").split("\n")]
    return "\n".join(lines).strip()


def main():
    print(f"[{datetime.now():%H:%M:%S}] Verifying CodeSearchNet Python...")
    if not IN_FILE.exists():
        print(f"ERROR: {IN_FILE} not found")
        return

    table = pq.read_table(IN_FILE)
    print(f"Loaded {table.num_rows} rows, {table.num_columns} cols")
    print(f"Columns: {table.column_names}")

    kept = 0
    skipped = 0
    seen_ids = set()
    bytes_written = 0
    start = time.time()

    with OUT_FILE.open("w", encoding="utf-8") as f:
        for i in range(table.num_rows):
            row = table.slice(i, 1).to_pylist()[0]
            code = row.get("func_code_string") or row.get("whole_func_string") or ""
            doc = row.get("func_documentation_string") or ""
            lang = row.get("language", "")
            if lang and lang != "python":
                skipped += 1
                continue
            if not code or not doc:
                skipped += 1
                continue
            code = normalize_code(code)
            if not is_valid_python(code):
                skipped += 1
                continue
            line_count = code.count("\n") + 1
            if line_count < MIN_CODE_LINES or line_count > MAX_CODE_LINES:
                skipped += 1
                continue
            if len(doc) < MIN_DOC_CHARS or len(doc) > MAX_DOC_CHARS:
                skipped += 1
                continue

            rec_id = make_id(code)
            if rec_id in seen_ids:
                skipped += 1
                continue
            seen_ids.add(rec_id)

            record = {
                "id": rec_id,
                "type": "codesearchnet_function",
                "source_repo": row.get("repository_name", ""),
                "func_name": row.get("func_name", ""),
                "func_path": row.get("func_path_in_repository", ""),
                "docstring": doc.strip(),
                "code": code,
                "lines": line_count,
                "chars": len(code),
                "url": row.get("func_code_url", ""),
            }
            line = json.dumps(record, ensure_ascii=False)
            f.write(line + "\n")
            kept += 1
            bytes_written += len(line) + 1

            if kept % 25000 == 0:
                elapsed = time.time() - start
                rate = kept / elapsed if elapsed > 0 else 0
                print(f"  kept={kept} skipped={skipped} | "
                      f"{bytes_written/1e6:.1f} MB | "
                      f"{rate:.0f} ex/s | "
                      f"elapsed={elapsed:.0f}s", flush=True)

    elapsed = time.time() - start
    print(f"\nDONE: {kept} verified functions ({bytes_written/1e6:.1f} MB)")
    print(f"Skipped: {skipped}")
    print(f"Time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
