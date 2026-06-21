"""
Pass 2b - CodeSearchNet all languages only.
Python was already verified in pass 1, this adds JS, Java, Go.
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
RAW_DIR = Path("/home/z/my-project/dreamgtm/data/raw")


def make_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return s.replace("\r\n", "\n").strip()


def main():
    print(f"[{datetime.now():%H:%M:%S}] CodeSearchNet all-languages verification...")
    OUT_FILE = VERIFIED_DIR / "codesearchnet_all_lang.jsonl"

    # Load existing IDs from codesearchnet_python.jsonl to skip dupes
    seen_ids = set()
    cs_py_file = VERIFIED_DIR / "codesearchnet_python.jsonl"
    if cs_py_file.exists():
        with cs_py_file.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if "code" in rec:
                        seen_ids.add(make_id(rec["code"]))
                except json.JSONDecodeError:
                    continue
    print(f"Loaded {len(seen_ids)} Python IDs for dedup")

    kept = 0
    skipped = 0
    bytes_written = 0
    start = time.time()
    languages = {}

    with OUT_FILE.open("w", encoding="utf-8") as out:
        for shard in ["all_train_00000.parquet", "all_train_00001.parquet"]:
            path = RAW_DIR / "codesearchnet" / shard
            if not path.exists():
                continue
            print(f"\nProcessing {shard}...")
            try:
                table = pq.read_table(path)
            except Exception as e:
                print(f"  ERROR reading: {e}")
                continue
            print(f"  Cols: {table.column_names}, rows: {table.num_rows}")

            for i in range(table.num_rows):
                row = table.slice(i, 1).to_pylist()[0]
                lang = (row.get("language") or "").lower()
                if lang not in {"python", "javascript", "java", "go"}:
                    skipped += 1
                    continue
                # Skip python (already done)
                if lang == "python":
                    skipped += 1
                    continue
                code = row.get("func_code_string") or row.get("whole_func_string") or ""
                doc = row.get("func_documentation_string") or ""
                if not code or not doc:
                    skipped += 1
                    continue
                code = clean_text(code)
                if len(code) < 50 or len(code) > 12000:
                    skipped += 1
                    continue
                line_count = code.count("\n") + 1
                if line_count < 5 or line_count > 250:
                    skipped += 1
                    continue
                rec_id = make_id(code)
                if rec_id in seen_ids:
                    skipped += 1
                    continue
                seen_ids.add(rec_id)
                languages[lang] = languages.get(lang, 0) + 1

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

                if kept % 25000 == 0:
                    elapsed = time.time() - start
                    print(f"  kept={kept} skipped={skipped} | "
                          f"{bytes_written/1e6:.1f}MB | {elapsed:.0f}s | "
                          f"langs={languages}", flush=True)

    elapsed = time.time() - start
    print(f"\nDONE: {kept} verified multi-lang functions ({bytes_written/1e6:.1f} MB)")
    print(f"Skipped: {skipped}")
    print(f"Time: {elapsed:.0f}s")
    print(f"Languages: {languages}")


if __name__ == "__main__":
    main()
