"""
Pass 3 verification - additional instruction datasets.
- Magicoder OSS-Instruct 75K (high quality code instruction)
- self-oss-instruct-sc2-exec-filter-50k (execution filtered)
- Databricks Dolly 15K
- Capybara (multi-turn)
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
RAW_DIR = Path("/home/z/my-project/dreamgtm/data/raw/more_data")


def make_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def has_valid_python(text: str) -> bool:
    if "```python" not in text and "```py\n" not in text:
        return False
    parts = text.split("```")
    for i in range(1, len(parts), 2):
        if i >= len(parts):
            break
        block = parts[i]
        if block.startswith("python") or block.startswith("py"):
            block = block.split("\n", 1)[1] if "\n" in block else block[6:]
        else:
            continue
        try:
            ast.parse(block)
            return True
        except (SyntaxError, ValueError):
            continue
    return False


def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return s.replace("\r\n", "\n").strip()


def load_existing_ids():
    """Load IDs from all verified files."""
    ids = set()
    for f in VERIFIED_DIR.glob("*.jsonl"):
        with f.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                    if "id" in rec:
                        ids.add(rec["id"])
                    if "code" in rec:
                        ids.add("code:" + make_id(rec["code"]))
                    if "prompt" in rec and "response" in rec:
                        ids.add("qa:" + make_id(rec["prompt"] + rec["response"][:500]))
                    if "content" in rec:
                        ids.add("content:" + make_id(rec["content"][:500]))
                except json.JSONDecodeError:
                    continue
    print(f"Loaded {len(ids)} existing IDs")
    return ids


def process_jsonl(jsonl_path: Path, name: str, prompt_key: str, response_key: str,
                  require_code: bool, out, seen: set):
    """Process a JSONL file with prompt/response fields."""
    print(f"\n--- {name} ---")
    if not jsonl_path.exists():
        print(f"  SKIP (missing)")
        return 0, 0
    kept = 0
    skipped = 0
    bytes_written = 0
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            prompt = clean_text(str(ex.get(prompt_key, "")))
            response = clean_text(str(ex.get(response_key, "")))
            if not prompt or not response:
                skipped += 1
                continue
            if len(prompt) + len(response) < 200:
                skipped += 1
                continue
            if len(response) > 16000:
                skipped += 1
                continue
            if require_code and not has_valid_python(response):
                skipped += 1
                continue
            rec_id = make_id(prompt + response[:500])
            if rec_id in seen:
                skipped += 1
                continue
            seen.add(rec_id)
            record = {
                "id": rec_id,
                "type": name,
                "prompt": prompt,
                "response": response,
                "chars": len(prompt) + len(response),
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1
            bytes_written += record["chars"] + 500
            if kept % 10000 == 0:
                print(f"  kept={kept} skipped={skipped} ({bytes_written/1e6:.1f}MB)", flush=True)
    print(f"  FINAL kept={kept} skipped={skipped} ({bytes_written/1e6:.1f}MB)")
    return kept, bytes_written


def process_self_oss_sc2(parquet_path: Path, out, seen: set):
    """Process self-oss-instruct-sc2 - execution-filtered instructions."""
    print(f"\n--- self_oss_sc2_exec_filter ---")
    if not parquet_path.exists():
        print("  SKIP (missing)")
        return 0, 0
    table = pq.read_table(parquet_path)
    print(f"  Cols: {table.column_names}, rows: {table.num_rows}")
    kept = 0
    skipped = 0
    bytes_written = 0
    for i in range(table.num_rows):
        row = table.slice(i, 1).to_pylist()[0]
        # self-oss-sc2 has 'instruction' + 'response' typically
        prompt = clean_text(str(row.get("instruction", "") or row.get("prompt", "")))
        response = clean_text(str(row.get("response", "") or row.get("output", "")))
        if not prompt or not response:
            skipped += 1
            continue
        if len(prompt) + len(response) < 200:
            skipped += 1
            continue
        if len(response) > 16000:
            skipped += 1
            continue
        if not has_valid_python(response):
            skipped += 1
            continue
        rec_id = make_id(prompt + response[:500])
        if rec_id in seen:
            skipped += 1
            continue
        seen.add(rec_id)
        record = {
            "id": rec_id,
            "type": "self_oss_sc2_exec_filter",
            "prompt": prompt,
            "response": response,
            "chars": len(prompt) + len(response),
        }
        out.write(json.dumps(record, ensure_ascii=False) + "\n")
        kept += 1
        bytes_written += record["chars"] + 500
        if kept % 10000 == 0:
            print(f"  kept={kept} skipped={skipped} ({bytes_written/1e6:.1f}MB)", flush=True)
    print(f"  FINAL kept={kept} skipped={skipped} ({bytes_written/1e6:.1f}MB)")
    return kept, bytes_written


def process_capybara(jsonl_path: Path, out, seen: set):
    """Process Capybara - multi-turn conversations."""
    print(f"\n--- capybara ---")
    if not jsonl_path.exists():
        print("  SKIP (missing)")
        return 0, 0
    kept = 0
    skipped = 0
    bytes_written = 0
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            convs = ex.get("conversation", [])
            if not isinstance(convs, list) or not convs:
                skipped += 1
                continue
            # Take first user + first GPT response
            user_msgs = [c for c in convs if c.get("human", "") or c.get("input", "")]
            asst_msgs = [c for c in convs if c.get("gpt", "") or c.get("output", "")]
            if not user_msgs or not asst_msgs:
                skipped += 1
                continue
            prompt = clean_text(user_msgs[0].get("human", "") or user_msgs[0].get("input", ""))
            response = clean_text(asst_msgs[0].get("gpt", "") or asst_msgs[0].get("output", ""))
            if not prompt or not response:
                skipped += 1
                continue
            if len(prompt) + len(response) < 200:
                skipped += 1
                continue
            if len(response) > 16000:
                skipped += 1
                continue
            if not has_valid_python(response):
                skipped += 1
                continue
            rec_id = make_id(prompt + response[:500])
            if rec_id in seen:
                skipped += 1
                continue
            seen.add(rec_id)
            record = {
                "id": rec_id,
                "type": "capybara_code",
                "prompt": prompt,
                "response": response,
                "chars": len(prompt) + len(response),
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1
            bytes_written += record["chars"] + 500
    print(f"  FINAL kept={kept} skipped={skipped} ({bytes_written/1e6:.1f}MB)")
    return kept, bytes_written


def main():
    print(f"[{datetime.now():%H:%M:%S}] Pass 3 verification...")
    seen = load_existing_ids()
    OUT_FILE = VERIFIED_DIR / "instruction_data_3.jsonl"
    total_kept = 0
    total_bytes = 0
    with OUT_FILE.open("w", encoding="utf-8") as out:
        # 1. Magicoder OSS-Instruct (high quality code)
        k, b = process_jsonl(RAW_DIR / "magicoder.jsonl", "magicoder_oss_instruct",
                              "problem", "solution", True, out, seen)
        total_kept += k
        total_bytes += b

        # 2. Databricks Dolly - keep only code
        k, b = process_jsonl(RAW_DIR / "dolly.jsonl", "dolly_code",
                              "instruction", "response", True, out, seen)
        total_kept += k
        total_bytes += b

        # 3. self-oss-instruct-sc2 (exec filtered)
        k, b = process_self_oss_sc2(RAW_DIR / "self_oss_sc2.parquet", out, seen)
        total_kept += k
        total_bytes += b

        # 4. Capybara (multi-turn code)
        k, b = process_capybara(RAW_DIR / "capybara.jsonl", out, seen)
        total_kept += k
        total_bytes += b

    print(f"\n=== TOTAL ADDED: {total_kept} ({total_bytes/1e6:.1f} MB) ===")


if __name__ == "__main__":
    main()
