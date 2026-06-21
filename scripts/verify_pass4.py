"""
Pass 4 verification - more instruction datasets:
- Open-Orca 1M GPT4
- GPT4-LLM-Cleaned
- OpenAssistant (oasst2)
- no_robots
"""
import os
import ast
import json
import gzip
import time
import hashlib
import pyarrow.parquet as pq
from pathlib import Path
from datetime import datetime

VERIFIED_DIR = Path("/home/z/my-project/dreamgtm/data/verified")
RAW_DIR = Path("/home/z/my-project/dreamgtm/data/raw/more_data_2")


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
    print(f"Loaded {len(ids)} existing IDs", flush=True)
    return ids


def process_openorca(parquet_path: Path, out, seen: set, max_examples=50000):
    """Process Open-Orca - keep only Python code examples."""
    print(f"\n--- OpenOrca 1M GPT4 ---", flush=True)
    if not parquet_path.exists():
        print("  SKIP", flush=True)
        return 0, 0
    table = pq.read_table(parquet_path)
    print(f"  Cols: {table.column_names}, rows: {table.num_rows}", flush=True)
    kept = 0
    skipped = 0
    bytes_written = 0
    for i in range(table.num_rows):
        if kept >= max_examples:
            break
        row = table.slice(i, 1).to_pylist()[0]
        prompt = clean_text(str(row.get("question", "")))
        response = clean_text(str(row.get("response", "")))
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
            "type": "openorca_code",
            "prompt": prompt,
            "response": response,
            "chars": len(prompt) + len(response),
        }
        out.write(json.dumps(record, ensure_ascii=False) + "\n")
        kept += 1
        bytes_written += record["chars"] + 500
        if kept % 5000 == 0:
            print(f"  kept={kept} skipped={skipped} ({bytes_written/1e6:.1f}MB)", flush=True)
    print(f"  FINAL kept={kept} skipped={skipped} ({bytes_written/1e6:.1f}MB)", flush=True)
    return kept, bytes_written


def process_gpt4_llm(json_path: Path, out, seen: set):
    """Process GPT4-LLM-Cleaned (Alpaca-style)."""
    print(f"\n--- GPT4-LLM-Cleaned ---", flush=True)
    if not json_path.exists():
        print("  SKIP", flush=True)
        return 0, 0
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"  Loaded {len(data)} examples", flush=True)
    kept = 0
    skipped = 0
    bytes_written = 0
    for ex in data:
        prompt = clean_text(str(ex.get("instruction", "")))
        inp = clean_text(str(ex.get("input", "")))
        response = clean_text(str(ex.get("output", "")))
        if inp:
            prompt = prompt + "\n\nInput:\n" + inp
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
            "type": "gpt4_llm_code",
            "prompt": prompt,
            "response": response,
            "chars": len(prompt) + len(response),
        }
        out.write(json.dumps(record, ensure_ascii=False) + "\n")
        kept += 1
        bytes_written += record["chars"] + 500
    print(f"  FINAL kept={kept} skipped={skipped} ({bytes_written/1e6:.1f}MB)", flush=True)
    return kept, bytes_written


def process_oasst(jsonl_gz_path: Path, out, seen: set, max_examples=10000):
    """Process OpenAssistant oasst2 - take parent-child pairs."""
    print(f"\n--- OpenAssistant oasst2 ---", flush=True)
    if not jsonl_gz_path.exists():
        print("  SKIP", flush=True)
        return 0, 0
    # Read all messages and pair user-assistant
    messages = {}  # message_id -> message
    parent_ids = {}  # message_id -> parent_id
    with gzip.open(jsonl_gz_path, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                m = json.loads(line)
                mid = m.get("message_id", "")
                if mid:
                    messages[mid] = m
                    parent_ids[mid] = m.get("parent_id", "")
            except json.JSONDecodeError:
                continue
    print(f"  Loaded {len(messages)} messages", flush=True)
    kept = 0
    skipped = 0
    bytes_written = 0
    for mid, msg in messages.items():
        if kept >= max_examples:
            break
        if msg.get("role") != "assistant":
            continue
        parent_id = msg.get("parent_id", "")
        if not parent_id or parent_id not in messages:
            continue
        parent = messages[parent_id]
        if parent.get("role") != "prompter":
            continue
        prompt = clean_text(str(parent.get("text", "")))
        response = clean_text(str(msg.get("text", "")))
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
            "type": "oasst_code",
            "prompt": prompt,
            "response": response,
            "lang": msg.get("lang", ""),
            "chars": len(prompt) + len(response),
        }
        out.write(json.dumps(record, ensure_ascii=False) + "\n")
        kept += 1
        bytes_written += record["chars"] + 500
    print(f"  FINAL kept={kept} skipped={skipped} ({bytes_written/1e6:.1f}MB)", flush=True)
    return kept, bytes_written


def process_norobots(parquet_path: Path, out, seen: set):
    """Process no_robots dataset."""
    print(f"\n--- no_robots ---", flush=True)
    if not parquet_path.exists():
        print("  SKIP", flush=True)
        return 0, 0
    table = pq.read_table(parquet_path)
    print(f"  Cols: {table.column_names}, rows: {table.num_rows}", flush=True)
    kept = 0
    skipped = 0
    bytes_written = 0
    for i in range(table.num_rows):
        row = table.slice(i, 1).to_pylist()[0]
        messages = row.get("messages", [])
        if not isinstance(messages, list) or len(messages) < 2:
            skipped += 1
            continue
        user_msgs = [m for m in messages if m.get("role") == "user"]
        asst_msgs = [m for m in messages if m.get("role") == "assistant"]
        if not user_msgs or not asst_msgs:
            skipped += 1
            continue
        prompt = clean_text(str(user_msgs[0].get("content", "")))
        response = clean_text(str(asst_msgs[0].get("content", "")))
        if not prompt or not response:
            skipped += 1
            continue
        if len(prompt) + len(response) < 200:
            skipped += 1
            continue
        # Don't require Python - this is general data
        rec_id = make_id(prompt + response[:500])
        if rec_id in seen:
            skipped += 1
            continue
        seen.add(rec_id)
        record = {
            "id": rec_id,
            "type": "norobots",
            "prompt": prompt,
            "response": response,
            "category": row.get("category", ""),
            "chars": len(prompt) + len(response),
        }
        out.write(json.dumps(record, ensure_ascii=False) + "\n")
        kept += 1
        bytes_written += record["chars"] + 500
    print(f"  FINAL kept={kept} skipped={skipped} ({bytes_written/1e6:.1f}MB)", flush=True)
    return kept, bytes_written


def main():
    print(f"[{datetime.now():%H:%M:%S}] Pass 4 verification...", flush=True)
    seen = load_existing_ids()
    OUT_FILE = VERIFIED_DIR / "instruction_data_4.jsonl"
    total_kept = 0
    total_bytes = 0
    with OUT_FILE.open("w", encoding="utf-8") as out:
        k, b = process_openorca(RAW_DIR / "openorca.parquet", out, seen, max_examples=60000)
        total_kept += k
        total_bytes += b

        k, b = process_gpt4_llm(RAW_DIR / "gpt4_llm.json", out, seen)
        total_kept += k
        total_bytes += b

        k, b = process_oasst(RAW_DIR / "oasst.jsonl.gz", out, seen, max_examples=10000)
        total_kept += k
        total_bytes += b

        k, b = process_norobots(RAW_DIR / "norobots.parquet", out, seen)
        total_kept += k
        total_bytes += b

    print(f"\n=== TOTAL ADDED: {total_kept} ({total_bytes/1e6:.1f} MB) ===", flush=True)


if __name__ == "__main__":
    main()
