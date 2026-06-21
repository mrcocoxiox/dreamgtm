"""
Master verification pass 2 - additional datasets:
- CodeSearchNet all languages (already have Python, this adds JS/Java/Go/etc.)
- WizardLM evol_instruct_70k
- tatsu-lab/alpaca
- ultrachat_200k
- MBPP train/test/validation

Each example goes through:
1. UTF-8 encoding check
2. Length filter
3. AST validation (for Python code blocks)
4. Hash deduplication against ALL existing verified data
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


def load_existing_ids(jsonl_files):
    """Load all IDs we've already kept to avoid duplicates across sources."""
    ids = set()
    for f in jsonl_files:
        if not f.exists():
            continue
        with f.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                    # Use multiple keys for dedup
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
    print(f"Loaded {len(ids)} existing IDs for dedup")
    return ids


def process_wizardlm(json_path: Path, out, seen: set):
    """Process WizardLM evol_instruct_70k JSON."""
    print(f"\n--- WizardLM evol_instruct_70k ---")
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"  Loaded {len(data)} examples")
    kept = 0
    skipped = 0
    bytes_written = 0
    for ex in data:
        prompt = clean_text(ex.get("instruction", ""))
        inp = clean_text(ex.get("input", ""))
        response = clean_text(ex.get("output", ""))
        if inp:
            prompt = prompt + "\n\nInput:\n" + inp
        if not prompt or not response:
            skipped += 1
            continue
        if len(prompt) + len(response) < 200:
            skipped += 1
            continue
        # Require Python in response (this is for code-focused subset)
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
            "type": "wizardlm_evol_instruct",
            "prompt": prompt,
            "response": response,
            "chars": len(prompt) + len(response),
        }
        line = json.dumps(record, ensure_ascii=False)
        out.write(line + "\n")
        kept += 1
        bytes_written += len(line) + 1
    print(f"  kept={kept} skipped={skipped} ({bytes_written/1e6:.1f} MB)")
    return kept, bytes_written


def process_alpaca(parquet_path: Path, out, seen: set):
    """Process tatsu-lab/alpaca - keep only code-containing examples."""
    print(f"\n--- tatsu-lab/alpaca ---")
    table = pq.read_table(parquet_path)
    print(f"  Cols: {table.column_names}, rows: {table.num_rows}")
    kept = 0
    skipped = 0
    bytes_written = 0
    for i in range(table.num_rows):
        row = table.slice(i, 1).to_pylist()[0]
        prompt = clean_text(str(row.get("instruction", "")))
        inp = clean_text(str(row.get("input", "")))
        response = clean_text(str(row.get("output", "")))
        if inp:
            prompt = prompt + "\n\nInput:\n" + inp
        if not prompt or not response:
            skipped += 1
            continue
        if len(prompt) + len(response) < 200:
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
            "type": "alpaca_code",
            "prompt": prompt,
            "response": response,
            "chars": len(prompt) + len(response),
        }
        line = json.dumps(record, ensure_ascii=False)
        out.write(line + "\n")
        kept += 1
        bytes_written += len(line) + 1
    print(f"  kept={kept} skipped={skipped} ({bytes_written/1e6:.1f} MB)")
    return kept, bytes_written


def process_ultrachat(parquet_path: Path, out, seen: set):
    """Process UltraChat - keep only code-containing conversations."""
    print(f"\n--- ultrachat_200k ---")
    table = pq.read_table(parquet_path)
    print(f"  Cols: {table.column_names}, rows: {table.num_rows}")
    kept = 0
    skipped = 0
    bytes_written = 0
    for i in range(table.num_rows):
        row = table.slice(i, 1).to_pylist()[0]
        messages = row.get("messages", [])
        if not isinstance(messages, list) or len(messages) < 2:
            skipped += 1
            continue
        # Take first user + first assistant
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
            "type": "ultrachat_code",
            "prompt": prompt,
            "response": response,
            "chars": len(prompt) + len(response),
        }
        line = json.dumps(record, ensure_ascii=False)
        out.write(line + "\n")
        kept += 1
        bytes_written += len(line) + 1
    print(f"  kept={kept} skipped={skipped} ({bytes_written/1e6:.1f} MB)")
    return kept, bytes_written


def process_mbpp(parquet_path: Path, split: str, out, seen: set):
    """Process MBPP - beginner programming problems with verified solutions."""
    print(f"\n--- MBPP {split} ---")
    table = pq.read_table(parquet_path)
    print(f"  Cols: {table.column_names}, rows: {table.num_rows}")
    kept = 0
    skipped = 0
    bytes_written = 0
    for i in range(table.num_rows):
        row = table.slice(i, 1).to_pylist()[0]
        prompt = clean_text(str(row.get("text", "") or row.get("prompt", "")))
        if not prompt:
            skipped += 1
            continue
        code = ""
        # MBPP has 'code' or 'test_list' field
        c = row.get("code", "")
        if isinstance(c, str) and c.strip():
            code = c.strip()
        else:
            # Try 'solutions' field
            sols = row.get("solutions", "")
            if isinstance(sols, str):
                try:
                    sols = json.loads(sols)
                except json.JSONDecodeError:
                    sols = [sols]
            for s in sols:
                s = clean_text(str(s))
                if not s:
                    continue
                try:
                    ast.parse(s)
                    code = s
                    break
                except (SyntaxError, ValueError):
                    continue
        if not code:
            skipped += 1
            continue
        # Verify code parses
        try:
            ast.parse(code)
        except (SyntaxError, ValueError):
            skipped += 1
            continue
        # Build test info
        test_list = row.get("test_list", [])
        if isinstance(test_list, list):
            tests = "\n".join(str(t) for t in test_list)
        else:
            tests = ""
        full_response = f"```python\n{code}\n```"
        if tests:
            full_response += f"\n\nTest cases:\n```python\n{tests}\n```"
        rec_id = make_id(prompt + code)
        if rec_id in seen:
            skipped += 1
            continue
        seen.add(rec_id)
        record = {
            "id": rec_id,
            "type": "mbpp_problem",
            "dataset": "mbpp",
            "split": split,
            "prompt": prompt,
            "response": full_response,
            "chars": len(prompt) + len(full_response),
        }
        line = json.dumps(record, ensure_ascii=False)
        out.write(line + "\n")
        kept += 1
        bytes_written += len(line) + 1
    print(f"  kept={kept} skipped={skipped} ({bytes_written/1e6:.1f} MB)")
    return kept, bytes_written


def process_codesearchnet_all(parquet_path: Path, out, seen: set):
    """Process CodeSearchNet all-languages - keep Python + JS + Java + Go with docstrings."""
    print(f"\n--- CodeSearchNet all (multi-lang) ---")
    table = pq.read_table(parquet_path)
    print(f"  Cols: {table.column_names}, rows: {table.num_rows}")
    kept = 0
    skipped = 0
    bytes_written = 0
    for i in range(table.num_rows):
        row = table.slice(i, 1).to_pylist()[0]
        lang = (row.get("language") or "").lower()
        if lang not in {"python", "javascript", "java", "go"}:
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
        # For Python, AST validate
        if lang == "python":
            try:
                ast.parse(code)
            except (SyntaxError, ValueError):
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
    print(f"  kept={kept} skipped={skipped} ({bytes_written/1e6:.1f} MB)")
    return kept, bytes_written


def main():
    print(f"[{datetime.now():%H:%M:%S}] Pass 2 verification - additional datasets...")
    existing_files = [
        VERIFIED_DIR / "codesearchnet_python.jsonl",
        VERIFIED_DIR / "github_code_python.jsonl",
        VERIFIED_DIR / "instruction_data.jsonl",
        VERIFIED_DIR / "openhermes_code.jsonl",
        VERIFIED_DIR / "python_source.jsonl",
        VERIFIED_DIR / "python_docs.jsonl",
        VERIFIED_DIR / "security_docs.jsonl",
        VERIFIED_DIR / "vulnerabilities.jsonl",
    ]
    seen = load_existing_ids(existing_files)

    OUT_FILE = VERIFIED_DIR / "instruction_data_2.jsonl"
    total_kept = 0
    total_bytes = 0
    with OUT_FILE.open("w", encoding="utf-8") as out:
        # 1. WizardLM
        wizard_path = RAW_DIR / "more_instruction" / "wizardlm.json"
        if wizard_path.exists():
            k, b = process_wizardlm(wizard_path, out, seen)
            total_kept += k
            total_bytes += b

        # 2. Alpaca
        alpaca_path = RAW_DIR / "more_instruction" / "alpaca.parquet"
        if alpaca_path.exists():
            k, b = process_alpaca(alpaca_path, out, seen)
            total_kept += k
            total_bytes += b

        # 3. UltraChat
        ultra_path = RAW_DIR / "more_instruction" / "ultrachat.parquet"
        if ultra_path.exists():
            k, b = process_ultrachat(ultra_path, out, seen)
            total_kept += k
            total_bytes += b

        # 4. MBPP splits
        for split in ["train", "validation", "test"]:
            mbpp_path = RAW_DIR / "mbpp" / f"mbpp_{split}.parquet"
            if mbpp_path.exists():
                k, b = process_mbpp(mbpp_path, split, out, seen)
                total_kept += k
                total_bytes += b

        # 5. CodeSearchNet all languages
        for shard in ["all_train_00000.parquet", "all_train_00001.parquet"]:
            cs_path = RAW_DIR / "codesearchnet" / shard
            if cs_path.exists():
                k, b = process_codesearchnet_all(cs_path, out, seen)
                total_kept += k
                total_bytes += b

    print(f"\n=== TOTAL ADDED: {total_kept} examples ({total_bytes/1e6:.1f} MB) ===")
    print(f"Output: {OUT_FILE}")


if __name__ == "__main__":
    main()
