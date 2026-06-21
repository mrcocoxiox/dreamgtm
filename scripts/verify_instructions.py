"""
Verify instruction datasets:
- HuggingFaceH4/CodeAlpaca_20K
- iamtarun/python_code_instructions_18k_alpaca
- garage-bAInd/Open-Platypus
- Yukang/LongAlpaca-12k
- bigcode/self-oss-instruct-sc2-exec-filter-50k

Output: verified/instruction_data.jsonl
"""
import os
import ast
import json
import time
import hashlib
import pyarrow.parquet as pq
from pathlib import Path
from datetime import datetime

RAW_DIR = Path("/home/z/my-project/dreamgtm/data/raw")
OUT_FILE = Path("/home/z/my-project/dreamgtm/data/verified/instruction_data.jsonl")
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)


def make_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return s.replace("\r\n", "\n").strip()


def has_valid_python_block(text: str) -> bool:
    """Check if text contains at least one ```python block that parses."""
    if "```python" not in text and "```py" not in text and "```" not in text:
        return False
    # Extract code blocks
    blocks = []
    parts = text.split("```")
    for i in range(1, len(parts), 2):
        if i < len(parts):
            block = parts[i]
            # Strip language tag
            if block.startswith("python") or block.startswith("py"):
                block = block.split("\n", 1)[1] if "\n" in block else block[6:]
            blocks.append(block)
    if not blocks:
        return False
    # At least one block should be valid Python
    for b in blocks:
        try:
            ast.parse(b)
            return True
        except (SyntaxError, ValueError):
            continue
    return False


def process_parquet(parquet_path: Path, dataset_name: str, prompt_col: str, response_col: str):
    """Generic parquet processor for Alpaca-style datasets."""
    if not parquet_path.exists():
        print(f"  SKIP (missing): {parquet_path}")
        return
    print(f"  Processing {parquet_path.name} ({dataset_name})...")
    table = pq.read_table(parquet_path)
    cols = table.column_names
    # Auto-detect columns if not provided
    if prompt_col not in cols:
        for cand in ["prompt", "instruction", "question", "input"]:
            if cand in cols:
                prompt_col = cand
                break
    if response_col not in cols:
        for cand in ["completion", "output", "answer", "response", "text"]:
            if cand in cols:
                response_col = cand
                break
    print(f"    Cols: {cols}, using prompt='{prompt_col}' response='{response_col}'")
    kept = 0
    skipped = 0
    for i in range(table.num_rows):
        row = table.slice(i, 1).to_pylist()[0]
        prompt = clean_text(str(row.get(prompt_col, "")))
        response = clean_text(str(row.get(response_col, "")))
        if not prompt or not response:
            skipped += 1
            continue
        if len(prompt) < 10 or len(response) < 50:
            skipped += 1
            continue
        # For code datasets, require valid Python in response
        if "code" in dataset_name.lower() or "alpaca" in dataset_name.lower():
            if not has_valid_python_block(response):
                # Still keep non-code if dataset is Platypus (general)
                if "platypus" not in dataset_name.lower():
                    skipped += 1
                    continue
        rec_id = make_id(prompt + response)
        record = {
            "id": rec_id,
            "type": "instruction",
            "dataset": dataset_name,
            "prompt": prompt,
            "response": response,
            "chars": len(prompt) + len(response),
        }
        yield record
        kept += 1
    print(f"    kept={kept} skipped={skipped}")


def process_longalpaca_json(json_path: Path):
    """Process LongAlpaca-12k JSON file."""
    if not json_path.exists():
        print(f"  SKIP (missing): {json_path}")
        return
    print(f"  Processing {json_path.name}...")
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"    Loaded {len(data)} examples")
    kept = 0
    skipped = 0
    for ex in data:
        prompt = clean_text(ex.get("instruction", "") or ex.get("prompt", ""))
        response = clean_text(ex.get("output", "") or ex.get("response", ""))
        if not prompt or not response:
            skipped += 1
            continue
        if len(prompt) < 10 or len(response) < 50:
            skipped += 1
            continue
        rec_id = make_id(prompt + response)
        record = {
            "id": rec_id,
            "type": "instruction",
            "dataset": "longalpaca_12k",
            "prompt": prompt,
            "response": response,
            "chars": len(prompt) + len(response),
        }
        yield record
        kept += 1
    print(f"    kept={kept} skipped={skipped}")


def process_apps_jsonl(jsonl_path: Path):
    """Process APPS train.jsonl - programming contest problems with solutions."""
    if not jsonl_path.exists():
        print(f"  SKIP (missing): {jsonl_path}")
        return
    print(f"  Processing {jsonl_path.name}...")
    kept = 0
    skipped = 0
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            problem = clean_text(ex.get("question", ""))
            solutions = ex.get("solutions", "")
            if isinstance(solutions, str):
                try:
                    sols = json.loads(solutions)
                except json.JSONDecodeError:
                    sols = [solutions]
            else:
                sols = solutions
            if not problem or not sols:
                skipped += 1
                continue
            # Take first solution that parses
            chosen = None
            for s in sols[:3]:
                s_clean = clean_text(str(s))
                if not s_clean:
                    continue
                try:
                    ast.parse(s_clean)
                    chosen = s_clean
                    break
                except (SyntaxError, ValueError):
                    continue
            if not chosen:
                skipped += 1
                continue
            rec_id = make_id(problem + chosen)
            record = {
                "id": rec_id,
                "type": "apps_problem",
                "dataset": "apps",
                "difficulty": ex.get("difficulty", ""),
                "prompt": problem,
                "response": "```python\n" + chosen + "\n```",
                "chars": len(problem) + len(chosen),
            }
            yield record
            kept += 1
            if kept >= 8000:
                break  # cap
    print(f"    kept={kept} skipped={skipped}")


def process_self_oss(parquet_path: Path):
    """Process self-oss-instruct - execution-filtered instructions."""
    if not parquet_path.exists():
        print(f"  SKIP (missing): {parquet_path}")
        return
    print(f"  Processing {parquet_path.name}...")
    table = pq.read_table(parquet_path)
    cols = table.column_names
    print(f"    Cols: {cols}")
    # Find prompt/response columns
    prompt_col = "instruction" if "instruction" in cols else ("prompt" if "prompt" in cols else cols[0])
    response_col = "response" if "response" in cols else ("output" if "output" in cols else cols[1])
    kept = 0
    skipped = 0
    for i in range(table.num_rows):
        row = table.slice(i, 1).to_pylist()[0]
        prompt = clean_text(str(row.get(prompt_col, "")))
        response = clean_text(str(row.get(response_col, "")))
        if not prompt or not response:
            skipped += 1
            continue
        if len(prompt) < 10 or len(response) < 50:
            skipped += 1
            continue
        if not has_valid_python_block(response):
            skipped += 1
            continue
        rec_id = make_id(prompt + response)
        record = {
            "id": rec_id,
            "type": "instruction",
            "dataset": "self_oss_instruct",
            "prompt": prompt,
            "response": response,
            "chars": len(prompt) + len(response),
        }
        yield record
        kept += 1
    print(f"    kept={kept} skipped={skipped}")


def main():
    print(f"[{datetime.now():%H:%M:%S}] Verifying instruction datasets...")
    sources = []

    # 1. CodeAlpaca 20K
    sources.append(("codealpaca_20k",
                    "/home/z/my-project/dreamgtm/data/raw/instruction_data/codealpaca.parquet",
                    "prompt", "completion"))

    # 2. Python Instructions 18K
    sources.append(("py_inst_18k",
                    "/home/z/my-project/dreamgtm/data/raw/instruction_data/py_inst_18k.parquet",
                    "instruction", "output"))

    # 3. Open-Platypus (general)
    sources.append(("open_platypus",
                    "/home/z/my-project/dreamgtm/data/raw/instruction_data/platypus.parquet",
                    "instruction", "output"))

    # 4. LongAlpaca
    sources.append(("longalpaca_12k",
                    "/home/z/my-project/dreamgtm/data/raw/instruction_data/longalpaca.json",
                    None, None))

    # 5. APPS
    sources.append(("apps",
                    "/home/z/my-project/dreamgtm/data/raw/apps/train.jsonl",
                    None, None))

    # 6. self-oss-instruct
    sources.append(("self_oss_instruct",
                    "/home/z/my-project/dreamgtm/data/raw/self_oss/self_oss.parquet",
                    None, None))

    total = 0
    bytes_written = 0
    with OUT_FILE.open("w", encoding="utf-8") as f:
        for name, path, pcol, rcol in sources:
            p = Path(path)
            print(f"\n--- {name} ---")
            if name == "longalpaca_12k":
                gen = process_longalpaca_json(p)
            elif name == "apps":
                gen = process_apps_jsonl(p)
            elif name == "self_oss_instruct":
                gen = process_self_oss(p)
            else:
                gen = process_parquet(p, name, pcol or "", rcol or "")
            for record in gen:
                line = json.dumps(record, ensure_ascii=False)
                f.write(line + "\n")
                total += 1
                bytes_written += len(line) + 1

    print(f"\nDONE: {total} verified instruction examples ({bytes_written/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
