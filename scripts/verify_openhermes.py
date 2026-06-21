"""
Verify OpenHermes 2.5 - 1M examples.
Filter: keep only examples that contain Python code blocks (parses cleanly).
This gives us the highest-quality coding/reasoning subset.
"""
import os
import ast
import json
import time
import hashlib
import ijson  # streaming parser
from pathlib import Path
from datetime import datetime

IN_FILE = Path("/home/z/my-project/dreamgtm/data/raw/openhermes/openhermes.json")
OUT_FILE = Path("/home/z/my-project/dreamgtm/data/verified/openhermes_code.jsonl")
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

MAX_EXAMPLES = 200_000
MIN_CHARS = 200
MAX_CHARS = 16000


def make_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def has_valid_python(text: str) -> bool:
    """Check if text contains a ```python block that parses."""
    if "```python" not in text and "```py\n" not in text:
        return False
    parts = text.split("```")
    for i in range(1, len(parts), 2):
        if i >= len(parts):
            break
        block = parts[i]
        # Strip language tag
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


def extract_conversation(example: dict) -> tuple[str, str]:
    """Pull (prompt, response) from OpenHermes example."""
    # OpenHermes uses 'conversations' field
    convs = example.get("conversations", [])
    if isinstance(convs, list) and convs:
        # Take first user message as prompt, last assistant as response
        user_msgs = [c for c in convs if c.get("from") == "human" or c.get("from") == "user"]
        asst_msgs = [c for c in convs if c.get("from") == "gpt" or c.get("from") == "assistant"]
        if user_msgs and asst_msgs:
            prompt = user_msgs[0].get("value", "")
            response = asst_msgs[-1].get("value", "")
            return prompt, response
    # Fallback to instruction/output
    return example.get("instruction", ""), example.get("output", "")


def main():
    print(f"[{datetime.now():%H:%M:%S}] Verifying OpenHermes 2.5 (streaming)...")
    if not IN_FILE.exists():
        print(f"ERROR: {IN_FILE} not found")
        return

    # Install ijson if needed
    try:
        import ijson
    except ImportError:
        import subprocess
        subprocess.check_call(["/home/z/.venv/bin/python3", "-m", "pip", "install", "ijson"])
        import ijson

    kept = 0
    skipped = 0
    seen_ids = set()
    bytes_written = 0
    start = time.time()
    categories = {}

    with OUT_FILE.open("w", encoding="utf-8") as f:
        with IN_FILE.open("rb") as fb:
            # ijson streams objects one at a time - low memory
            for example in ijson.items(fb, "item"):
                if kept >= MAX_EXAMPLES:
                    break
                prompt, response = extract_conversation(example)
                if not prompt or not response:
                    skipped += 1
                    continue
                if len(prompt) + len(response) < MIN_CHARS:
                    skipped += 1
                    continue
                if len(response) > MAX_CHARS:
                    skipped += 1
                    continue
                # Require valid Python in response
                if not has_valid_python(response):
                    skipped += 1
                    continue
                rec_id = make_id(prompt + response[:500])
                if rec_id in seen_ids:
                    skipped += 1
                    continue
                seen_ids.add(rec_id)

                source = example.get("source", "")
                category = example.get("category", "")
                categories[source] = categories.get(source, 0) + 1

                record = {
                    "id": rec_id,
                    "type": "openhermes_code",
                    "source": source,
                    "category": category,
                    "prompt": prompt,
                    "response": response,
                    "chars": len(prompt) + len(response),
                }
                line = json.dumps(record, ensure_ascii=False)
                f.write(line + "\n")
                kept += 1
                bytes_written += len(line) + 1

                if kept % 10000 == 0:
                    elapsed = time.time() - start
                    rate = kept / elapsed if elapsed > 0 else 0
                    print(f"  kept={kept} skipped={skipped} | "
                          f"{bytes_written/1e6:.1f} MB | "
                          f"{rate:.0f} ex/s", flush=True)

    elapsed = time.time() - start
    print(f"\nDONE: {kept} verified OpenHermes code examples ({bytes_written/1e6:.1f} MB)")
    print(f"Skipped: {skipped}")
    print(f"Time: {elapsed:.0f}s")
    print(f"\nTop categories:")
    for src, cnt in sorted(categories.items(), key=lambda x: -x[1])[:10]:
        print(f"  {src}: {cnt}")


if __name__ == "__main__":
    main()
