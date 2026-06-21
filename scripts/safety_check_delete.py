"""Safety check: verify each file in the delete list is truly merged into train_final.jsonl."""
import json
import random
from pathlib import Path

VERIFIED_DIR = Path("/home/z/my-project/dreamgtm/data/verified")
TRAIN_FINAL = Path("/home/z/my-project/dreamgtm/data/train_final.jsonl")

DELETE_LIST = [
    VERIFIED_DIR / "codesearchnet_all_lang.jsonl",
    VERIFIED_DIR / "vulnerabilities.jsonl",
    VERIFIED_DIR / "instruction_data_3.jsonl",
    VERIFIED_DIR / "instruction_data_2.jsonl",
    VERIFIED_DIR / "openhermes_code.jsonl",
    VERIFIED_DIR / "github_code_python.jsonl",
    VERIFIED_DIR / "instruction_data.jsonl",
    VERIFIED_DIR / "codesearchnet_python.jsonl",
]

random.seed(42)


def load_train_final_ids():
    print("Loading train_final.jsonl source_ids...")
    ids = set()
    count = 0
    with TRAIN_FINAL.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                sid = rec.get("metadata", {}).get("source_id", "")
                if sid:
                    ids.add(sid)
                count += 1
            except json.JSONDecodeError:
                continue
    print(f"Loaded {len(ids):,} unique source_ids from {count:,} records")
    return ids


def sample_ids_from_file(path, n=10):
    ids = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                rid = rec.get("id", "")
                if rid:
                    ids.append(rid)
            except json.JSONDecodeError:
                continue
    if len(ids) <= n:
        return ids
    return random.sample(ids, n)


def main():
    print("=" * 70)
    print("SAFETY CHECK: verifying each file is merged into train_final.jsonl")
    print("=" * 70)

    train_final_ids = load_train_final_ids()

    print(f"\n{'File':<50} {'Status':<12} {'Found':<10}")
    print("-" * 75)
    all_safe = True
    for path in DELETE_LIST:
        if not path.exists():
            print(f"  {path.name:<48} MISSING")
            continue
        sample = sample_ids_from_file(path, 10)
        found = sum(1 for sid in sample if sid in train_final_ids)
        safe = found == len(sample)
        if not safe:
            all_safe = False
        print(f"  {path.name:<48} {'SAFE' if safe else 'NOT MERGED'} ({found}/{len(sample)})")

    print("\n" + "=" * 70)
    print("ALL FILES SAFE TO DELETE" if all_safe else "SOME FILES NOT MERGED — REVIEW")
    print("=" * 70)


if __name__ == "__main__":
    main()
