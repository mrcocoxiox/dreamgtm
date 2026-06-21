"""
Ultra-minimal split — writes to gzip in a single stream.
Must complete in <50 seconds for 1.35M records.
"""
import json
import gzip
import sys
from pathlib import Path
from datetime import datetime

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data"
EVAL_DIR = BASE / "eval"
TRAIN_FINAL = DATA_DIR / "train_final.jsonl"
TRAIN_SPLIT_GZ = DATA_DIR / "train_split.jsonl.gz"
VAL_SPLIT = DATA_DIR / "val_split.jsonl"
MBPP_HELDOUT = EVAL_DIR / "mbpp_heldout.jsonl"
SEC_HELDOUT = EVAL_DIR / "security_patch_eval.jsonl"

VAL_EVERY = 33
SEC_HELDOUT_N = 100
VULN_HELDOUT_N = 100

def main():
    t0 = datetime.now()
    print(f"[{t0:%H:%M:%S}] Ultra-minimal split", flush=True)

    sec_seen = 0
    vuln_seen = 0
    non_ho = 0
    c_train = c_val = c_mbpp = c_sec = c_vuln = 0

    fin = TRAIN_FINAL.open("r", encoding="utf-8", buffering=1024*1024)
    ftrain = gzip.open(TRAIN_SPLIT_GZ, "wt", encoding="utf-8")
    fval = VAL_SPLIT.open("w", encoding="utf-8", buffering=1024*1024)
    fmbpp = MBPP_HELDOUT.open("w", encoding="utf-8", buffering=1024*1024)
    fsec = SEC_HELDOUT.open("w", encoding="utf-8", buffering=1024*1024)

    try:
        for line in fin:
            try:
                rec = json.loads(line)
            except:
                continue

            stype = rec.get("metadata", {}).get("source_type", "")
            msgs = rec.get("messages", [])
            stripped = {
                "messages": [m for m in msgs if m.get("role") != "system"],
                "metadata": rec.get("metadata", {}),
            }
            out = json.dumps(stripped, ensure_ascii=False) + "\n"

            if stype == "mbpp_problem":
                fmbpp.write(line)
                c_mbpp += 1
            elif stype == "security_patch":
                sec_seen += 1
                if sec_seen <= SEC_HELDOUT_N:
                    fsec.write(line)
                    c_sec += 1
                else:
                    non_ho += 1
                    if non_ho % VAL_EVERY == 0:
                        fval.write(out)
                        c_val += 1
                    else:
                        ftrain.write(out)
                        c_train += 1
            elif stype == "vulnerability":
                vuln_seen += 1
                if vuln_seen <= VULN_HELDOUT_N:
                    fsec.write(line)
                    c_vuln += 1
                else:
                    non_ho += 1
                    if non_ho % VAL_EVERY == 0:
                        fval.write(out)
                        c_val += 1
                    else:
                        ftrain.write(out)
                        c_train += 1
            else:
                non_ho += 1
                if non_ho % VAL_EVERY == 0:
                    fval.write(out)
                    c_val += 1
                else:
                    ftrain.write(out)
                    c_train += 1
    finally:
        fin.close()
        ftrain.close()
        fval.close()
        fmbpp.close()
        fsec.close()

    elapsed = (datetime.now() - t0).total_seconds()
    print(f"\n✅ Done in {elapsed:.0f}s", flush=True)
    print(f"  train: {c_train:,} ({TRAIN_SPLIT_GZ.stat().st_size/1e6:.0f} MB)", flush=True)
    print(f"  val: {c_val:,} ({VAL_SPLIT.stat().st_size/1e6:.0f} MB)", flush=True)
    print(f"  mbpp: {c_mbpp:,} | sec_eval: {c_sec + c_vuln}", flush=True)

if __name__ == "__main__":
    main()
