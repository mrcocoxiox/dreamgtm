"""
DreamGTM Tokenizer Trainer
==========================
Trains a 32K BPE tokenizer on train_final.jsonl using HuggingFace tokenizers.

Special tokens (32 total, matching architecture.py SPECIAL_TOKENS):
  <PAD> <BOS> <EOS> <UNK>
  <SYSTEM> </SYSTEM> <USER> </USER> <ASSISTANT> </ASSISTANT>
  <THINK> </THINK> <CODE> </CODE> <PATCH> </PATCH>
  <TEST> </TEST> <SEARCH> </SEARCH> <TOOL> </TOOL>
  <RESULT> </RESULT> <SECURITY_REPORT> </SECURITY_REPORT>
  <MICROCODE> </MICROCODE> <VERIFY> </VERIFY>
  <DREAM_STATE> </DREAM_STATE>

Output: data/dreamgtm.tokenizer.json (HuggingFace format, ~2 MB)
"""
import json
import os
import sys
import time
import random
from pathlib import Path
from datetime import datetime

# Ensure tokenizers is available
try:
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
    from tokenizers.models import BPE
except ImportError:
    print("Installing tokenizers...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tokenizers"])
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
    from tokenizers.models import BPE

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data"
TRAIN_FINAL = DATA_DIR / "train_final.jsonl"
OUT_FILE = DATA_DIR / "dreamgtm.tokenizer.json"

# 32 special tokens (ids 0-31)
SPECIAL_TOKENS = [
    "<PAD>",      # 0
    "<BOS>",      # 1
    "<EOS>",      # 2
    "<UNK>",      # 3
    "<SYSTEM>",   # 4
    "</SYSTEM>",  # 5
    "<USER>",     # 6
    "</USER>",    # 7
    "<ASSISTANT>",# 8
    "</ASSISTANT>",# 9
    "<THINK>",    # 10
    "</THINK>",   # 11
    "<CODE>",     # 12
    "</CODE>",    # 13
    "<PATCH>",    # 14
    "</PATCH>",   # 15
    "<TEST>",     # 16
    "</TEST>",    # 17
    "<SEARCH>",   # 18
    "</SEARCH>",  # 19
    "<TOOL>",     # 20
    "</TOOL>",    # 21
    "<RESULT>",   # 22
    "</RESULT>",  # 23
    "<SECURITY_REPORT>",  # 24
    "</SECURITY_REPORT>", # 25
    "<MICROCODE>",        # 26
    "</MICROCODE>",       # 27
    "<VERIFY>",           # 28
    "</VERIFY>",          # 29
    "<DREAM_STATE>",      # 30
    "</DREAM_STATE>",     # 31
]

VOCAB_SIZE = 32000  # Matches dreamgtm_1b.yaml; all configs will use this
SAMPLE_SIZE = 200000  # 200K records is enough for stable BPE merges


def extract_text_from_record(rec: dict) -> str:
    """Extract user + assistant text from a chat record (skip system prompt)."""
    msgs = rec.get("messages", [])
    parts = []
    for msg in msgs:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role in ("user", "assistant") and content:
            parts.append(content)
    return "\n\n".join(parts)


def generate_corpus(sample_path: Path, n: int = SAMPLE_SIZE):
    """Stream-read train_final.jsonl, sample n records, write their text to a temp file."""
    print(f"[{datetime.now():%H:%M:%S}] Sampling {n:,} records from {TRAIN_FINAL.name}...")
    
    # First pass: count total records
    total = 0
    with TRAIN_FINAL.open("r", encoding="utf-8") as f:
        for _ in f:
            total += 1
    print(f"  Total records: {total:,}")
    
    # Reservoir sampling to pick n records
    random.seed(42)
    indices_to_keep = set(random.sample(range(total), min(n, total)))
    
    print(f"  Selected {len(indices_to_keep):,} records for tokenizer training")
    
    # Second pass: write selected records' text
    written = 0
    with TRAIN_FINAL.open("r", encoding="utf-8") as fin, \
         sample_path.open("w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            if i in indices_to_keep:
                try:
                    rec = json.loads(line)
                    text = extract_text_from_record(rec)
                    if text and len(text) > 50:
                        fout.write(text + "\n\n")
                        written += 1
                except json.JSONDecodeError:
                    continue
            if i % 200000 == 0 and i > 0:
                print(f"  Scanned {i:,}/{total:,}...")
    
    print(f"  Written {written:,} text blocks to {sample_path.name}")
    return written


def train_tokenizer(corpus_path: Path):
    """Train BPE tokenizer on the corpus."""
    print(f"\n[{datetime.now():%H:%M:%S}] Training BPE tokenizer (vocab={VOCAB_SIZE})...")
    
    # Initialize BPE with empty vocab
    tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
    
    # Byte-level pre-tokenizer (handles all UTF-8, no <UNK> ever)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False,
        use_regex=True
    )
    
    # Byte-level decoder (inverse of pre-tokenizer)
    tokenizer.decoder = decoders.ByteLevel()
    
    # Post-processor: add BOS/EOS automatically (we'll handle this in wrapper instead)
    # tokenizer.post_processor = processors.TemplateProcessing(...)
    
    # Trainer
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    
    # Train
    start = time.time()
    tokenizer.train([str(corpus_path)], trainer)
    elapsed = time.time() - start
    
    vocab_size_actual = tokenizer.get_vocab_size()
    print(f"  Trained in {elapsed:.0f}s")
    print(f"  Actual vocab size: {vocab_size_actual}")
    
    # Verify special tokens have correct IDs
    print(f"\n  Special token IDs:")
    for tok in SPECIAL_TOKENS[:10]:
        tid = tokenizer.token_to_id(tok)
        print(f"    {tok}: {tid}")
    print(f"    ... ({len(SPECIAL_TOKENS)} total)")
    
    return tokenizer


def validate_tokenizer(tokenizer, n: int = 1000):
    """Validate tokenizer: encode/decode roundtrip on n random records."""
    print(f"\n[{datetime.now():%H:%M:%S}] Validating tokenizer (roundtrip test on {n} records)...")
    
    # Pick n random records
    random.seed(123)
    total = 0
    with TRAIN_FINAL.open("r", encoding="utf-8") as f:
        for _ in f:
            total += 1
    indices_to_test = set(random.sample(range(total), min(n, total)))
    
    faithful = 0
    total_tested = 0
    with TRAIN_FINAL.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i not in indices_to_test:
                continue
            try:
                rec = json.loads(line)
                text = extract_text_from_record(rec)
                if not text or len(text) < 50:
                    continue
                
                # Encode + decode
                ids = tokenizer.encode(text).ids
                decoded = tokenizer.decode(ids)
                
                # Check faithfulness (allow minor whitespace differences)
                if decoded.strip() == text.strip():
                    faithful += 1
                elif len(decoded) > 0 and len(text) > 0:
                    # Check character-level similarity
                    common = sum(1 for a, b in zip(decoded, text) if a == b)
                    sim = common / max(len(decoded), len(text))
                    if sim > 0.99:
                        faithful += 1
                
                total_tested += 1
            except Exception as e:
                print(f"  ERROR on record {i}: {e}")
    
    fidelity = faithful / total_tested if total_tested > 0 else 0
    print(f"  Roundtrip fidelity: {faithful}/{total_tested} = {fidelity*100:.1f}%")
    print(f"  Target: ≥99.5%")
    
    if fidelity < 0.995:
        print(f"  ⚠️  Fidelity below target — tokenizer may need more training data")
    else:
        print(f"  ✅ Fidelity meets target")
    
    return fidelity


def main():
    print("=" * 70)
    print("DreamGTM Tokenizer Trainer")
    print(f"Vocab size: {VOCAB_SIZE:,}")
    print(f"Special tokens: {len(SPECIAL_TOKENS)}")
    print("=" * 70)
    
    if not TRAIN_FINAL.exists():
        print(f"ERROR: {TRAIN_FINAL} not found")
        return
    
    # Step 1: Generate corpus sample
    corpus_path = DATA_DIR / "tokenizer_corpus.txt"
    n_written = generate_corpus(corpus_path, SAMPLE_SIZE)
    
    # Step 2: Train tokenizer
    tokenizer = train_tokenizer(corpus_path)
    
    # Step 3: Validate
    fidelity = validate_tokenizer(tokenizer, n=1000)
    
    # Step 4: Save
    print(f"\n[{datetime.now():%H:%M:%S}] Saving tokenizer to {OUT_FILE}...")
    tokenizer.save(str(OUT_FILE))
    
    # Step 5: Clean up corpus
    corpus_path.unlink()
    
    size_mb = OUT_FILE.stat().st_size / 1e6
    print(f"\n{'=' * 70}")
    print(f"DONE: Tokenizer saved to {OUT_FILE}")
    print(f"  Size: {size_mb:.1f} MB")
    print(f"  Vocab: {tokenizer.get_vocab_size():,}")
    print(f"  Fidelity: {fidelity*100:.1f}%")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
