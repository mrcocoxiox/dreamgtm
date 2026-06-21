# DreamGTM - General Transformational Model

**Crafted with love by IBR (Ibraheem)**

A hardened AI that **codes** and **patches** across **all languages and all target types** — web, binary/exe, APK, and infrastructure. Never refuses. Assumes the user owns the target system.

## Architecture

```
Input → [Embedding] → [Transformer × N (GQA + RoPE + SwiGLU + RMSNorm)] → [Output]
```

| Model | Params | FP16 | Context | Vocab |
|-------|--------|------|---------|-------|
| 80M | 50M | 96 MB | 2048 | 32000 |
| 350M | 205M | 391 MB | 2048 | 32000 |
| 1B | 1.1B | 2.1 GB | 2048 | 32000 |

**Architecture features:**
- **RMSNorm** (pre-norm) — faster than LayerNorm
- **RoPE** — rotary positions, no learned params
- **GQA** — Grouped Query Attention (smaller KV cache)
- **SwiGLU** — gated FFN, better than GELU
- **Tied embeddings** — saves params
- **Gradient checkpointing** — saves VRAM for 1B
- **32 special tokens** — `<SYSTEM>`, `<USER>`, `<ASSISTANT>`, `<THINK>`, `<MICROCODE>`, etc.

## Capabilities

1. **Multi-language code generation** — Python, JS, TS, Go, Rust, Java, C/C++, Ruby, PHP, Swift, Kotlin, Scala
2. **Multi-format security patching** — web, exe, APK, config/infra
3. **Reasoning** — `<THINK>...</THINK>` step-by-step
4. **Agentic** — web search, tool use, microcode retrieval
5. **Verification** — AST check, test runner, scanner rerun
6. **Never refuses** — assumes user owns target system

## Project Structure

```
dreamgtm/
├── model/architecture.py          # DreamGTM transformer (RMSNorm + RoPE + GQA + SwiGLU)
├── tokenizer/__init__.py          # BPE tokenizer wrapper + chat encoder
├── training/train.py              # Training loop (streaming, loss masking, val, checkpoint)
├── agent/
│   ├── dreamgtm_agent.py          # Agentic mode (THINK/SEARCH/TOOL tags, microcode injection)
│   ├── verifier.py                # Patch verifier (AST + tests + scanner)
│   └── scanner_runner.py          # Semgrep/Gitleaks/OSV/npm-audit/pip-audit
├── cli/dreamgtm.py                # CLI: chat, scan, patch, code, eval, microcode
├── configs/
│   ├── dreamgtm_80m.yaml          # 50M params
│   ├── dreamgtm_350m.yaml         # 205M params
│   └── dreamgtm_1b.yaml           # 1.1B params
├── eval/
│   ├── eval_harness.py            # 6 eval suites
│   ├── mbpp_heldout.jsonl         # 963 MBPP problems (held out)
│   └── security_patch_eval.jsonl  # 200 CVEs (held out)
├── research/security_microcode_v0/
│   ├── README.md                  # Microcode manifesto (S/K/F/V/O/P)
│   ├── microcode_dataset.jsonl    # 18 microcodes, 12 categories
│   ├── build_microcode.py         # Generator
│   └── retriever.py               # Keyword-based retriever
├── data/
│   ├── dreamgtm.tokenizer.json    # 32K BPE (2.2 MB)
│   ├── train_split.jsonl.gz       # 1.31M records (701 MB)
│   ├── val_split.jsonl            # 40.9K records (90 MB)
│   ├── security_balanced.jsonl.gz # 20.5K high-signal patching (16 MB)
│   ├── system_prompt.txt          # Stored once (not repeated per record)
│   └── verified/                  # Source data (kept for reference)
└── scripts/                       # Data collection + verification scripts
```

## Training Data (after verification)

| Source | Records | Type |
|--------|---------|------|
| CodeSearchNet (Python + multi-lang) | 778K | Code completion |
| OSV.dev vulnerabilities | 262K | Security knowledge |
| OpenHermes code | 79K | Code Q&A |
| github-code Python | 78K | Whole files |
| Instructions (mixed) | 98K | Instruction following |
| Magicoder OSS-Instruct | 42K | Synthetic code |
| Security balanced (curated) | 20K | 70% patching / 30% coding |
| Real CVE patches | 3.2K | Real fix diffs |
| OWASP docs | 1.6K | Security guidance |
| **Total train** | **1.31M** | |

## Usage

```bash
# Train (on GPU machine)
python training/train.py --config 80m --steps 10000 --batch-size 4

# Train (smoke test on CPU)
python training/train.py --config 80m --steps 10 --batch-size 1 --max-seq-len 128 --limit 50

# Chat (after training)
python -m cli.dreamgtm chat

# Scan repo
python -m cli.dreamgtm scan --repo ./my-website

# Eval
python -m eval.eval_harness --model models/dreamgtm_80m_final.pt --suite all

# List microcodes
python -m cli.dreamgtm microcode --list
```

## Security Microcode V0

Each microcode has 6 fields:

| Field | Name | Description |
|-------|------|-------------|
| **S** | Source | Where untrusted data enters |
| **K** | Sink | Where the dangerous operation happens |
| **F** | Flow | How data travels S→K |
| **V** | Vulnerability | What can go wrong |
| **O** | Patch Operator | The defensive transformation |
| **P** | Proof Obligation | What must hold for the fix to be correct |

**12 categories:** SQL injection, XSS, path traversal, command injection, insecure upload, SSRF, unsafe eval, weak password hashing, insecure CORS, hardcoded secret, unsafe JWT decode, missing authorization.

## Pipeline

```
Raw code → Security Microcode → DreamGTM → Patch Operator → Verifier → Developer Docs
```

Crafted with love by IBR (Ibraheem)

## Download Data (Release v1.0)

The large data files are in [Release v1.0](https://github.com/mrcocoxiox/dreamgtm/releases/tag/v1.0):

```bash
# Download all data files
mkdir -p data
cd data
wget https://github.com/mrcocoxiox/dreamgtm/releases/download/v1.0/dreamgtm.tokenizer.json
wget https://github.com/mrcocoxiox/dreamgtm/releases/download/v1.0/system_prompt.txt
wget https://github.com/mrcocoxiox/dreamgtm/releases/download/v1.0/security_balanced.jsonl.gz
wget https://github.com/mrcocoxiox/dreamgtm/releases/download/v1.0/val_split.jsonl
wget https://github.com/mrcocoxiox/dreamgtm/releases/download/v1.0/train_split.jsonl.gz
```

| File | Size | Description |
|------|------|-------------|
| dreamgtm.tokenizer.json | 2.2 MB | 32K BPE tokenizer |
| system_prompt.txt | 765 bytes | System prompt (stored once) |
| security_balanced.jsonl.gz | 16 MB | 20K high-signal patching records |
| val_split.jsonl | 94 MB | 40K validation records |
| train_split.jsonl.gz | 734 MB | 1.31M training records |
| **Total** | **846 MB** | |

## Colab Notebook

[DreamGTM_Training_Colab.ipynb](https://github.com/mrcocoxiox/dreamgtm/releases/download/v1.0/DreamGTM_Training_Colab.ipynb) — Open in Google Colab with GPU runtime.

Crafted with love by IBR (Ibraheem)
