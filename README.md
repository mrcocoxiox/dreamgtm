# DreamGTM — Hardened AI for Coding + Security Patching

**Crafted with love by IBR (Ibraheem)**

A compact AI that **codes** and **patches security vulnerabilities**. Trained from scratch on T4 GPU using the **IBR Method** (7 CS tricks).

## ⚠️ HONEST STATUS (No False Claims)

### What's PROVEN to work:
| Component | Status | How Verified |
|-----------|--------|-------------|
| Architecture (GQA+RoPE+SwiGLU+RMSNorm) | ✅ REAL | Forward/backward pass tested |
| Tokenizer (32K BPE) | ✅ REAL | 100% roundtrip fidelity |
| Dataset (1.31M records) | ✅ REAL | Collected from 14 sources, AST-verified |
| Training pipeline (IBR Method) | ✅ REAL | Loss decreasing on Colab (10→6→...) |
| Security microcode (18 patterns) | ✅ REAL | Retriever tested, correct retrieval |
| T4 optimization (8-bit Adam, packing, checkpointing) | ✅ REAL | VRAM fits 16GB, checkpoints save |

### What's THEORETICAL (architecture designed, not yet proven with real AI):
| Component | Status | Why |
|-----------|--------|-----|
| Zero-Error Architecture | ⚠️ DESIGN ONLY | Code structure is sound, but tested with templates not AI |
| Zero-Loss Architecture | ⚠️ DESIGN ONLY | Lookup table works mathematically, not connected to model |
| Zero-Hallucination | ⚠️ DESIGN ONLY | Knowledge base approach is sound, not tested with AI |
| 5 Novel Methods | ⚠️ CODE ONLY | CMA tested for speed, others need real model |
| Benchmark suite | ⚠️ PARTIAL | Tests architecture/data, not model output |

### What's NOT YET DONE:
| Item | Why |
|------|-----|
| Real model evaluation | Model still training on Colab |
| Real code generation test | Need trained checkpoint |
| Real security patching test | Need trained checkpoint |
| Real loss measurement | Need trained checkpoint |
| Novel methods validation | Need trained model + more compute |

## 📁 Project Structure (27 files, clean)

```
dreamgtm/
├── model/architecture.py          # DreamGTM transformer
├── tokenizer/__init__.py          # 32K BPE tokenizer
├── training/train_t4.py           # IBR Method training (7 tricks)
├── inference/smart_inference.py   # Smart inference engine
├── agent/                         # Agent + verifier + scanner
├── cli/dreamgtm.py                # CLI interface
├── eval/
│   ├── real_eval.py               # REAL evaluation (no shortcuts)
│   ├── eval_harness.py            # Basic eval harness
│   └── comprehensive_benchmarks.py # Benchmark suite
├── research/
│   ├── zero_error_architecture.py # DESIGN: 0% error via verifier
│   ├── zero_loss_architecture.py  # DESIGN: 0.0 loss via lookup table
│   ├── zero_hallucination.py      # DESIGN: 0% hallucination via grounding
│   ├── novel_methods.py           # 5 novel methods (CMA tested)
│   └── security_microcode_v0/     # 18 microcodes
├── scripts/                       # Tokenizer, split, quantize
└── configs/                       # 80M, 350M, 1B YAML
```

## 🧠 The IBR Method (7 CS Tricks for T4 Training)

| # | Trick | Effect |
|---|-------|--------|
| 1 | Sequence packing | 4-5x more tokens/batch |
| 2 | 8-bit AdamW | Saves 6GB VRAM |
| 3 | FP16 mixed precision | 2x gradient savings |
| 4 | Gradient checkpointing | Activations 3GB→1GB |
| 5 | Curriculum learning | Disabled (T4 OOM fix) |
| 6 | Importance sampling | Security 5x oversampled |
| 7 | Microcode conditioning | Inject defensive patterns |

## 🏗️ Architecture

```
Input → [Embedding] → [Transformer × N (GQA + RoPE + SwiGLU + RMSNorm)] → [Output]
```

| Model | Params | FP16 | INT4 |
|-------|--------|------|------|
| 80M | 50M | 96 MB | 24 MB |
| 350M | 205M | 391 MB | 98 MB |
| 1B | 1.13B | 2.3 GB | 565 MB |

## 📊 Training Data (REAL, verified)

| Source | Records | Type |
|--------|---------|------|
| CodeSearchNet | 778K | Code |
| OSV.dev | 262K | Vulnerabilities |
| OpenHermes | 79K | Code Q&A |
| github-code | 78K | Files |
| Security balanced | 20K | Patching |
| CVE patches | 3.2K | Real fixes |
| **Total** | **1.31M** | |

## 🔧 Usage

### Train (T4 GPU)
```bash
python training/train_t4.py --config 1b --steps 15000 --batch-size 1 --max-seq-len 256
```

### REAL Evaluation (after training)
```bash
python eval/real_eval.py --model models/dreamgtm_1b_t4_step15000.pt
```

### Compress
```bash
python scripts/quantize_model.py --input models/dreamgtm_1b.pt --all
```

## 📦 GitHub

- **Repo:** https://github.com/mrcocoxiox/dreamgtm
- **Release v1.0:** Data + notebooks (846 MB)
- **Colab:** DreamGTM_FINAL.ipynb

## 🎯 Next Steps (HONEST)

1. **Complete training** (15K steps on T4, ~5 hours)
2. **Run REAL evaluation** (`python eval/real_eval.py --model ...`)
3. **Report ACTUAL numbers** (even if bad)
4. **Then validate** zero-error/loss/hallucination claims with real AI
5. **Test novel methods** with trained model

---

Crafted with love by IBR (Ibraheem) 🚀
