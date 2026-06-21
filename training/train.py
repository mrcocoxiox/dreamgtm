"""
DreamGTM Training Script (v2 — Fixed)
======================================
Fixes:
1. BPE tokenizer (no more char-level)
2. Assistant-only loss masking (-100 for system+user)
3. Attention mask for padding
4. max_seq_len from config (not hardcoded)
5. Train/val split with validation loss
6. Checkpoint saving
7. Gradient checkpointing when config enables it
8. System prompt stripped from training data (stored once in data/system_prompt.txt)
9. Streaming dataset (doesn't load all records into RAM)

Usage:
  python training/train.py --config 80m --steps 100 --batch-size 2 --max-seq-len 512
  python training/train.py --config 80m --steps 10000 --batch-size 4 --val-every 500
"""
import os, sys, json, math, time, argparse, gzip
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.architecture import DreamGTM, DreamGTMConfig, get_config_80m, get_config_350m, get_config_1b, SPECIAL_TOKENS

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / 'data'
MODELS_DIR = BASE / 'models'
MODELS_DIR.mkdir(exist_ok=True)


class DreamGTMStreamDataset(IterableDataset):
    """Streaming dataset — reads JSONL.gz line by line, doesn't load all into RAM."""
    def __init__(self, data_path, tokenizer, max_seq_len=2048, limit=None):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.limit = limit

    def __iter__(self):
        if self.data_path.suffix == '.gz':
            f = gzip.open(self.data_path, 'rt', encoding='utf-8')
        else:
            f = self.data_path.open('r', encoding='utf-8')
        count = 0
        try:
            for line in f:
                if self.limit and count >= self.limit:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    result = self._encode_record(rec)
                    if result is not None:
                        yield result
                        count += 1
                except (json.JSONDecodeError, Exception):
                    continue
        finally:
            f.close()

    def _encode_record(self, rec):
        """Encode a chat record into (input_ids, attention_mask, labels)."""
        messages = rec.get('messages', [])
        if len(messages) < 2:
            return None

        # Build token sequence with special tokens
        # Format: <USER>{user}</USER><ASSISTANT>{assistant}<EOS>
        # Labels: -100 for user, actual IDs for assistant + <EOS>
        tokens = []
        labels = []

        for msg in messages:
            role = msg.get('role', '')
            content = msg.get('content', '')
            if not content:
                continue

            if role == 'user':
                user_tokens = [SPECIAL_TOKENS['<USER>']] + self.tokenizer.encode(content) + [SPECIAL_TOKENS['</USER>']]
                tokens.extend(user_tokens)
                labels.extend([-100] * len(user_tokens))
            elif role == 'assistant':
                asst_tokens = [SPECIAL_TOKENS['<ASSISTANT>']] + self.tokenizer.encode(content) + [SPECIAL_TOKENS['<EOS>']]
                tokens.extend(asst_tokens)
                # Labels: -100 for <ASSISTANT> tag, actual IDs for content + <EOS>
                labels.extend([-100])  # <ASSISTANT> tag
                labels.extend(asst_tokens[1:])  # content + <EOS>

        if len(tokens) < 10:
            return None

        # Truncate from left if too long (keep the assistant response)
        if len(tokens) > self.max_seq_len:
            tokens = tokens[-self.max_seq_len:]
            labels = labels[-self.max_seq_len:]

        # Pad to max_seq_len
        attention_mask = [1] * len(tokens)
        pad_len = self.max_seq_len - len(tokens)
        if pad_len > 0:
            tokens.extend([SPECIAL_TOKENS['<PAD>']] * pad_len)
            attention_mask.extend([0] * pad_len)
            labels.extend([-100] * pad_len)

        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
        }


def collate_fn(batch):
    """Collate a batch of dicts into batched tensors."""
    return {
        'input_ids': torch.stack([b['input_ids'] for b in batch]),
        'attention_mask': torch.stack([b['attention_mask'] for b in batch]),
        'labels': torch.stack([b['labels'] for b in batch]),
    }


def train(data_path=None, n_steps=100, batch_size=2, lr=3e-4,
          config_name='80m', max_seq_len=None, val_path=None,
          val_every=500, checkpoint_every=1000, limit=None):
    print("=" * 70)
    print("DreamGTM - Training v2 (Fixed)")
    print("Crafted with love by IBR (Ibraheem)")
    print("=" * 70)

    # Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Device: GPU ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device('cpu')
        print("Device: CPU (smoke test mode)")

    # Config
    configs = {'80m': get_config_80m, '350m': get_config_350m, '1b': get_config_1b}
    config = configs.get(config_name, get_config_80m)()

    # Override max_seq_len if specified
    if max_seq_len is not None:
        config.max_seq_len = max_seq_len
    print(f"Config: {config_name}, vocab={config.vocab_size}, hidden={config.hidden_size}, "
          f"layers={config.n_layers}, max_seq={config.max_seq_len}")

    # Tokenizer
    print(f"\n[1/5] Loading tokenizer...")
    from tokenizer import DreamGTMTokenizer
    tok_path = DATA_DIR / 'dreamgtm.tokenizer.json'
    if not tok_path.exists():
        print(f"ERROR: Tokenizer not found at {tok_path}")
        return
    tokenizer = DreamGTMTokenizer(tok_path)
    print(f"  Tokenizer loaded: vocab={tokenizer.vocab_size}")

    # Data
    if data_path is None:
        data_path = str(DATA_DIR / 'train_split.jsonl.gz')
    print(f"\n[2/5] Building dataset from {data_path}...")
    dataset = DreamGTMStreamDataset(data_path, tokenizer, config.max_seq_len, limit=limit)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=0)

    # Val dataset
    val_dataloader = None
    if val_path and Path(val_path).exists():
        val_dataset = DreamGTMStreamDataset(val_path, tokenizer, config.max_seq_len, limit=500)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=0)
        print(f"  Val dataset: {val_path}")

    # Model
    print(f"\n[3/5] Building DreamGTM-{config_name.upper()}...")
    model = DreamGTM(config).to(device)
    params = model.count_parameters()
    print(f"  Parameters: {params['total']:,}")
    print(f"  FP16 size: {params['total_mb_fp16']:.0f} MB")

    # Mixed precision
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    if use_amp:
        print("  Mixed precision: FP16")

    # Optimizer
    print(f"\n[4/5] Training setup...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    warmup = min(500, n_steps // 10)

    def lr_at(s):
        if s < warmup:
            return lr * s / max(warmup, 1)
        return lr * 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(n_steps - warmup, 1)))

    # Training
    print(f"\n[5/5] Training {n_steps} steps...")
    print("-" * 70)

    model.train()
    start = time.time()
    losses = []
    step = 0
    it = iter(dataloader)

    while step < n_steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dataloader)
            try:
                batch = next(it)
            except StopIteration:
                print("  Dataset exhausted, stopping.")
                break

        try:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            if use_amp:
                with torch.amp.autocast('cuda'):
                    loss, logits = model(input_ids, targets=labels, attention_mask=attention_mask)
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr_at(step)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss, logits = model(input_ids, targets=labels, attention_mask=attention_mask)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr_at(step)
                optimizer.step()

            losses.append(loss.item())
            step += 1

            if step % 10 == 0:
                avg = sum(losses[-10:]) / min(len(losses), 10)
                el = time.time() - start
                eta = el * (n_steps - step) / max(step, 1)
                print(f"  Step {step:5d}/{n_steps} | Loss: {avg:.4f} | LR: {lr_at(step):.2e} | ETA: {eta/60:.1f}min", flush=True)

            # Validation
            if val_dataloader and step % val_every == 0:
                model.eval()
                val_losses = []
                val_it = iter(val_dataloader)
                for _ in range(min(20, len(val_dataloader) if hasattr(val_dataloader, '__len__') else 20)):
                    try:
                        vbatch = next(val_it)
                        with torch.no_grad():
                            vloss, _ = model(
                                vbatch['input_ids'].to(device),
                                targets=vbatch['labels'].to(device),
                                attention_mask=vbatch['attention_mask'].to(device),
                            )
                        val_losses.append(vloss.item())
                    except StopIteration:
                        break
                if val_losses:
                    val_avg = sum(val_losses) / len(val_losses)
                    print(f"  → Val loss at step {step}: {val_avg:.4f}", flush=True)
                model.train()

            # Checkpoint
            if step % checkpoint_every == 0:
                ckpt_path = MODELS_DIR / f'dreamgtm_{config_name}_step{step}.pt'
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': {
                        'vocab_size': config.vocab_size,
                        'hidden_size': config.hidden_size,
                        'intermediate_size': config.intermediate_size,
                        'n_layers': config.n_layers,
                        'n_heads': config.n_heads,
                        'n_kv_heads': config.n_kv_heads,
                        'max_seq_len': config.max_seq_len,
                    },
                    'loss': losses[-1],
                }, str(ckpt_path))
                print(f"  → Checkpoint saved: {ckpt_path.name}", flush=True)

        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                torch.cuda.empty_cache() if use_amp else None
                print(f"  Step {step}: OOM, skipping batch", flush=True)
                continue
            print(f"  Step {step}: {str(e)[:80]}", flush=True)
            continue

    total = time.time() - start
    print(f"\nDone in {total:.1f}s ({total/60:.1f} min)")
    if losses:
        print(f"Final loss: {losses[-1]:.4f} | Min: {min(losses):.4f} | Avg: {sum(losses)/len(losses):.4f}")

    # Final checkpoint
    final_path = MODELS_DIR / f'dreamgtm_{config_name}_final.pt'
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': {
            'vocab_size': config.vocab_size,
            'hidden_size': config.hidden_size,
            'intermediate_size': config.intermediate_size,
            'n_layers': config.n_layers,
            'n_heads': config.n_heads,
            'n_kv_heads': config.n_kv_heads,
            'max_seq_len': config.max_seq_len,
        },
        'loss': losses[-1] if losses else 0,
    }, str(final_path))
    print(f"Final model saved: {final_path.name}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DreamGTM Training')
    parser.add_argument('--config', type=str, default='80m', choices=['80m', '350m', '1b'])
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--max-seq-len', type=int, default=None, help='Override config max_seq_len')
    parser.add_argument('--data-path', type=str, default=None)
    parser.add_argument('--val-path', type=str, default=None)
    parser.add_argument('--val-every', type=int, default=500)
    parser.add_argument('--checkpoint-every', type=int, default=1000)
    parser.add_argument('--limit', type=int, default=None, help='Limit number of records (for smoke test)')
    args = parser.parse_args()

    val_path = args.val_path or str(DATA_DIR / 'val_split.jsonl')

    train(
        data_path=args.data_path,
        n_steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        config_name=args.config,
        max_seq_len=args.max_seq_len,
        val_path=val_path,
        val_every=args.val_every,
        checkpoint_every=args.checkpoint_every,
        limit=args.limit,
    )
