"""
DreamGTM - General Transformational Model
==========================================
Unified AI system merging DreamCode (coding) + DreamPatch (security).

Capabilities:
1. Code generation (Python, JS, TS, SQL, etc.)
2. Security patching (SQL injection, XSS, CSRF, etc.)
3. Reasoning & thinking (step-by-step analysis)
4. Agentic mode (web search, tool use, multi-step)
5. Code verification (AST, tests, scanner rerun)

Architecture (Modern Transformer):
- RMSNorm (no mean subtraction, faster)
- RoPE (rotary positions, no learned params)
- GQA (Grouped Query Attention, smaller KV cache)
- SwiGLU (gated FFN, better than GELU)
- Tied embeddings (input = output)
- Gradient checkpointing (save VRAM)
- FlashAttention compatible

Model sizes:
- DreamGTM-80M: testing (42M params, 80MB)
- DreamGTM-350M: stronger (205M params, 391MB)
- DreamGTM-1B: main model (1.1B params, 2.1GB)

Special tokens:
  <PAD> <BOS> <EOS> <UNK>
  <USER> <ASSISTANT> <THINK> <CODE> <PATCH>
  <TEST> <SEARCH> <TOOL> <RESULT> <SECURITY_REPORT>
  <DREAM_STATE> <VERIFY> <END_THINK>

Crafted with love by IBR (Ibraheem)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict


@dataclass
class DreamGTMConfig:
    """DreamGTM model configuration"""
    # Vocabulary
    vocab_size: int = 16000
    
    # Model dimensions
    hidden_size: int = 512
    intermediate_size: int = 1408
    n_layers: int = 12
    n_heads: int = 8        # Query heads
    n_kv_heads: int = 2     # Key/Value heads (GQA)
    
    # Context
    max_seq_len: int = 4096
    
    # Regularization
    dropout: float = 0.0
    attention_dropout: float = 0.0
    
    # RoPE
    rope_theta: float = 10000.0
    
    # Initialization
    initializer_range: float = 0.02
    
    # Tied embeddings
    tie_embeddings: bool = True
    
    # Training
    label_smoothing: float = 0.1
    
    # Gradient checkpointing
    gradient_checkpointing: bool = False
    
    # Sliding window (for long context)
    sliding_window: int = 0  # 0 = disabled
    
    def param_count(self) -> int:
        emb = self.vocab_size * self.hidden_size
        head_dim = self.hidden_size // self.n_heads
        kv_dim = self.n_kv_heads * head_dim
        per_layer = (
            self.hidden_size * self.n_heads * head_dim +  # Q
            self.hidden_size * kv_dim +                    # K
            self.hidden_size * kv_dim +                    # V
            self.n_heads * head_dim * self.hidden_size +   # O
            self.hidden_size * self.intermediate_size * 3 + # SwiGLU (gate, up, down)
            self.hidden_size * 2                            # 2x RMSNorm
        )
        layers = per_layer * self.n_layers
        final_norm = self.hidden_size
        output = 0 if self.tie_embeddings else self.vocab_size * self.hidden_size
        return emb + layers + final_norm + output


# ============================================================
# SPECIAL TOKENS
# ============================================================
SPECIAL_TOKENS = {
    "<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<UNK>": 3,
    "<SYSTEM>": 4, "</SYSTEM>": 5,
    "<USER>": 6, "</USER>": 7,
    "<ASSISTANT>": 8, "</ASSISTANT>": 9,
    "<THINK>": 10, "</THINK>": 11,
    "<CODE>": 12, "</CODE>": 13,
    "<PATCH>": 14, "</PATCH>": 15,
    "<TEST>": 16, "</TEST>": 17,
    "<SEARCH>": 18, "</SEARCH>": 19,
    "<TOOL>": 20, "</TOOL>": 21,
    "<RESULT>": 22, "</RESULT>": 23,
    "<SECURITY_REPORT>": 24, "</SECURITY_REPORT>": 25,
    "<MICROCODE>": 26, "</MICROCODE>": 27,
    "<VERIFY>": 28, "</VERIFY>": 29,
    "<DREAM_STATE>": 30, "</DREAM_STATE>": 31,
}


# ============================================================
# 1. RMSNorm
# ============================================================
class RMSNorm(nn.Module):
    """Root Mean Square Normalization: y = x / sqrt(mean(x²) + ε) × γ"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight


# ============================================================
# 2. RoPE
# ============================================================
class RoPE(nn.Module):
    """Rotary Position Embeddings - no learnable params"""
    def __init__(self, dim: int, base: float = 10000.0, max_len: int = 8192):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self._build_cache(max_len)
    
    def _build_cache(self, max_len: int):
        positions = torch.arange(max_len).float()
        freqs = torch.outer(positions, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cached', emb.cos(), persistent=False)
        self.register_buffer('sin_cached', emb.sin(), persistent=False)
    
    def _rotate_half(self, x):
        x1, x2 = x[..., :x.size(-1) // 2], x[..., x.size(-1) // 2:]
        return torch.cat([-x2, x1], dim=-1)
    
    def forward(self, q, k, seq_len):
        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        return q * cos + self._rotate_half(q) * sin, k * cos + self._rotate_half(k) * sin


# ============================================================
# 3. GQA (Grouped Query Attention)
# ============================================================
class GQA(nn.Module):
    """Grouped Query Attention with optional sliding window"""
    def __init__(self, config: DreamGTMConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.hidden_size // config.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        
        self.wq = nn.Linear(config.hidden_size, config.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.hidden_size, config.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.hidden_size, config.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.hidden_size, bias=False)
        
        self.attn_dropout = config.attention_dropout
        self.rope = RoPE(self.head_dim, config.rope_theta, config.max_seq_len)
        self.sliding_window = config.sliding_window
    
    def forward(self, x, past_kv=None, use_cache=False, attention_mask=None):
        B, T, C = x.size()
        
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        
        # KV cache for inference
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        
        new_kv = (k, v) if use_cache else None
        
        # RoPE
        q, k = self.rope(q, k, q.size(2))
        
        # Repeat KV for GQA
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)
        
        # Causal mask + optional sliding window
        T_q = q.size(2)
        T_k = k.size(2)
        # Build causal mask as 4D float (B, 1, T_q, T_k) — works with FP16 autocast
        causal_2d = torch.tril(torch.ones(T_q, T_k, device=x.device, dtype=torch.bool))
        causal_4d = causal_2d.unsqueeze(0).unsqueeze(0).expand(B, 1, T_q, T_k)
        
        # Sliding window
        if self.sliding_window > 0:
            window_mask = torch.ones(T_q, T_k, device=x.device, dtype=torch.bool)
            for i in range(T_q):
                start = max(0, i - self.sliding_window + 1)
                window_mask[i, :start] = False
            causal_4d = causal_4d & window_mask.unsqueeze(0).unsqueeze(0)
        
        # Combine with padding mask if provided
        if attention_mask is not None:
            pad_mask = attention_mask[:, :T_k].bool().unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T_k)
            full_mask = causal_4d & pad_mask  # (B, 1, T_q, T_k)
        else:
            full_mask = causal_4d
        
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=full_mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        
        out = out.transpose(1, 2).contiguous().view(B, T_q, -1)
        return self.wo(out), new_kv


# ============================================================
# 4. SwiGLU
# ============================================================
class SwiGLU(nn.Module):
    """SwiGLU: FFN(x) = (SiLU(x·W_gate) × (x·W_up)) · W_down"""
    def __init__(self, hidden_size, intermediate_size, dropout=0.0):
        super().__init__()
        self.w_gate = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w_up = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w_down = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.dropout = dropout
    
    def forward(self, x):
        return self.w_down(F.dropout(F.silu(self.w_gate(x)) * self.w_up(x), p=self.dropout, training=self.training))


# ============================================================
# 5. Transformer Block
# ============================================================
class TransformerBlock(nn.Module):
    """Pre-norm block: x = x + attn(rmsnorm(x)); x = x + ffn(rmsnorm(x))"""
    def __init__(self, config: DreamGTMConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.hidden_size)
        self.attn = GQA(config)
        self.norm2 = RMSNorm(config.hidden_size)
        self.ffn = SwiGLU(config.hidden_size, config.intermediate_size, config.dropout)
    
    def forward(self, x, past_kv=None, use_cache=False, attention_mask=None):
        attn_out, new_kv = self.attn(self.norm1(x), past_kv, use_cache, attention_mask=attention_mask)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, new_kv


# ============================================================
# 6. DreamGTM Model
# ============================================================
class DreamGTM(nn.Module):
    """
    DreamGTM - General Transformational Model
    
    Unified coding + security + reasoning AI.
    
    Pipeline:
        Input → [Embedding] → [Transformer Blocks] → [RMSNorm] → [Output]
        
    Features:
    - GQA: 8 Q heads, 2 KV heads (4x smaller cache)
    - RoPE: No learned positions
    - RMSNorm: Faster than LayerNorm
    - SwiGLU: Better than GELU
    - Tied embeddings
    - KV cache for fast inference
    - Gradient checkpointing (save VRAM)
    - Sliding window attention (long context)
    """
    
    def __init__(self, config: DreamGTMConfig):
        super().__init__()
        self.config = config
        
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.emb_dropout = nn.Dropout(config.dropout)
        
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        self.norm_f = RMSNorm(config.hidden_size)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=self.config.initializer_range)
            elif isinstance(m, RMSNorm):
                nn.init.ones_(m.weight)
        
        scale = 1.0 / math.sqrt(2 * self.config.n_layers)
        for layer in self.layers:
            layer.attn.wo.weight.data.mul_(scale)
            layer.ffn.w_down.weight.data.mul_(scale)
    
    def forward(self, input_ids, targets=None, attention_mask=None, past_kvs=None, use_cache=False):
        B, T = input_ids.size()
        
        x = self.emb_dropout(self.token_embedding(input_ids))
        
        new_kvs = []
        for i, layer in enumerate(self.layers):
            past_kv = past_kvs[i] if past_kvs is not None else None
            
            if self.config.gradient_checkpointing and self.training:
                x, new_kv = torch.utils.checkpoint.checkpoint(
                    layer, x, past_kv, use_cache, attention_mask, use_reentrant=False
                )
            else:
                x, new_kv = layer(x, past_kv, use_cache, attention_mask=attention_mask)
            new_kvs.append(new_kv)
        
        x = self.norm_f(x)
        logits = x @ self.token_embedding.weight.T  # Tied
        
        if targets is not None:
            logits_shifted = logits[:, :-1, :].contiguous()
            targets_shifted = targets[:, 1:].contiguous()
            loss = F.cross_entropy(
                logits_shifted.view(-1, logits.size(-1)),
                targets_shifted.view(-1),
                ignore_index=-100,  # Standard PyTorch ignore_index for masking
                label_smoothing=self.config.label_smoothing,
            )
            return loss, logits
        
        return logits, new_kvs if use_cache else None
    
    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=512, temperature=0.2,
                 top_k=50, top_p=0.9, repetition_penalty=1.1, eos_token_id=2):
        """Generate with KV cache for speed"""
        self.eval()
        
        # Prefill
        logits, past_kvs = self.forward(input_ids, use_cache=True)
        
        for _ in range(max_new_tokens):
            next_logits = logits[:, -1, :] / max(temperature, 1e-6)
            
            if repetition_penalty > 1.0:
                recent = input_ids[0, -30:].tolist()
                for tid in set(recent):
                    if tid < next_logits.size(-1):
                        next_logits[0, tid] /= repetition_penalty
            
            if top_k > 0:
                values, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits = torch.where(next_logits < values[:, -1:],
                                          torch.full_like(next_logits, float('-inf')), next_logits)
            
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
                cum_probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                sorted_mask = cum_probs > top_p
                sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
                sorted_mask[..., 0] = False
                indices_to_remove = sorted_mask.scatter(1, sorted_idx, sorted_mask)
                next_logits = next_logits.masked_fill(indices_to_remove, float('-inf'))
            
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            if next_token.item() == eos_token_id:
                break
            
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Decode with KV cache
            logits, past_kvs = self.forward(next_token, past_kvs=past_kvs, use_cache=True)
        
        return input_ids
    
    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        return {
            'total': total,
            'total_mb_fp16': total * 2 / 1024**2,
            'total_mb_fp32': total * 4 / 1024**2,
        }


# ============================================================
# CONFIG PRESETS
# ============================================================
def get_config_80m() -> DreamGTMConfig:
    return DreamGTMConfig(
        vocab_size=32000, hidden_size=512, intermediate_size=1408,
        n_layers=12, n_heads=8, n_kv_heads=2, max_seq_len=2048,
    )

def get_config_350m() -> DreamGTMConfig:
    return DreamGTMConfig(
        vocab_size=32000, hidden_size=1024, intermediate_size=2816,
        n_layers=16, n_heads=8, n_kv_heads=2, max_seq_len=2048,
    )

def get_config_1b() -> DreamGTMConfig:
    return DreamGTMConfig(
        vocab_size=32000, hidden_size=2048, intermediate_size=5504,
        n_layers=24, n_heads=16, n_kv_heads=4, max_seq_len=2048,
        gradient_checkpointing=True,
    )


if __name__ == "__main__":
    for name, fn in [("80M", get_config_80m), ("350M", get_config_350m), ("1B", get_config_1b)]:
        cfg = fn()
        m = DreamGTM(cfg)
        p = m.count_parameters()
        x = torch.randint(0, cfg.vocab_size, (2, 64))
        loss, logits = m(x, targets=x)
        print(f"DreamGTM-{name}: {p['total']:,} params | FP16: {p['total_mb_fp16']:.0f}MB | loss={loss.item():.2f}")
        del m
