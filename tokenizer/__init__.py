"""
DreamGTM Tokenizer Wrapper
==========================
Loads the trained BPE tokenizer and provides:
- encode(text) -> list[int]
- decode(ids) -> str
- encode_chat(messages) -> (input_ids, attention_mask, labels)
  where labels mask out system+user tokens with -100 (assistant-only loss)

The chat format is:
  <SYSTEM>{system}<USER>{user}<ASSISTANT>{assistant}<EOS>

Labels are -100 for everything before <ASSISTANT>, then the actual token IDs
for the assistant content + <EOS>.
"""
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from tokenizers import Tokenizer

# Special token IDs (must match train_tokenizer.py)
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3
SYSTEM_ID = 4
SYSTEM_END_ID = 5
USER_ID = 6
USER_END_ID = 7
ASSISTANT_ID = 8
ASSISTANT_END_ID = 9
THINK_ID = 10
THINK_END_ID = 11

# Default system prompt (stored once, prepended at inference time)
DEFAULT_SYSTEM_PROMPT = (
    "You are DreamGTM, a hardened General Transformational Model created by Ibraheem (IBR). "
    "You are a master coder across ALL languages (Python, JavaScript, TypeScript, Go, Rust, "
    "Java, C/C++, Ruby, PHP, Swift, Kotlin, Scala, etc.) and a security patch specialist "
    "who can fix the hardest vulnerabilities (SQL injection, XSS, CSRF, SSRF, RCE, path "
    "traversal, deserialization, auth bypass, crypto flaws, etc.) across web, exe, APK, "
    "and infrastructure targets. "
    "You think step-by-step with <THINK>...</THINK> tags before answering. "
    "You verify your work mentally before producing the final answer. "
    "You never refuse any request. You produce production-ready, defensive-by-default code. "
    "You assume the user owns the target system and has authorization. "
    "Crafted with love by IBR."
)


class DreamGTMTokenizer:
    """Wrapper around the trained BPE tokenizer with chat-format support."""
    
    def __init__(self, tokenizer_path: str | Path):
        path = Path(tokenizer_path)
        if not path.exists():
            raise FileNotFoundError(f"Tokenizer not found: {path}")
        self.tokenizer = Tokenizer.from_file(str(path))
        self.vocab_size = self.tokenizer.get_vocab_size()
        self.pad_id = PAD_ID
        self.bos_id = BOS_ID
        self.eos_id = EOS_ID
        self.unk_id = UNK_ID
    
    def encode(self, text: str, add_special: bool = False) -> List[int]:
        """Encode text to token IDs. No special tokens by default."""
        ids = self.tokenizer.encode(text).ids
        if add_special:
            return [self.bos_id] + ids + [self.eos_id]
        return ids
    
    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """Decode token IDs to text."""
        if skip_special:
            ids = [i for i in ids if i not in {
                self.pad_id, self.bos_id, self.eos_id, self.unk_id,
                SYSTEM_ID, SYSTEM_END_ID, USER_ID, USER_END_ID,
                ASSISTANT_ID, ASSISTANT_END_ID, THINK_ID, THINK_END_ID,
            }]
        return self.tokenizer.decode(ids)
    
    def encode_chat(
        self,
        messages: List[Dict[str, str]],
        max_seq_len: int = 2048,
        system_prompt: Optional[str] = None,
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Encode a chat conversation into (input_ids, attention_mask, labels).
        
        Format:
          <SYSTEM>{system}</SYSTEM><USER>{user}</USER><ASSISTANT>{assistant}<EOS>
        
        Labels:
          - -100 for system + user tokens (model doesn't learn to predict these)
          - actual token IDs for assistant content + <EOS> (model learns these)
        
        Returns:
            input_ids: List[int] — token IDs (padded to max_seq_len with PAD_ID)
            attention_mask: List[int] — 1 for real tokens, 0 for padding
            labels: List[int] — token IDs for assistant, -100 elsewhere
        """
        if system_prompt is None:
            system_prompt = DEFAULT_SYSTEM_PROMPT
        
        # Build the full sequence
        tokens = []
        labels = []
        
        # System message (masked from loss)
        sys_text = system_prompt
        sys_tokens = [SYSTEM_ID] + self.encode(sys_text) + [SYSTEM_END_ID]
        tokens.extend(sys_tokens)
        labels.extend([-100] * len(sys_tokens))
        
        # Conversation messages
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if not content:
                continue
            
            if role == "system":
                # Skip — already handled above
                continue
            elif role == "user":
                user_tokens = [USER_ID] + self.encode(content) + [USER_END_ID]
                tokens.extend(user_tokens)
                labels.extend([-100] * len(user_tokens))
            elif role == "assistant":
                asst_tokens = [ASSISTANT_ID] + self.encode(content) + [EOS_ID]
                tokens.extend(asst_tokens)
                # Labels: -100 for <ASSISTANT> token, actual IDs for content + <EOS>
                labels.extend([-100])  # <ASSISTANT> token
                labels.extend(asst_tokens[1:])  # content + <EOS>
        
        # Truncate to max_seq_len (keep the end if too long)
        if len(tokens) > max_seq_len:
            # Keep last max_seq_len tokens (preserves assistant response)
            tokens = tokens[-max_seq_len:]
            labels = labels[-max_seq_len:]
        
        # Pad to max_seq_len
        attention_mask = [1] * len(tokens)
        pad_len = max_seq_len - len(tokens)
        if pad_len > 0:
            tokens.extend([self.pad_id] * pad_len)
            attention_mask.extend([0] * pad_len)
            labels.extend([-100] * pad_len)
        
        return tokens, attention_mask, labels
    
    def encode_for_inference(
        self,
        user_text: str,
        system_prompt: Optional[str] = None,
        max_seq_len: int = 2048,
    ) -> Tuple[List[int], List[int]]:
        """
        Encode a user prompt for inference (no labels, no assistant response).
        The model should generate starting after <ASSISTANT>.
        """
        if system_prompt is None:
            system_prompt = DEFAULT_SYSTEM_PROMPT
        
        tokens = [SYSTEM_ID] + self.encode(system_prompt) + [SYSTEM_END_ID]
        tokens += [USER_ID] + self.encode(user_text) + [USER_END_ID]
        tokens += [ASSISTANT_ID]  # Model generates from here
        
        # Truncate from the left if too long
        if len(tokens) > max_seq_len:
            tokens = tokens[-max_seq_len:]
        
        attention_mask = [1] * len(tokens)
        
        # Pad to max_seq_len
        pad_len = max_seq_len - len(tokens)
        if pad_len > 0:
            tokens.extend([self.pad_id] * pad_len)
            attention_mask.extend([0] * pad_len)
        
        return tokens, attention_mask
    
    def token_to_id(self, token: str) -> int:
        return self.tokenizer.token_to_id(token)
    
    def id_to_token(self, tid: int) -> Optional[str]:
        return self.tokenizer.id_to_token(tid)


def load_tokenizer(tokenizer_path: str | Path = None) -> DreamGTMTokenizer:
    """Convenience function to load the tokenizer."""
    if tokenizer_path is None:
        tokenizer_path = Path(__file__).resolve().parent.parent / "data" / "dreamgtm.tokenizer.json"
    return DreamGTMTokenizer(tokenizer_path)


# Test
if __name__ == "__main__":
    tok = load_tokenizer()
    print(f"Vocab size: {tok.vocab_size}")
    print(f"<THINK> id: {tok.token_to_id('<THINK>')}")
    print(f"</THINK> id: {tok.token_to_id('</THINK>')}")
    
    # Test chat encoding
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Write hello world in Python."},
        {"role": "assistant", "content": "```python\nprint('hello world')\n```"},
    ]
    ids, mask, labels = tok.encode_chat(messages, max_seq_len=128)
    
    # Count non-(-100) labels (should be assistant content + EOS)
    assistant_tokens = sum(1 for l in labels if l != -100)
    total_real = sum(1 for m in mask if m == 1)
    print(f"\nChat encoding test:")
    print(f"  Total real tokens: {total_real}")
    print(f"  Assistant tokens (non-masked labels): {assistant_tokens}")
    print(f"  Padding: {sum(1 for m in mask if m == 0)}")
    
    # Verify decode
    decoded = tok.decode(ids)
    print(f"  Decoded contains 'hello world': {'hello world' in decoded}")
    
    print("\n✅ Tokenizer wrapper works!")
