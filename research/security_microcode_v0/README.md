# Security Microcode V0

## What is Security Microcode?

Inspired by CPU microcode — a thin layer of low-level primitives that the higher-level instruction set is built on — **Security Microcode** is a library of the smallest possible defensive code patterns that, when composed, defend against an entire class of vulnerability.

## Structure

Each microcode record has 6 fields:

| Field | Name | Description |
|-------|------|-------------|
| **S** | Source | Where untrusted data enters the system |
| **K** | Sink | Where the dangerous operation happens |
| **F** | Flow | How data travels from S to K |
| **V** | Vulnerability | What can go wrong |
| **O** | Patch Operator | The defensive transformation |
| **P** | Proof Obligation | What must hold for the fix to be correct |

## Pipeline

```
Raw code → Security Microcode → Small Model → Patch Operator → Verifier → Developer Documentation
```

1. **Raw code**: User submits vulnerable code
2. **Security Microcode**: Retrieve the relevant microcode(s) based on detected vulnerability class
3. **Small Model**: DreamGTM generates a patch using the microcode as context
4. **Patch Operator**: Apply the defensive transformation
5. **Verifier**: AST check + test run + scanner rerun
6. **Developer Documentation**: Auto-generate docs explaining the fix

## Categories (V0)

1. SQL Injection (CWE-89)
2. XSS (CWE-79)
3. Path Traversal (CWE-22)
4. Command Injection (CWE-78)
5. Insecure Upload (CWE-434)
6. SSRF (CWE-918)
7. Unsafe Eval (CWE-95)
8. Weak Password Hashing (CWE-327)
9. Insecure CORS (CWE-942)
10. Hardcoded Secret (CWE-798)
11. Unsafe JWT Decode (CWE-345)
12. Missing Authorization (CWE-862)

## Usage

```python
from research.security_microcode_v0.retriever import MicrocodeRetriever

retriever = MicrocodeRetriever()
results = retriever.retrieve("patch this SQL injection: cursor.execute(f'SELECT * FROM users WHERE id={user_id}')")
# Returns top-3 microcodes relevant to SQL injection
for mc in results:
    print(mc["id"], mc["category"])
    print("  Patch Operator:", mc["O"])
    print("  Microcode:", mc["microcode"])
```

## Integration with DreamGTM

At inference time, the DreamGTM agent:
1. Detects the vulnerability class from the user's prompt
2. Retrieves the top-3 relevant microcodes
3. Injects them as `<MICROCODE>` blocks in the prompt
4. The model generates a patch that applies the microcode's defensive pattern

## Why Separate from Training Data?

- **Tiny**: 18 microcodes (V0), ~19 KB. Mixing into 1.3M training records would dilute it to 0.001%.
- **Retrieval-augmented**: The model doesn't memorize microcodes; it retrieves them at inference.
- **Evolves independently**: We can iterate on microcode without retraining the model.
- **The "hidden trick" layer**: This is where CS-formula compression tricks live — constant-time comparison, BLAKE2 vs MD5, Argon2id memory parameters, context-aware encoding, etc.

## V0 → V1 Roadmap

- V0: 12 categories, 18 microcodes (current)
- V1: 25 categories (OWASP Top 10 + CWE Top 25), ~80 microcodes
- V2: 50 categories, ~200 microcodes, with embedding-based retrieval
- V3: Automated microcode generation from CVE patches

Crafted with love by IBR.
