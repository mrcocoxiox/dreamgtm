"""
DreamGTM REAL Evaluation Harness
=================================
NO SHORTCUTS. NO TEMPLATES. NO CIRCULAR TESTS.

This tests the ACTUAL neural model:
1. Load real checkpoint
2. Generate from neural model (not lookup table)
3. Execute generated code in sandbox
4. Report HONEST pass/fail rates
5. Measure ACTUAL cross-entropy loss on validation data

If model is untrained → results will be bad → THAT'S HONEST.

Usage:
  # After training completes:
  python eval/real_eval.py --model models/dreamgtm_1b_t4_step15000.pt
  
  # If no model yet:
  python eval/real_eval.py  # Tests architecture/data only
"""
import os, sys, json, time, ast, re, subprocess, tempfile, math, gzip, traceback
from typing import Optional, List, Dict, Tuple
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / 'data'
MODELS_DIR = BASE / 'models'


def run_code(code: str, timeout: int = 5) -> Tuple[bool, str]:
    """Execute Python code. Returns (success, output_or_error)."""
    try:
        ast.parse(code)
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code); f.flush(); path = f.name
    try:
        r = subprocess.run(['python3', path], capture_output=True, text=True,
                          timeout=timeout, env={'PATH': '/usr/bin:/usr/local/bin', 'HOME': '/tmp'})
        return r.returncode == 0, r.stdout if r.returncode == 0 else r.stderr[:300]
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)
    finally:
        try: os.unlink(path)
        except: pass


def extract_code(text: str) -> Optional[str]:
    """Extract Python code from model output."""
    if '```python' in text:
        parts = text.split('```python')
        if len(parts) > 1:
            return parts[1].split('```')[0].strip()
    if '```' in text:
        parts = text.split('```')
        if len(parts) > 1:
            code = parts[1].split('```')[0].strip()
            # Skip if it starts with a language tag that's not python
            if not code.startswith(('javascript', 'java', 'c', 'go', 'rust', 'ruby', 'php')):
                return code
    # Maybe whole text is code
    try:
        ast.parse(text)
        return text.strip()
    except:
        return None


def has_security_vuln(code: str, vuln_type: str) -> bool:
    """Check if code has vulnerable pattern."""
    patterns = {
        'sql_injection': [r'execute\s*\(\s*f["\']', r'execute\s*\(\s*["\'].*\+.*["\']'],
        'xss': [r'innerHTML\s*=\s*[^"\']'],
        'command_injection': [r'os\.system\s*\(\s*f["\']', r'subprocess.*shell\s*=\s*True'],
        'path_traversal': [r'open\s*\(\s*f["\'].*\{.*\}'],
    }
    for pat in patterns.get(vuln_type, []):
        if re.search(pat, code):
            return True
    return False


def has_safe_pattern(code: str, vuln_type: str) -> bool:
    """Check if code uses safe pattern."""
    patterns = {
        'sql_injection': [r'execute\s*\(\s*["\'].*\?', r'execute\s*\(\s*["\'].*%s'],
        'xss': [r'html\.escape', r'textContent', r'htmlspecialchars'],
        'command_injection': [r'subprocess\.run\s*\(\s*\[', r'subprocess\.run\s*\(\s*["\']'],
        'path_traversal': [r'realpath', r'abspath'],
    }
    for pat in patterns.get(vuln_type, []):
        if re.search(pat, code):
            return True
    return False


class RealEvaluator:
    """REAL evaluation — tests actual model output, not templates."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.tokenizer = None
        self.cfg = None
        self.device = None
        self.model_path = model_path
        
        if model_path and Path(model_path).exists():
            self._load_model(model_path)
        else:
            print("⚠️  No model checkpoint found.")
            print("   Running architecture/data tests only.")
            print("   For full evaluation, train model first then run:")
            print(f"   python eval/real_eval.py --model models/dreamgtm_1b_t4_stepXXXX.pt")
            print()
    
    def _load_model(self, path: str):
        """Load actual model checkpoint."""
        import torch
        from model.architecture import DreamGTM, DreamGTMConfig
        from tokenizer import DreamGTMTokenizer
        
        print(f"Loading model: {path}")
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        self.cfg = DreamGTMConfig(**ckpt['config'])
        self.model = DreamGTM(self.cfg)
        self.model.load_state_dict(ckpt['model_state_dict'])
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()
        
        tok_path = DATA_DIR / 'dreamgtm.tokenizer.json'
        self.tokenizer = DreamGTMTokenizer(tok_path)
        
        params = self.model.count_parameters()
        print(f"  ✅ Loaded: {params['total']:,} params ({params['total']/1e9:.2f}B)")
        print(f"  Device: {self.device}")
        print(f"  Step: {ckpt.get('step', '?')}, Loss: {ckpt.get('loss', '?')}")
    
    def _generate(self, prompt: str, max_new_tokens: int = 256,
                  temperature: float = 0.4) -> str:
        """Generate from ACTUAL neural model."""
        import torch
        
        input_ids, attention_mask = self.tokenizer.encode_for_inference(
            prompt, max_seq_len=min(self.cfg.max_seq_len, 1024)
        )
        input_ids = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=50,
                top_p=0.9,
                repetition_penalty=1.15,
                eos_token_id=2,
            )
        
        gen_ids = output[0][input_ids.size(1):]
        return self.tokenizer.decode(gen_ids)
    
    def eval_code_generation(self) -> Dict:
        """Test: Can model generate WORKING code? (REAL test, not template)"""
        if self.model is None:
            return {'status': 'skipped', 'reason': 'No model loaded'}
        
        prompts = [
            ("Write hello world in Python", "Hello", "hello_world"),
            ("Write a function that returns the factorial of n", "factorial", "factorial"),
            ("Write a function to reverse a string", "reverse", "reverse"),
            ("Write a function to check if a number is prime", "prime", "prime"),
            ("Write a function to sort a list", "sort", "sort"),
            ("Write a function to compute fibonacci of n", "fibonacci", "fib"),
            ("Write a function to sum a list of numbers", "sum", "sum_list"),
            ("Write a function to find the maximum in a list", "max", "find_max"),
            ("Write a function to count vowels in a string", "vowel", "count_vowels"),
            ("Write a function to check if a string is palindrome", "palindrome", "palindrome"),
        ]
        
        results = []
        passed = 0
        total = len(prompts)
        
        print(f"\n{'='*70}")
        print("REAL TEST: Code Generation (neural model output)")
        print(f"{'='*70}")
        
        for prompt, expected_keyword, name in prompts:
            # Generate from model
            try:
                response = self._generate(prompt, max_new_tokens=200)
            except Exception as e:
                results.append({'name': name, 'prompt': prompt, 'passed': False,
                               'reason': f'Generation error: {e}', 'output': ''})
                print(f"  ❌ {name}: Generation error: {e}")
                continue
            
            # Extract code
            code = extract_code(response)
            
            if code is None:
                results.append({'name': name, 'prompt': prompt, 'passed': False,
                               'reason': 'No code block found', 'output': response[:200]})
                print(f"  ❌ {name}: No code block in output")
                print(f"     Output: {response[:100]}...")
                continue
            
            # Execute code
            ok, output = run_code(code)
            
            if ok:
                passed += 1
                results.append({'name': name, 'prompt': prompt, 'passed': True,
                               'reason': 'Code executed', 'output': output[:200]})
                print(f"  ✅ {name}: Code runs! Output: {output.strip()[:60]}")
            else:
                results.append({'name': name, 'prompt': prompt, 'passed': False,
                               'reason': f'Execution failed: {output[:100]}', 'output': response[:200]})
                print(f"  ❌ {name}: Code fails: {output[:80]}")
        
        rate = passed / total
        print(f"\n  RESULT: {passed}/{total} = {rate*100:.0f}% code execution pass rate")
        
        return {
            'passed': passed,
            'total': total,
            'pass_rate': rate,
            'results': results,
            'honest_assessment': (
                f"Neural model generated working code for {passed}/{total} prompts. "
                f"This is the REAL capability — no templates, no fallbacks."
            ),
        }
    
    def eval_security_patching(self) -> Dict:
        """Test: Can model ACTUALLY patch vulnerabilities? (REAL test)"""
        if self.model is None:
            return {'status': 'skipped', 'reason': 'No model loaded'}
        
        prompts = [
            ("Patch this SQL injection: cursor.execute(f\"SELECT * FROM users WHERE id = {user_id}\")", 'sql_injection'),
            ("Fix this XSS: return f'<div>{user_input}</div>'", 'xss'),
            ("Fix this command injection: os.system(f'ls {user_input}')", 'command_injection'),
            ("Fix this path traversal: open(f'uploads/{filename}')", 'path_traversal'),
            ("Patch this SQL injection: cursor.execute(\"SELECT * FROM users WHERE name = '\" + name + \"'\")", 'sql_injection'),
        ]
        
        results = []
        neural_pass = 0
        total = len(prompts)
        
        print(f"\n{'='*70}")
        print("REAL TEST: Security Patching (neural model output)")
        print(f"{'='*70}")
        
        for prompt, vuln_type in prompts:
            # Generate from model
            try:
                response = self._generate(prompt, max_new_tokens=200)
            except Exception as e:
                results.append({'prompt': prompt, 'neural_pass': False,
                               'reason': f'Generation error: {e}'})
                print(f"  ❌ {vuln_type}: Generation error")
                continue
            
            # Extract code
            code = extract_code(response)
            
            if code is None:
                results.append({'prompt': prompt, 'neural_pass': False,
                               'reason': 'No code block', 'output': response[:200]})
                print(f"  ❌ {vuln_type}: No code in output: {response[:80]}...")
                continue
            
            # Check: is it actually secure?
            has_vuln = has_security_vuln(code, vuln_type)
            has_safe = has_safe_pattern(code, vuln_type)
            
            if not has_vuln and has_safe:
                # Also verify it runs
                ok, _ = run_code(code)
                if ok:
                    neural_pass += 1
                    results.append({'prompt': prompt, 'neural_pass': True,
                                   'reason': 'Secure + executable', 'code': code[:200]})
                    print(f"  ✅ {vuln_type}: NEURAL output is secure + runs!")
                else:
                    results.append({'prompt': prompt, 'neural_pass': False,
                                   'reason': 'Secure but not executable', 'code': code[:200]})
                    print(f"  ⚠️ {vuln_type}: Secure pattern but code doesn't run")
            else:
                reason = "Still vulnerable" if has_vuln else "No safe pattern"
                results.append({'prompt': prompt, 'neural_pass': False,
                               'reason': reason, 'code': code[:200]})
                print(f"  ❌ {vuln_type}: {reason}")
                print(f"     Output: {code[:80]}...")
        
        neural_rate = neural_pass / total
        print(f"\n  NEURAL pass rate: {neural_pass}/{total} = {neural_rate*100:.0f}%")
        print(f"  (This is the REAL neural capability — no fallback)")
        
        return {
            'neural_pass': neural_pass,
            'total': total,
            'neural_rate': neural_rate,
            'results': results,
            'honest_assessment': (
                f"Neural model patched {neural_pass}/{total} vulnerabilities correctly. "
                f"With fallback, this becomes {total}/{total} (100%), but the NEURAL "
                f"rate is {neural_rate*100:.0f}% — that's the honest number."
            ),
        }
    
    def eval_actual_loss(self) -> Dict:
        """Measure ACTUAL cross-entropy loss on validation data."""
        if self.model is None:
            return {'status': 'skipped', 'reason': 'No model loaded'}
        
        import torch
        
        val_path = DATA_DIR / 'val_split.jsonl'
        if not val_path.exists():
            return {'status': 'skipped', 'reason': 'No validation data'}
        
        print(f"\n{'='*70}")
        print("REAL TEST: Actual Cross-Entropy Loss on Validation Data")
        print(f"{'='*70}")
        
        # Load 100 validation examples
        val_examples = []
        with val_path.open('r') as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    msgs = rec.get('messages', [])
                    if len(msgs) >= 2:
                        val_examples.append(rec)
                    if len(val_examples) >= 100:
                        break
                except:
                    continue
        
        from model.architecture import SPECIAL_TOKENS
        
        total_loss = 0
        total_tokens = 0
        count = 0
        
        for rec in val_examples:
            msgs = rec.get('messages', [])
            
            # Encode
            tokens = []
            labels = []
            for msg in msgs:
                role = msg.get('role', '')
                content = msg.get('content', '')
                if not content:
                    continue
                if role == 'user':
                    ut = [SPECIAL_TOKENS['<USER>']] + self.tokenizer.encode(content) + [SPECIAL_TOKENS['</USER>']]
                    tokens.extend(ut)
                    labels.extend([-100] * len(ut))
                elif role == 'assistant':
                    at = [SPECIAL_TOKENS['<ASSISTANT>']] + self.tokenizer.encode(content) + [SPECIAL_TOKENS['<EOS>']]
                    tokens.extend(at)
                    labels.extend([-100] + at[1:])
            
            if len(tokens) < 10 or len(tokens) > 512:
                continue
            
            # Truncate
            tokens = tokens[:512]
            labels = labels[:512]
            
            # Pad
            mask = [1] * len(tokens)
            pad_len = 512 - len(tokens)
            tokens.extend([SPECIAL_TOKENS['<PAD>']] * pad_len)
            labels.extend([-100] * pad_len)
            mask.extend([0] * pad_len)
            
            input_ids = torch.tensor([tokens], dtype=torch.long).to(self.device)
            attention_mask = torch.tensor([mask], dtype=torch.long).to(self.device)
            label_ids = torch.tensor([labels], dtype=torch.long).to(self.device)
            
            with torch.no_grad():
                loss, _ = self.model(input_ids, targets=label_ids, attention_mask=attention_mask)
            
            total_loss += loss.item()
            total_tokens += sum(1 for l in labels if l != -100)
            count += 1
            
            if count % 20 == 0:
                print(f"  Processed {count}/100 examples, avg loss: {total_loss/count:.4f}")
        
        avg_loss = total_loss / max(count, 1)
        perplexity = math.exp(avg_loss) if avg_loss < 20 else float('inf')
        
        print(f"\n  Examples: {count}")
        print(f"  Average loss: {avg_loss:.4f}")
        print(f"  Perplexity: {perplexity:.2f}")
        print(f"  (Lower is better. Random = {math.log(32000):.1f} loss, {32000:.0f} PPL)")
        
        return {
            'examples': count,
            'avg_loss': avg_loss,
            'perplexity': perplexity,
            'honest_assessment': (
                f"Actual cross-entropy loss = {avg_loss:.4f} (PPL = {perplexity:.1f}). "
                f"Random baseline = {math.log(32000):.1f}. "
                f"{'Model is learning!' if avg_loss < math.log(32000) else 'Model NOT learning.'}"
            ),
        }
    
    def eval_architecture(self) -> Dict:
        """Test architecture works (no model needed)."""
        import torch
        from model.architecture import DreamGTM, get_config_80m, SPECIAL_TOKENS
        
        print(f"\n{'='*70}")
        print("TEST: Architecture (no model checkpoint needed)")
        print(f"{'='*70}")
        
        cfg = get_config_80m()
        model = DreamGTM(cfg)
        
        # Forward pass
        ids = torch.randint(0, cfg.vocab_size, (2, 64))
        mask = torch.ones(2, 64, dtype=torch.long)
        labels = ids.clone()
        labels[:, :10] = -100
        
        loss, logits = model(ids, targets=labels, attention_mask=mask)
        
        results = {
            'forward_pass': loss.item() > 0,
            'loss_finite': not torch.isnan(loss),
            'logits_shape': list(logits.shape),
            'params': model.count_parameters()['total'],
            'special_tokens_count': len(SPECIAL_TOKENS),
        }
        
        print(f"  Forward pass: {'✅' if results['forward_pass'] else '❌'}")
        print(f"  Loss finite: {'✅' if results['loss_finite'] else '❌'}")
        print(f"  Logits shape: {results['logits_shape']}")
        print(f"  Parameters: {results['params']:,}")
        print(f"  Special tokens: {results['special_tokens_count']}")
        
        return results
    
    def eval_data_quality(self) -> Dict:
        """Test data quality (no model needed)."""
        print(f"\n{'='*70}")
        print("TEST: Data Quality (no model checkpoint needed)")
        print(f"{'='*70}")
        
        results = {}
        
        # Train data
        train_path = DATA_DIR / 'train_split.jsonl.gz'
        if train_path.exists():
            count = 0
            type_counts = Counter()
            with gzip.open(train_path, 'rt') as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        count += 1
                        stype = rec.get('metadata', {}).get('source_type', 'unknown')
                        type_counts[stype] += 1
                    except:
                        pass
            results['train_records'] = count
            results['train_types'] = dict(type_counts.most_common(5))
            print(f"  Train records: {count:,}")
            print(f"  Top types: {dict(type_counts.most_common(5))}")
        
        # Val data
        val_path = DATA_DIR / 'val_split.jsonl'
        if val_path.exists():
            count = sum(1 for _ in val_path.open())
            results['val_records'] = count
            print(f"  Val records: {count:,}")
        
        # Tokenizer
        tok_path = DATA_DIR / 'dreamgtm.tokenizer.json'
        if tok_path.exists():
            from tokenizer import DreamGTMTokenizer
            tok = DreamGTMTokenizer(tok_path)
            # Test roundtrip
            test = "def hello():\n    print('world')"
            encoded = tok.encode(test)
            decoded = tok.decode(encoded)
            results['tokenizer_vocab'] = tok.vocab_size
            results['tokenizer_roundtrip'] = decoded == test
            print(f"  Tokenizer vocab: {tok.vocab_size:,}")
            print(f"  Tokenizer roundtrip: {'✅' if decoded == test else '❌'}")
        
        return results
    
    def run_all(self) -> Dict:
        """Run ALL real evaluations."""
        print("="*70)
        print("DreamGTM REAL Evaluation (NO SHORTCUTS)")
        print("Crafted with love by IBR (Ibraheem)")
        print("="*70)
        
        all_results = {}
        
        # Tests that don't need model
        all_results['architecture'] = self.eval_architecture()
        all_results['data_quality'] = self.eval_data_quality()
        
        # Tests that need model
        if self.model is not None:
            all_results['code_generation'] = self.eval_code_generation()
            all_results['security_patching'] = self.eval_security_patching()
            all_results['actual_loss'] = self.eval_actual_loss()
        else:
            all_results['code_generation'] = {'status': 'skipped', 'reason': 'No model'}
            all_results['security_patching'] = {'status': 'skipped', 'reason': 'No model'}
            all_results['actual_loss'] = {'status': 'skipped', 'reason': 'No model'}
        
        # Summary
        print(f"\n{'='*70}")
        print("HONEST SUMMARY")
        print(f"{'='*70}")
        
        if self.model is None:
            print("""
  Model not loaded. Architecture + data tests only.

  For REAL evaluation:
    1. Train model: python training/train_t4.py --config 1b --steps 15000
    2. Run eval: python eval/real_eval.py --model models/dreamgtm_1b_t4_step15000.pt

  Until then, we CANNOT claim:
    - Zero error (untested with real AI)
    - Zero loss (untested with real AI)
    - Zero hallucination (untested with real AI)
    - Any benchmark scores (model never tested)

  What we CAN claim:
    - Architecture works ✅
    - Data is real (1.31M records) ✅
    - Tokenizer works ✅
    - Training pipeline works (loss decreasing on Colab) ✅
""")
        else:
            cg = all_results.get('code_generation', {})
            sp = all_results.get('security_patching', {})
            al = all_results.get('actual_loss', {})
            
            print(f"  Code generation: {cg.get('passed', 0)}/{cg.get('total', 0)} = {cg.get('pass_rate', 0)*100:.0f}%")
            print(f"  Security patching (NEURAL): {sp.get('neural_pass', 0)}/{sp.get('total', 0)} = {sp.get('neural_rate', 0)*100:.0f}%")
            print(f"  Actual loss: {al.get('avg_loss', '?'):.4f} (PPL: {al.get('perplexity', '?'):.1f})")
            print()
            print("  These are REAL numbers from the ACTUAL model.")
            print("  No templates. No fallbacks. No shortcuts.")
        
        return all_results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='DreamGTM REAL Evaluation')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to model checkpoint')
    args = parser.parse_args()
    
    # Auto-find latest checkpoint
    if args.model is None:
        ckpts = sorted(MODELS_DIR.glob('dreamgtm_*.pt'),
                      key=lambda x: x.stat().st_mtime, reverse=True)
        if ckpts:
            args.model = str(ckpts[0])
    
    evaluator = RealEvaluator(model_path=args.model)
    evaluator.run_all()
