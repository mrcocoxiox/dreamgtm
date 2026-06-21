"""
DreamGTM CLI - Command-line interface
======================================
Usage:
  python -m cli.dreamgtm chat
  python -m cli.dreamgtm scan --repo ./my-website
  python -m cli.dreamgtm patch --repo ./my-website
  python -m cli.dreamgtm code "Write factorial"
  python -m cli.dreamgtm eval --model models/dreamgtm_80m_final.pt --suite all
  python -m cli.dreamgtm microcode --list
  python -m cli.dreamgtm microcode --search "sql injection"
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / 'data'
MODELS_DIR = BASE / 'models'


def load_model_and_tokenizer(model_path):
    """Load model + tokenizer for chat/patch/code commands."""
    import torch
    from model.architecture import DreamGTM, DreamGTMConfig
    from tokenizer import DreamGTMTokenizer

    if not Path(model_path).exists():
        print(f"Model not found: {model_path}")
        return None, None, None

    ckpt = torch.load(str(model_path), map_location='cpu', weights_only=False)
    cfg_dict = ckpt['config']
    cfg = DreamGTMConfig(**cfg_dict)
    model = DreamGTM(cfg)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    tok_path = DATA_DIR / 'dreamgtm.tokenizer.json'
    if not tok_path.exists():
        print(f"Tokenizer not found: {tok_path}")
        return None, None, None
    tokenizer = DreamGTMTokenizer(tok_path)

    return model, tokenizer, cfg


def cmd_chat(args):
    """Interactive chat with DreamGTM."""
    print("=" * 70)
    print("DreamGTM - Interactive Chat")
    print("Crafted with love by IBR (Ibraheem)")
    print("Never refuses. Assumes you own the target system.")
    print("=" * 70)
    print("Type 'quit' to exit\n")

    # Load agent if model available
    agent = None
    model_path = args.model or str(MODELS_DIR / 'dreamgtm_80m_final.pt')
    if Path(model_path).exists():
        try:
            model, tokenizer, cfg = load_model_and_tokenizer(model_path)
            from agent.dreamgtm_agent import DreamGTMAgent
            agent = DreamGTMAgent(model=model, tokenizer=tokenizer)
            print(f"✅ Model loaded: {model_path}\n")
        except Exception as e:
            print(f"⚠️ Model load failed: {e}")
            print("Running in passthrough mode.\n")
    else:
        print(f"⚠️ No model at {model_path}")
        print("Running in passthrough mode.\n")

    while True:
        try:
            user_input = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if user_input.lower() in ('quit', 'exit', 'q'):
            print("Bye!")
            break
        if not user_input:
            continue

        if agent:
            response = agent.process(user_input)
        else:
            response = f"[Passthrough mode — no model loaded]\n\nYou asked: {user_input}"

        print(f"\nDreamGTM> {response}\n")


def cmd_scan(args):
    """Scan a repo for vulnerabilities."""
    from agent.scanner_runner import ScannerRunner
    runner = ScannerRunner()
    repo = args.repo
    if not Path(repo).exists():
        print(f"Repo not found: {repo}")
        return
    print(f"Scanning {repo}...")
    results = runner.scan_all(repo)
    print(json.dumps(results, indent=2, default=str))


def cmd_patch(args):
    """Patch vulnerabilities in a repo."""
    model_path = args.model or str(MODELS_DIR / 'dreamgtm_80m_final.pt')
    if not Path(model_path).exists():
        print(f"Model not found: {model_path}")
        return

    model, tokenizer, cfg = load_model_and_tokenizer(model_path)
    if model is None:
        return

    from agent.dreamgtm_agent import DreamGTMAgent
    agent = DreamGTMAgent(model=model, tokenizer=tokenizer)

    # Read the file to patch
    file_path = args.file
    if not Path(file_path).exists():
        print(f"File not found: {file_path}")
        return

    code = Path(file_path).read_text(encoding='utf-8', errors='replace')
    prompt = f"Patch this code for security vulnerabilities:\n\n```\n{code}\n```"
    response = agent.process(prompt)
    print(response)


def cmd_code(args):
    """Generate code from a prompt."""
    model_path = args.model or str(MODELS_DIR / 'dreamgtm_80m_final.pt')
    if not Path(model_path).exists():
        print(f"Model not found: {model_path}")
        return

    model, tokenizer, cfg = load_model_and_tokenizer(model_path)
    if model is None:
        return

    from agent.dreamgtm_agent import DreamGTMAgent
    agent = DreamGTMAgent(model=model, tokenizer=tokenizer)
    response = agent.process(args.prompt)
    print(response)


def cmd_eval(args):
    """Run the eval harness."""
    from eval.eval_harness import main as eval_main
    sys.argv = ['eval_harness', '--model', args.model, '--suite', args.suite, '--n', str(args.n)]
    eval_main()


def cmd_microcode(args):
    """List or search security microcodes."""
    from research.security_microcode_v0.retriever import MicrocodeRetriever
    retriever = MicrocodeRetriever()

    if args.list:
        print(f"Security Microcode V0 — {len(retriever.microcodes)} microcodes\n")
        print(f"{'ID':<35} {'Category':<25} {'CWE':<10} {'Language':<12}")
        print("-" * 85)
        for mc in retriever.microcodes:
            print(f"{mc['id']:<35} {mc['category']:<25} {mc['cwe']:<10} {mc['language']:<12}")
    elif args.search:
        print(f"Searching for: '{args.search}'\n")
        results = retriever.retrieve(args.search, k=args.k)
        print(f"Top {len(results)} results:\n")
        for i, mc in enumerate(results, 1):
            print(f"{i}. {mc['id']} ({mc['category']}, {mc['cwe']})")
            print(f"   S: {mc['S'][:80]}")
            print(f"   K: {mc['K'][:80]}")
            print(f"   O: {mc['O'][:80]}")
            print(f"   Microcode: {mc['microcode'][:100]}...")
            print()
    else:
        print("Use --list or --search 'query'")


def main():
    parser = argparse.ArgumentParser(
        description='DreamGTM CLI - Code + Security Patching AI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest='command')

    # Chat
    chat_parser = subparsers.add_parser('chat', help='Interactive chat')
    chat_parser.add_argument('--model', type=str, default=None)
    chat_parser.set_defaults(func=cmd_chat)

    # Scan
    scan_parser = subparsers.add_parser('scan', help='Scan repo for vulnerabilities')
    scan_parser.add_argument('--repo', type=str, required=True)
    scan_parser.set_defaults(func=cmd_scan)

    # Patch
    patch_parser = subparsers.add_parser('patch', help='Patch a file')
    patch_parser.add_argument('--file', type=str, required=True)
    patch_parser.add_argument('--model', type=str, default=None)
    patch_parser.set_defaults(func=cmd_patch)

    # Code
    code_parser = subparsers.add_parser('code', help='Generate code from prompt')
    code_parser.add_argument('prompt', type=str)
    code_parser.add_argument('--model', type=str, default=None)
    code_parser.set_defaults(func=cmd_code)

    # Eval
    eval_parser = subparsers.add_parser('eval', help='Run eval harness')
    eval_parser.add_argument('--model', type=str, required=True)
    eval_parser.add_argument('--suite', type=str, default='all',
                            choices=['all', 'syntax', 'mbpp', 'security_patch',
                                    'scanner', 'no_regression', 'defensive_boundary'])
    eval_parser.add_argument('--n', type=int, default=20)
    eval_parser.set_defaults(func=cmd_eval)

    # Microcode
    mc_parser = subparsers.add_parser('microcode', help='List or search security microcodes')
    mc_group = mc_parser.add_mutually_exclusive_group(required=True)
    mc_group.add_argument('--list', action='store_true', help='List all microcodes')
    mc_group.add_argument('--search', type=str, help='Search microcodes')
    mc_parser.add_argument('--k', type=int, default=3, help='Top K results for search')
    mc_parser.set_defaults(func=cmd_microcode)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    args.func(args)


if __name__ == '__main__':
    main()
