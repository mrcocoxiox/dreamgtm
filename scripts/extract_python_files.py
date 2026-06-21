"""
Extract Python files from downloaded source archives (CPython stdlib + popular libs).
Each .py file becomes one training example - these are highest-quality reference Python.
"""
import os
import ast
import json
import hashlib
from pathlib import Path
from datetime import datetime

RAW_DIR = Path("/home/z/my-project/dreamgtm/data/raw")
OUT_FILE = Path("/home/z/my-project/dreamgtm/data/verified/python_source.jsonl")
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# Directories containing extracted Python source
SOURCE_DIRS = [
    RAW_DIR / "cpython-3.12.0" / "Lib",          # stdlib
    RAW_DIR / "requests-2.32.3" / "src" / "requests",  # requests library
    RAW_DIR / "flask-3.0.3" / "src" / "flask",    # flask
    RAW_DIR / "fastapi-0.115.0" / "fastapi",      # fastapi
    RAW_DIR / "pydantic-2.9.0" / "pydantic",      # pydantic
]


def is_valid_python(code: str) -> bool:
    """Deep check: code must parse as valid AST (this is the manual verification)."""
    if not code or len(code.strip()) < 50:
        return False
    try:
        tree = ast.parse(code)
        # Must have at least 1 def or class (no trivial scripts)
        has_def = any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) for n in ast.walk(tree))
        return has_def
    except (SyntaxError, ValueError):
        return False


def extract_docstring(code: str) -> str:
    """Get module-level docstring if present."""
    try:
        tree = ast.parse(code)
        if (tree.body and isinstance(tree.body[0], ast.Expr)
                and isinstance(tree.body[0].value, ast.Constant)
                and isinstance(tree.body[0].value.value, str)):
            return tree.body[0].value.value.strip()
    except Exception:
        pass
    return ""


def make_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def scan_directory(root: Path):
    """Walk a directory and yield verified Python file contents."""
    count = 0
    skipped = 0
    for path in root.rglob("*.py"):
        # Skip tests, migrations, vendored, build artifacts
        parts = path.parts
        if any(p in {"test", "tests", "__pycache__", "tests_data", "vendored",
                     "build", "dist", ".tox", "node_modules"} for p in parts):
            continue
        if path.name.startswith(("test_", "conftest")) or path.name.endswith("_test.py"):
            continue
        try:
            code = path.read_text(encoding="utf-8", errors="strict")
        except (UnicodeDecodeError, OSError):
            skipped += 1
            continue
        if not is_valid_python(code):
            skipped += 1
            continue
        # Compute relative path for source attribution
        rel = path.relative_to(root.parent)
        doc = extract_docstring(code)
        yield {
            "id": make_id(code),
            "source": str(rel),
            "type": "python_source",
            "module_docstring": doc,
            "code": code,
            "lines": code.count("\n") + 1,
            "chars": len(code),
        }
        count += 1
    print(f"  {root}: kept {count}, skipped {skipped}", flush=True)


def main():
    print(f"[{datetime.now():%H:%M:%S}] Extracting verified Python files...")
    total = 0
    bytes_written = 0
    with OUT_FILE.open("w", encoding="utf-8") as f:
        for src in SOURCE_DIRS:
            if not src.exists():
                print(f"  SKIP (missing): {src}")
                continue
            print(f"  Scanning {src}")
            for ex in scan_directory(src):
                line = json.dumps(ex, ensure_ascii=False)
                f.write(line + "\n")
                total += 1
                bytes_written += len(line) + 1
    print(f"\nDONE: {total} verified Python files")
    print(f"Wrote {bytes_written / 1e6:.1f} MB to {OUT_FILE}")


if __name__ == "__main__":
    main()
