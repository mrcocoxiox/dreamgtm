"""
Extract Python source from additional high-quality libraries:
Django, Werkzeug, Jinja2, Click, Uvicorn, Starlette
"""
import os
import ast
import json
import hashlib
from pathlib import Path
from datetime import datetime

RAW_DIR = Path("/home/z/my-project/dreamgtm/data/raw/p3code")
OUT_FILE = Path("/home/z/my-project/dreamgtm/data/verified/python_source_2.jsonl")
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

SOURCE_DIRS = [
    RAW_DIR / "django-4.2" / "django",
    RAW_DIR / "werkzeug-3.0.4" / "src" / "werkzeug",
    RAW_DIR / "jinja-3.1.4" / "src" / "jinja2",
    RAW_DIR / "click-8.1.7" / "src" / "click",
    RAW_DIR / "uvicorn-0.30.6" / "uvicorn",
    RAW_DIR / "starlette-0.38.2" / "starlette",
]


def make_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def is_valid_python(code: str) -> bool:
    if not code or len(code.strip()) < 50:
        return False
    try:
        tree = ast.parse(code)
        has_def = any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) for n in ast.walk(tree))
        return has_def
    except (SyntaxError, ValueError):
        return False


def extract_docstring(code: str) -> str:
    try:
        tree = ast.parse(code)
        if (tree.body and isinstance(tree.body[0], ast.Expr)
                and isinstance(tree.body[0].value, ast.Constant)
                and isinstance(tree.body[0].value.value, str)):
            return tree.body[0].value.value.strip()
    except Exception:
        pass
    return ""


def scan_directory(root: Path):
    count = 0
    skipped = 0
    for path in root.rglob("*.py"):
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
    print(f"[{datetime.now():%H:%M:%S}] Extracting additional Python sources...")
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
