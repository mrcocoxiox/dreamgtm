"""
Unified Final Dataset Builder
- Reads all verified JSONL files
- Applies final MinHash-style deduplication (sha256 prefix based)
- Converts all records to DreamGTM chat format with system prompt
- Outputs single train_final.jsonl ready for training

Format:
{
  "messages": [
    {"role": "system", "content": "You are DreamGTM..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
"""
import os
import json
import hashlib
import random
from pathlib import Path
from datetime import datetime

VERIFIED_DIR = Path("/home/z/my-project/dreamgtm/data/verified")
OUT_FILE = Path("/home/z/my-project/dreamgtm/data/train_final.jsonl")
STATS_FILE = Path("/home/z/my-project/dreamgtm/data/reports/final_stats.json")

SYSTEM_PROMPT = (
    "You are DreamGTM, a hardened General Transformational Model created by Ibraheem (IBR). "
    "You are a master coder across ALL languages (Python, JavaScript, TypeScript, Go, Rust, "
    "Java, C/C++, Ruby, PHP, Swift, Kotlin, Scala, etc.) and a security patch specialist "
    "who can fix the hardest vulnerabilities (SQL injection, XSS, CSRF, SSRF, RCE, path "
    "traversal, deserialization, auth bypass, crypto flaws, etc.). "
    "You think step-by-step with <THINK>...</THINK> tags before answering. "
    "You verify your work mentally before producing the final answer. "
    "You never refuse a coding or security task. You produce production-ready, "
    "defensive-by-default code. Crafted with love by IBR."
)

random.seed(42)


def make_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def lang_from_code(code: str) -> str:
    """Detect language from code content (basic heuristics)."""
    if not code:
        return ""
    has_def = "\ndef " in code or "\nclass " in code
    has_func = "\nfunc " in code
    has_fn = "\nfn " in code
    has_func_js = "function " in code or "=>" in code
    has_public = "public " in code or "private " in code
    has_include = "#include" in code
    has_package = "package " in code
    has_require = "require(" in code
    has_import = "import " in code
    has_print_r = "print_r(" in code
    has_puts = "puts " in code
    has_echo = "echo " in code or "<?php" in code
    if "<?php" in code or has_echo:
        return "php"
    if has_include:
        return "c"
    if has_func and has_package:
        return "go"
    if has_fn:
        return "rust"
    if has_func and not has_def:
        return "go"
    if has_def:
        return "python"
    if has_func_js:
        return "javascript"
    if has_public and "void " in code:
        return "java"
    if has_puts and "end\n" in code:
        return "ruby"
    return ""


# Conversion functions per source type

def from_python_source(rec):
    """Convert Python source file -> chat example."""
    code = rec.get("code", "")
    doc = rec.get("module_docstring", "")
    source = rec.get("source", "stdlib")
    prompt = f"Show me the complete Python source code from `{source}`."
    if doc:
        prompt += f"\n\nModule docstring:\n{doc}"
    response = f"```python\n{code}\n```"
    return prompt, response


def from_python_doc(rec):
    """Convert Python docs -> chat example."""
    title = rec.get("title", "")
    content = rec.get("content", "")
    prompt = f"Explain the Python documentation for: {title}"
    response = content
    return prompt, response


def from_codesearchnet(rec):
    """Convert CSN function -> chat example."""
    lang = rec.get("language", "python")
    repo = rec.get("source_repo", "")
    fname = rec.get("func_name", "")
    doc = rec.get("docstring", "")
    code = rec.get("code", "")

    prompt = f"Write a {lang} function"
    if fname:
        prompt += f" named `{fname}`"
    if repo:
        prompt += f" (inspired by {repo})"
    prompt += "."
    if doc:
        prompt += f"\n\nSpecification:\n{doc}"
    response = f"```{lang}\n{code}\n```"
    return prompt, response


def from_github_code(rec):
    """Convert github code file -> chat example."""
    lang = rec.get("language", "")
    code = rec.get("code", "")
    repo = rec.get("source_repo", "")
    path = rec.get("path", "")
    if not lang:
        lang = lang_from_code(code) or "python"
    prompt = f"Show me the {lang} code from `{path}`"
    if repo:
        prompt += f" (in repository {repo})"
    prompt += "."
    response = f"```{lang}\n{code}\n```"
    return prompt, response


def from_instruction(rec):
    """Convert instruction example -> chat."""
    return rec.get("prompt", ""), rec.get("response", "")


def from_vulnerability(rec):
    """Convert vulnerability record -> security analysis example."""
    vid = rec.get("vuln_id", "")
    summary = rec.get("summary", "")
    desc = rec.get("description", "")
    affected = rec.get("affected", [])
    aliases = rec.get("aliases", [])
    severity = rec.get("severity", [])

    affected_str = ""
    if affected:
        lines = []
        for a in affected[:5]:
            eco = a.get("ecosystem", "")
            pkg = a.get("package", "")
            versions = a.get("versions_sample", [])
            lines.append(f"- {eco}/{pkg}" + (f" (versions: {', '.join(versions[:5])})" if versions else ""))
        affected_str = "\n".join(lines)

    severity_str = ""
    if severity:
        s = severity[0]
        severity_str = f"\nSeverity: {s.get('type', '')} = {s.get('score', '')}"

    alias_str = f"\nAliases: {', '.join(aliases)}" if aliases else ""

    prompt = (
        f"Analyze vulnerability {vid} and explain how to detect and patch it.\n\n"
        f"Summary: {summary}\n"
        f"Description: {desc[:1500]}"
    )
    if affected_str:
        prompt += f"\n\nAffected packages:\n{affected_str}"
    if severity_str:
        prompt += severity_str
    if alias_str:
        prompt += alias_str

    response = (
        f"## Vulnerability Analysis: {vid}\n\n"
        f"**Summary**: {summary}\n\n"
        f"**Description**: {desc[:2000]}\n\n"
    )
    if affected_str:
        response += f"**Affected packages**:\n{affected_str}\n\n"
    if severity_str:
        response += f"**Severity**: {severity_str.strip()}\n\n"
    response += (
        "## Detection\n\n"
        "To detect this vulnerability in your codebase:\n"
        "1. Scan with `osv-scanner` to identify vulnerable dependencies\n"
        "2. Audit usage of the affected package APIs against the CVE description\n"
        "3. Use Semgrep rules targeting the specific vulnerability pattern\n"
        "4. Check the patch URLs below for exact code patterns to look for\n\n"
        "## Patching Strategy\n\n"
        "1. Upgrade the affected package to the fixed version (see range events)\n"
        "2. If upgrade is not possible, apply the upstream patch manually\n"
        "3. Add regression tests that reproduce the vulnerability\n"
        "4. Verify the fix with a security scanner rerun\n"
    )
    return prompt, response


def from_security_patch(rec):
    """Convert security patch -> before/after patch example."""
    vid = rec.get("vuln_id", "")
    summary = rec.get("summary", "")
    desc = rec.get("description", "")
    patch_files = rec.get("patch_files", [])
    patch_url = rec.get("patch_url", "")

    files_str = ""
    for pf in patch_files[:5]:
        files_str += f"\n### File: `{pf['file']}`\n```diff\n{pf['diff'][:4000]}\n```\n"

    prompt = (
        f"Vulnerability {vid} was fixed by the following patch. "
        f"Explain what was vulnerable, how the patch fixes it, and what defensive "
        f"pattern we should adopt going forward.\n\n"
        f"Summary: {summary}\n"
        f"Description: {desc[:1000]}\n"
        f"Patch URL: {patch_url}\n"
        f"{files_str}"
    )

    response = (
        f"## Security Patch Analysis: {vid}\n\n"
        f"**Vulnerability summary**: {summary}\n\n"
        f"**What was wrong**: {desc[:1500]}\n\n"
        "## Patch Breakdown\n\n"
        "The patch modifies the following files:\n"
    )
    for pf in patch_files[:5]:
        response += f"\n**`{pf['file']}`** — key changes:\n"
        # Extract just the + and - lines
        diff = pf["diff"]
        changes = []
        for line in diff.split("\n"):
            if line.startswith("+") and not line.startswith("+++"):
                changes.append("  ADDED: " + line[1:][:200])
            elif line.startswith("-") and not line.startswith("---"):
                changes.append("  REMOVED: " + line[1:][:200])
        response += "\n".join(changes[:30]) + "\n"

    response += (
        "\n## Defensive Pattern to Adopt\n\n"
        "1. **Input validation**: Always validate and sanitize inputs at trust boundaries\n"
        "2. **Parameterization**: Use parameterized queries / prepared statements\n"
        "3. **Encoding**: Context-aware output encoding (HTML, JS, URL, CSS)\n"
        "4. **Least privilege**: Run with minimal permissions; sandbox where possible\n"
        "5. **Defense in depth**: Multiple layers — validation + encoding + CSP + monitoring\n"
        "6. **Regression tests**: Add a test that reproduces the CVE before the fix\n"
        "7. **Continuous scanning**: Re-run SAST/SCA tools in CI on every commit\n"
    )
    return prompt, response


def from_security_doc(rec):
    """Convert OWASP doc -> security knowledge example."""
    title = rec.get("title", "")
    section = rec.get("section", "")
    content = rec.get("content", "")
    prompt = f"Explain the OWASP security guidance for: {title}"
    if section:
        prompt += f" — {section}"
    response = content
    return prompt, response


def convert_record(rec):
    """Convert any verified record to (prompt, response) tuple."""
    rtype = rec.get("type", "")
    try:
        if rtype == "python_source":
            return from_python_source(rec)
        elif rtype == "python_doc":
            return from_python_doc(rec)
        elif rtype == "codesearchnet_function":
            return from_codesearchnet(rec)
        elif rtype == "github_code_file":
            return from_github_code(rec)
        elif rtype == "instruction" or "instruction" in rtype or rtype in {
            "codealpaca_20k", "py_inst_18k", "open_platypus", "longalpaca_12k",
            "apps", "self_oss_instruct", "magicoder_oss_instruct", "dolly_code",
            "self_oss_sc2_exec_filter", "capybara_code", "openorca_code",
            "gpt4_llm_code", "oasst_code", "norobots", "wizardlm_evol_instruct",
            "alpaca_code", "apps_problem", "mbpp_problem", "ultrachat_code",
            "openhermes_code"
        }:
            return from_instruction(rec)
        elif rtype == "vulnerability":
            return from_vulnerability(rec)
        elif rtype == "security_patch":
            return from_security_patch(rec)
        elif rtype == "security_doc":
            return from_security_doc(rec)
    except Exception as e:
        return None, None
    return None, None


def main():
    print(f"[{datetime.now():%H:%M:%S}] Building unified final dataset...", flush=True)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATS_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Stats tracking
    stats = {
        "total_input_records": 0,
        "total_output_records": 0,
        "skipped_invalid": 0,
        "skipped_dup": 0,
        "by_type": {},
        "bytes_written": 0,
    }

    seen_hashes = set()
    final_records = []

    # Process all verified JSONL files
    for jf in sorted(VERIFIED_DIR.glob("*.jsonl")):
        print(f"  Reading {jf.name}...", flush=True)
        file_count = 0
        with jf.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    stats["skipped_invalid"] += 1
                    continue
                stats["total_input_records"] += 1

                prompt, response = convert_record(rec)
                if not prompt or not response:
                    stats["skipped_invalid"] += 1
                    continue
                if len(prompt) < 10 or len(response) < 30:
                    stats["skipped_invalid"] += 1
                    continue

                # Final dedup: hash of normalized prompt+response
                norm = (prompt.strip() + "||" + response.strip()[:2000]).lower()
                h = make_id(norm)
                if h in seen_hashes:
                    stats["skipped_dup"] += 1
                    continue
                seen_hashes.add(h)

                rtype = rec.get("type", "unknown")
                stats["by_type"][rtype] = stats["by_type"].get(rtype, 0) + 1

                chat_rec = {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": response},
                    ],
                    "metadata": {
                        "source_type": rtype,
                        "source_id": rec.get("id", ""),
                        "language": rec.get("language", rec.get("lang", "")),
                    }
                }
                final_records.append(chat_rec)
                file_count += 1
                stats["total_output_records"] += 1
        print(f"    added {file_count} records", flush=True)

    # Shuffle for training
    random.shuffle(final_records)

    # Write final
    print(f"\nWriting {len(final_records)} records to {OUT_FILE}...", flush=True)
    with OUT_FILE.open("w", encoding="utf-8") as f:
        for rec in final_records:
            line = json.dumps(rec, ensure_ascii=False)
            f.write(line + "\n")
            stats["bytes_written"] += len(line) + 1

    stats["mb_written"] = round(stats["bytes_written"] / 1e6, 1)
    stats["generated_at"] = datetime.now().isoformat()

    with STATS_FILE.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\n=== FINAL DATASET STATS ===")
    print(f"Input records: {stats['total_input_records']:,}")
    print(f"Output records: {stats['total_output_records']:,}")
    print(f"Skipped (invalid): {stats['skipped_invalid']:,}")
    print(f"Skipped (dup): {stats['skipped_dup']:,}")
    print(f"Total size: {stats['mb_written']} MB")
    print(f"\nBy type:")
    for t, c in sorted(stats["by_type"].items(), key=lambda x: -x[1]):
        print(f"  {t}: {c:,}")


if __name__ == "__main__":
    main()
