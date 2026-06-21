"""
Continue fetching security patches - skip already-fetched URLs.
Append mode to existing file.
"""
import os
import json
import time
import hashlib
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

IN_FILE = Path("/home/z/my-project/dreamgtm/data/verified/vulnerabilities.jsonl")
OUT_FILE = Path("/home/z/my-project/dreamgtm/data/verified/security_patches.jsonl")


def make_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def fetch_patch(url: str) -> str:
    req = urllib.request.Request(url, headers={
        "User-Agent": "DreamGTM-DataCollector/1.0",
        "Accept": "text/plain, */*",
    })
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            if resp.status != 200:
                return ""
            data = resp.read()
            try:
                return data.decode("utf-8")
            except UnicodeDecodeError:
                return data.decode("latin-1", errors="replace")
    except Exception:
        return ""


def extract_code_diffs(patch_text: str) -> list:
    if not patch_text:
        return []
    files = []
    current_file = None
    current_lines = []
    in_hunk = False
    for line in patch_text.split("\n"):
        if line.startswith("diff --git"):
            if current_file and current_lines:
                files.append({"file": current_file, "diff": "\n".join(current_lines)})
            current_file = None
            current_lines = []
            in_hunk = False
        elif line.startswith("+++ b/"):
            current_file = line[6:].strip()
        elif line.startswith("@@"):
            in_hunk = True
            current_lines.append(line)
        elif in_hunk:
            current_lines.append(line)
    if current_file and current_lines:
        files.append({"file": current_file, "diff": "\n".join(current_lines)})
    return [f for f in files if any(f["file"].endswith(ext) for ext in
            [".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".java", ".c", ".h",
             ".cpp", ".cc", ".hpp", ".rb", ".php", ".rs", ".kt", ".swift",
             ".scala", ".sh", ".lua", ".pl", ".r", ".dart", ".cs", ".m"])]


def fetch_one(vuln):
    url = vuln["patch_url"]
    if not url.endswith(".patch"):
        url = url + ".patch"
    patch_text = fetch_patch(url)
    if not patch_text or len(patch_text) < 100:
        return None
    code_files = extract_code_diffs(patch_text)
    if not code_files:
        return None
    return {
        "id": make_id(vuln["vuln_id"] + vuln["patch_url"]),
        "type": "security_patch",
        "vuln_id": vuln["vuln_id"],
        "summary": vuln["summary"],
        "description": vuln["description"],
        "ecosystem": vuln["ecosystem"],
        "aliases": vuln["aliases"],
        "affected": vuln["affected"],
        "patch_url": vuln["patch_url"],
        "patch_files": code_files,
        "patch_chars": len(patch_text),
    }


def main():
    print(f"[{datetime.now():%H:%M:%S}] Continuing security patch fetch...", flush=True)

    # Load already-fetched URLs
    seen_urls = set()
    if OUT_FILE.exists():
        with OUT_FILE.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    seen_urls.add(rec.get("patch_url", ""))
                except json.JSONDecodeError:
                    continue
    print(f"Already fetched: {len(seen_urls)} patches", flush=True)

    # Load vulns
    vulns = []
    with IN_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                for url in rec.get("patch_urls", []):
                    if "github.com" in url and "/commit/" in url and url not in seen_urls:
                        vulns.append({
                            "vuln_id": rec.get("vuln_id", ""),
                            "summary": rec.get("summary", ""),
                            "description": rec.get("description", "")[:1000],
                            "ecosystem": rec.get("ecosystem", ""),
                            "patch_url": url,
                            "affected": rec.get("affected", [])[:5],
                            "aliases": rec.get("aliases", []),
                        })
            except json.JSONDecodeError:
                continue
    print(f"To fetch: {len(vulns)} new URLs", flush=True)

    kept = 0
    skipped = 0
    bytes_written = 0
    start = time.time()

    with OUT_FILE.open("a", encoding="utf-8", buffering=1) as out:
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(fetch_one, v): v for v in vulns}
            for future in as_completed(futures):
                result = future.result()
                if result is None:
                    skipped += 1
                else:
                    line = json.dumps(result, ensure_ascii=False)
                    out.write(line + "\n")
                    kept += 1
                    bytes_written += len(line) + 1
                total_done = kept + skipped
                if total_done % 100 == 0:
                    elapsed = time.time() - start
                    print(f"  done={total_done} kept={kept} skipped={skipped} | "
                          f"{bytes_written/1e6:.1f}MB | "
                          f"{total_done/elapsed:.1f}/s | "
                          f"elapsed={elapsed:.0f}s", flush=True)
                    out.flush()

    elapsed = time.time() - start
    print(f"\nDONE: +{kept} new patches ({bytes_written/1e6:.1f} MB)")
    print(f"Skipped: {skipped}")
    print(f"Time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
