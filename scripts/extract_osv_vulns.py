"""
Extract real vulnerability data from OSV.dev (Open Source Vulnerability Database).
Each CVE becomes a security training example with: vulnerability description,
affected versions, and (where available) the patch reference.

Source: https://osv.dev (downloaded from Google Cloud Storage)
License: CC-BY 4.0 (OSV data is open)
"""
import os
import json
import zipfile
import hashlib
from pathlib import Path
from datetime import datetime

RAW_DIR = Path("/home/z/my-project/dreamgtm/data/raw/osv")
OUT_FILE = Path("/home/z/my-project/dreamgtm/data/verified/vulnerabilities.jsonl")
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)


def make_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def extract_vuln_record(record: dict) -> dict | None:
    """Pull the useful fields out of an OSV record. Return None if record is too sparse."""
    if not record:
        return None
    vuln_id = record.get("id", "")
    summary = record.get("summary", "").strip()
    details = record.get("details", "").strip()

    if not summary and not details:
        return None

    # Build a clean description (drop raw HTML/Markdown noise)
    description = summary
    if details and len(details) > len(summary):
        # Use details but trim to first 2000 chars to keep examples manageable
        description = details[:2000]

    # Extract affected packages + ranges
    affected = []
    for aff in record.get("affected", []):
        pkg = aff.get("package", {})
        eco = pkg.get("ecosystem", "")
        name = pkg.get("name", "")
        ranges = []
        for r in aff.get("ranges", []):
            events = r.get("events", [])
            ranges.append({"type": r.get("type", ""), "events": events})
        versions = aff.get("versions", [])[:20]  # cap to avoid huge examples
        if name and eco:
            affected.append({
                "ecosystem": eco,
                "package": name,
                "ranges": ranges,
                "versions_sample": versions,
            })

    # Extract references (often contain the patch URL)
    references = []
    for ref in record.get("references", []):
        url = ref.get("url", "")
        rtype = ref.get("type", "")
        if url:
            references.append({"type": rtype, "url": url})

    # Severity (CVSS scores etc.)
    severity = []
    for sev in record.get("severity", []):
        severity.append({"type": sev.get("type", ""), "score": sev.get("score", "")})

    # Aliases (CVE IDs, GHSA IDs)
    aliases = record.get("aliases", [])

    # Credits
    credits = []
    for c in record.get("credits", []):
        credits.append({"name": c.get("name", ""), "type": c.get("type", "")})

    # Build the patch URL list (most useful for our purposes)
    patch_urls = [r["url"] for r in references if r["type"] in {"FIX", "PATCH"}]
    advisory_urls = [r["url"] for r in references if r["type"] in {"ADVISORY", "WEB"}]

    text_blob = f"{vuln_id}\n{summary}\n{description}\n{json.dumps(affected)}"
    return {
        "id": make_id(text_blob),
        "vuln_id": vuln_id,
        "type": "vulnerability",
        "summary": summary,
        "description": description,
        "aliases": aliases,
        "affected": affected,
        "severity": severity,
        "patch_urls": patch_urls,
        "advisory_urls": advisory_urls,
        "credits": credits,
        "published": record.get("published", ""),
        "modified": record.get("modified", ""),
    }


def process_zip(zip_path: Path, out_handle, stats: dict):
    """Stream-extract all JSON files from an OSV ecosystem zip."""
    ecosystem = zip_path.stem
    kept = 0
    skipped = 0
    seen_ids = set()
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            for name in zf.namelist():
                if not name.endswith(".json"):
                    continue
                try:
                    with zf.open(name) as f:
                        data = json.load(f)
                except (json.JSONDecodeError, OSError):
                    skipped += 1
                    continue
                rec = extract_vuln_record(data)
                if rec is None or rec["id"] in seen_ids:
                    skipped += 1
                    continue
                seen_ids.add(rec["id"])
                rec["ecosystem"] = ecosystem
                out_handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += 1
    except zipfile.BadZipFile:
        print(f"  BAD ZIP: {zip_path}")
    stats["kept"] += kept
    stats["skipped"] += skipped
    print(f"  {ecosystem}: kept {kept}, skipped {skipped}", flush=True)


def main():
    print(f"[{datetime.now():%H:%M:%S}] Extracting vulnerabilities from OSV.dev...")
    stats = {"kept": 0, "skipped": 0}
    bytes_written = 0
    with OUT_FILE.open("w", encoding="utf-8") as f:
        for zip_path in sorted(RAW_DIR.glob("*.zip")):
            process_zip(zip_path, f, stats)
    print(f"\nDONE: {stats['kept']} verified vulnerability records")
    print(f"Skipped: {stats['skipped']}")
    print(f"Wrote {OUT_FILE.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
