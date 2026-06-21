"""
Security Microcode Retriever
==============================
Given a user prompt, retrieves the top-K relevant microcodes via keyword matching.

Usage:
    from research.security_microcode_v0.retriever import MicrocodeRetriever
    retriever = MicrocodeRetriever()
    results = retriever.retrieve("patch SQL injection", k=3)
"""
import json
import re
from pathlib import Path
from typing import List, Dict


class MicrocodeRetriever:
    def __init__(self, dataset_path: str = None):
        if dataset_path is None:
            dataset_path = Path(__file__).parent / "microcode_dataset.jsonl"
        else:
            dataset_path = Path(dataset_path)
        
        self.microcodes = []
        with dataset_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        self.microcodes.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        # Build keyword index
        self._build_index()
    
    def _build_index(self):
        """Build a keyword → microcode index for fast retrieval."""
        self.keyword_index = {}
        for mc in self.microcodes:
            # Index by category, cwe, language, detection keywords
            keywords = set()
            keywords.add(mc.get("category", ""))
            keywords.add(mc.get("cwe", "").lower())
            keywords.add(mc.get("owasp", "").lower())
            keywords.add(mc.get("language", ""))
            for kw in mc.get("detection_keywords", []):
                keywords.add(kw.lower())
            # Also index by words in the microcode itself
            for field in ["S", "K", "F", "V", "O", "P", "rationale", "vuln_pattern", "microcode"]:
                text = mc.get(field, "").lower()
                for word in re.findall(r'\w+', text):
                    if len(word) > 3:
                        keywords.add(word)
            
            for kw in keywords:
                if kw not in self.keyword_index:
                    self.keyword_index[kw] = []
                self.keyword_index[kw].append(mc["id"])
    
    def retrieve(self, query: str, k: int = 3) -> List[Dict]:
        """Retrieve top-K microcodes relevant to the query."""
        query_lower = query.lower()
        scores = {}
        
        for mc in self.microcodes:
            score = 0
            # Check detection keywords
            for kw in mc.get("detection_keywords", []):
                if kw.lower() in query_lower:
                    score += 3  # High weight for direct keyword match
            # Check category
            if mc.get("category", "").replace("_", " ") in query_lower:
                score += 5
            # Check CWE
            cwe = mc.get("cwe", "").lower()
            if cwe and cwe in query_lower:
                score += 4
            # Check owasp
            owasp = mc.get("owasp", "").lower()
            if owasp and owasp in query_lower:
                score += 3
            # Check language
            lang = mc.get("language", "")
            if lang and lang in query_lower:
                score += 1
            # Check vuln_pattern keywords
            vuln = mc.get("vuln_pattern", "").lower()
            for word in re.findall(r'\w+', vuln):
                if len(word) > 3 and word in query_lower:
                    score += 1
            
            if score > 0:
                scores[mc["id"]] = (score, mc)
        
        # Sort by score, return top K
        sorted_results = sorted(scores.values(), key=lambda x: -x[0])
        return [mc for _, mc in sorted_results[:k]]
    
    def list_all(self) -> List[Dict]:
        """Return all microcodes."""
        return self.microcodes
    
    def get_by_category(self, category: str) -> List[Dict]:
        """Get all microcodes for a specific category."""
        return [mc for mc in self.microcodes if mc.get("category") == category]
    
    def get_by_cwe(self, cwe: str) -> List[Dict]:
        """Get all microcodes for a specific CWE."""
        return [mc for mc in self.microcodes if mc.get("cwe", "").lower() == cwe.lower()]


if __name__ == "__main__":
    # Test
    retriever = MicrocodeRetriever()
    print(f"Loaded {len(retriever.microcodes)} microcodes")
    print(f"Index has {len(retriever.keyword_index)} keywords")
    
    # Test retrieval
    test_queries = [
        "patch this SQL injection: cursor.execute(f'SELECT * FROM users WHERE id={user_id}')",
        "fix XSS vulnerability in my HTML output",
        "secure this password hashing code: hashlib.md5(password.encode())",
        "prevent path traversal in file download",
        "fix command injection: os.system(f'ping {host}')",
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query[:80]}...")
        results = retriever.retrieve(query, k=3)
        for mc in results:
            print(f"  → {mc['id']} ({mc['category']})")
