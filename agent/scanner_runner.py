"""
DreamPatch - Scanner Runner
=============================
Runs security scanners and parses results:
- Semgrep (code analysis)
- Gitleaks (secret detection)
- OSV Scanner (dependency vulnerabilities)
- npm audit (Node.js dependencies)
- pip-audit (Python dependencies)
"""
import subprocess
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class ScanResult:
    """Result of a security scan"""
    scanner: str
    findings: List[Dict] = field(default_factory=list)
    raw_output: str = ""
    error: Optional[str] = None
    success: bool = False


class ScannerRunner:
    """Runs security scanners on a repository"""
    
    SCANNERS = {
        'semgrep': {
            'command': 'semgrep --config p/owasp-top-ten --config p/cwe-top-25 --json {path}',
            'description': 'Code analysis (OWASP + CWE)',
        },
        'gitleaks': {
            'command': 'gitleaks detect --source {path} --report-format json --report-path {path}/gitleaks.json',
            'description': 'Secret detection',
        },
        'osv': {
            'command': 'osv-scanner -r {path} --format json',
            'description': 'Dependency vulnerabilities',
        },
        'npm_audit': {
            'command': 'cd {path} && npm audit --json',
            'description': 'Node.js dependency audit',
        },
        'pip_audit': {
            'command': 'cd {path} && pip-audit -f json',
            'description': 'Python dependency audit',
        },
    }
    
    def run_scanner(self, scanner_name: str, repo_path: str) -> ScanResult:
        """Run a single scanner"""
        if scanner_name not in self.SCANNERS:
            return ScanResult(scanner=scanner_name, error=f"Unknown scanner: {scanner_name}")
        
        config = self.SCANNERS[scanner_name]
        command = config['command'].format(path=repo_path)
        
        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True, timeout=60
            )
            
            scan_result = ScanResult(
                scanner=scanner_name,
                raw_output=result.stdout + result.stderr,
                success=True,
            )
            
            # Parse JSON output if available
            try:
                data = json.loads(result.stdout)
                if 'results' in data:
                    scan_result.findings = data['results']
                elif 'vulns' in data:
                    scan_result.findings = data['vulns']
                elif isinstance(data, list):
                    scan_result.findings = data
            except:
                # Parse text output
                for line in result.stdout.split('\n'):
                    if 'error' in line.lower() or 'warning' in line.lower() or 'vulnerability' in line.lower():
                        scan_result.findings.append({'description': line.strip()})
            
            return scan_result
            
        except subprocess.TimeoutExpired:
            return ScanResult(scanner=scanner_name, error="Timeout")
        except FileNotFoundError:
            return ScanResult(scanner=scanner_name, error=f"{scanner_name} not installed")
        except Exception as e:
            return ScanResult(scanner=scanner_name, error=str(e))
    
    def run_all(self, repo_path: str, scanners: List[str] = None) -> Dict[str, ScanResult]:
        """Run multiple scanners"""
        if scanners is None:
            scanners = list(self.SCANNERS.keys())
        
        results = {}
        for scanner in scanners:
            print(f"  Running {scanner}...")
            results[scanner] = self.run_scanner(scanner, repo_path)
        
        return results
    
    def summarize(self, results: Dict[str, ScanResult]) -> str:
        """Generate summary of all scan results"""
        lines = ["=" * 60, "DreamPatch Scan Summary", "=" * 60]
        
        total_findings = 0
        for name, result in results.items():
            status = "OK" if result.success and not result.findings else "ISSUES FOUND"
            count = len(result.findings) if result.findings else 0
            total_findings += count
            lines.append(f"\n{name}: {status} ({count} findings)")
            if result.error:
                lines.append(f"  Error: {result.error}")
            for finding in result.findings[:5]:
                if isinstance(finding, dict):
                    desc = finding.get('description', finding.get('message', str(finding)[:100]))
                    lines.append(f"  - {desc}")
        
        lines.append(f"\nTotal findings: {total_findings}")
        lines.append("\n" + "=" * 60)
        return '\n'.join(lines)


if __name__ == "__main__":
    runner = ScannerRunner()
    
    # Test (won't actually run scanners, just shows interface)
    print("DreamPatch Scanner Runner")
    print("\nAvailable scanners:")
    for name, config in runner.SCANNERS.items():
        print(f"  - {name}: {config['description']}")
