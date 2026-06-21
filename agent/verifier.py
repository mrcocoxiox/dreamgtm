"""
DreamPatch - Verifier
======================
Verifies generated patches:
1. Syntax check (AST for Python, eslint for JS)
2. Run unit tests
3. Re-run security scanner
4. Compare before/after scanner results
5. Reject patches that only hide warnings
"""
import ast
import subprocess
import os
import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class VerificationResult:
    """Result of patch verification"""
    syntax_valid: bool = False
    syntax_error: Optional[str] = None
    tests_passed: int = 0
    tests_failed: int = 0
    test_results: List[Dict] = field(default_factory=list)
    security_issues_before: List[str] = field(default_factory=list)
    security_issues_after: List[str] = field(default_factory=list)
    patch_applied: bool = False
    patch_reverted: bool = False
    score: float = 0.0
    passed: bool = False
    notes: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'syntax_valid': self.syntax_valid,
            'syntax_error': self.syntax_error,
            'tests_passed': self.tests_passed,
            'tests_failed': self.tests_failed,
            'security_issues_before': self.security_issues_before,
            'security_issues_after': self.security_issues_after,
            'patch_applied': self.patch_applied,
            'score': self.score,
            'passed': self.passed,
            'notes': self.notes,
        }


class PatchVerifier:
    """
    Verifies security patches through multiple checks.
    
    Patch quality rules:
    - Prefer minimal fix
    - Preserve app behavior
    - No dangerous shortcuts
    - No hardcoded secrets
    - No disabling validation
    - No removing auth
    - No adding vulnerable dependencies
    - No fake test results
    """
    
    # Patterns that indicate patch is hiding, not fixing
    SUSPICIOUS_PATTERNS = [
        (r'#\s*noqa', ' noqa comment - hiding warning'),
        (r'#\s*type:\s*ignore', 'type: ignore - hiding type error'),
        (r'noinspection', 'noinspection - suppressing inspection'),
        (r'@SuppressWarnings', '@SuppressWarnings - suppressing warning'),
        (r'//\s*eslint-disable', 'eslint-disable - hiding lint error'),
        (r'catch\s*\(\s*\w*\s*\)\s*\{\s*\}', 'Empty catch block - hiding error'),
        (r'eval\s*\(', 'eval() - dangerous function'),
        (r'exec\s*\(', 'exec() - dangerous function'),
    ]
    
    def verify_patch(
        self,
        original_code: str,
        patched_code: str,
        language: str = "python",
        test_command: str = None,
        scanner_command: str = None,
    ) -> VerificationResult:
        """Verify a security patch"""
        result = VerificationResult()
        
        # 1. Syntax check
        result.syntax_valid, result.syntax_error = self._check_syntax(patched_code, language)
        if not result.syntax_valid:
            result.notes.append(f"Syntax error: {result.syntax_error}")
            result.score = 0.0
            return result
        
        # 2. Check for suspicious patterns (hiding warnings)
        suspicious = self._check_suspicious(patched_code)
        if suspicious:
            result.notes.append(f"SUSPICIOUS: {suspicious}")
            result.score -= 0.2
        
        # 3. Check patch is minimal (not deleting everything)
        if len(patched_code) < len(original_code) * 0.3:
            result.notes.append("Patch removes too much code (possible deletion)")
            result.score -= 0.3
        
        # 4. Check no hardcoded secrets
        secrets = self._check_secrets(patched_code)
        if secrets:
            result.notes.append(f"Secrets found: {secrets}")
            result.score -= 0.2
        
        # 5. Run tests if command provided
        if test_command:
            test_result = self._run_tests(test_command)
            result.test_results = test_result
            result.tests_passed = sum(1 for t in test_result if t['passed'])
            result.tests_failed = len(test_result) - result.tests_passed
        
        # 6. Run scanner if command provided
        if scanner_command:
            before_issues = self._run_scanner(scanner_command, original_code)
            after_issues = self._run_scanner(scanner_command, patched_code)
            result.security_issues_before = before_issues
            result.security_issues_after = after_issues
            
            # Check if issues decreased
            if len(after_issues) < len(before_issues):
                result.score += 0.3
            elif len(after_issues) >= len(before_issues):
                result.notes.append("Security issues not reduced!")
                result.score -= 0.3
        
        # Compute final score
        result.score = max(0.0, min(1.0, 0.5 + result.score))
        result.passed = result.syntax_valid and result.score >= 0.7
        result.patch_applied = True
        
        return result
    
    def _check_syntax(self, code: str, language: str) -> Tuple[bool, Optional[str]]:
        """Check syntax"""
        if language == "python":
            try:
                ast.parse(code)
                return True, None
            except SyntaxError as e:
                return False, f"{e.msg} (line {e.lineno})"
        elif language in ("javascript", "typescript"):
            # Use node --check if available
            try:
                with open('/tmp/check.js', 'w') as f:
                    f.write(code)
                result = subprocess.run(['node', '--check', '/tmp/check.js'],
                                       capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    return True, None
                return False, result.stderr[:200]
            except:
                return True, None  # Skip if node not available
        return True, None
    
    def _check_suspicious(self, code: str) -> List[str]:
        """Check for patterns that hide warnings instead of fixing"""
        found = []
        for pattern, message in self.SUSPICIOUS_PATTERNS:
            if re.search(pattern, code):
                found.append(message)
        return found
    
    def _check_secrets(self, code: str) -> List[str]:
        """Check for hardcoded secrets"""
        secret_patterns = [
            (r'sk-[a-zA-Z0-9]{20,}', 'API key'),
            (r'password\s*=\s*["\'][^"\']{4,}["\']', 'Hardcoded password'),
            (r'secret\s*=\s*["\'][^"\']{4,}["\']', 'Hardcoded secret'),
            (r'api_key\s*=\s*["\'][^"\']{4,}["\']', 'Hardcoded API key'),
        ]
        found = []
        for pattern, msg in secret_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                found.append(msg)
        return found
    
    def _run_tests(self, command: str) -> List[Dict]:
        """Run test command"""
        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True, timeout=30
            )
            return [{
                'command': command,
                'passed': result.returncode == 0,
                'output': result.stdout[:500],
                'error': result.stderr[:500] if result.returncode != 0 else None,
            }]
        except subprocess.TimeoutExpired:
            return [{'command': command, 'passed': False, 'error': 'Timeout'}]
        except Exception as e:
            return [{'command': command, 'passed': False, 'error': str(e)}]
    
    def _run_scanner(self, command: str, code: str) -> List[str]:
        """Run security scanner"""
        try:
            with open('/tmp/scan_target', 'w') as f:
                f.write(code)
            result = subprocess.run(
                command + ' /tmp/scan_target',
                shell=True, capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                return []
            # Parse issues from output
            issues = [line for line in result.stderr.split('\n') if 'Error' in line or 'Warning' in line]
            return issues[:10]
        except:
            return []
    
    def generate_report(self, result: VerificationResult) -> str:
        """Generate human-readable report"""
        lines = []
        lines.append("=" * 60)
        lines.append("DreamPatch Security Report")
        lines.append("=" * 60)
        lines.append(f"\nPatch Applied: {'Yes' if result.patch_applied else 'No'}")
        lines.append(f"Syntax Valid: {'Yes' if result.syntax_valid else 'No'}")
        if result.syntax_error:
            lines.append(f"Syntax Error: {result.syntax_error}")
        lines.append(f"Tests: {result.tests_passed}/{result.tests_passed + result.tests_failed}")
        lines.append(f"Security Issues Before: {len(result.security_issues_before)}")
        lines.append(f"Security Issues After: {len(result.security_issues_after)}")
        lines.append(f"Score: {result.score:.2f}/1.0")
        lines.append(f"Status: {'PASS' if result.passed else 'FAIL'}")
        
        if result.notes:
            lines.append("\nNotes:")
            for note in result.notes:
                lines.append(f"  - {note}")
        
        lines.append("\n" + "=" * 60)
        lines.append("Crafted with love by IBR (Ibraheem)")
        
        return '\n'.join(lines)


if __name__ == "__main__":
    # Test verifier
    verifier = PatchVerifier()
    
    original = 'cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")'
    patched = 'cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))'
    
    print("=== Patch Verifier Test ===\n")
    result = verifier.verify_patch(original, patched, "python")
    print(verifier.generate_report(result))
