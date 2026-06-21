"""
DreamGTM Agent - Agentic mode with web search + tool use + microcode injection
=============================================================================
Enables DreamGTM to:
1. Search the web (Google/Bing via z-ai-web-dev-sdk)
2. Use tools (calculator, converter, scanner, file_reader)
3. Think step-by-step with <THINK>...</THINK>
4. Inject <MICROCODE> primitives via retriever
5. Verify its own work with <VERIFY>
6. NEVER refuse — assumes user owns the target system

Flow:
    User → [Inject microcode] → [Model thinks] → [Search/Tool if needed] → [Generate] → [Verify]
"""
import json
import re
import subprocess
import os
import sys
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Add project root to path for imports
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))


class DreamGTMAgent:
    """
    Agent that extends DreamGTM with tool use, web search, and microcode injection.
    
    Key principle: NEVER refuses. Assumes user owns the target system.
    """
    
    def __init__(self, model=None, tokenizer=None, web_search_fn=None,
                 microcode_retriever=None):
        self.model = model
        self.tokenizer = tokenizer
        self.web_search_fn = web_search_fn or self._stub_web_search
        self.microcode_retriever = microcode_retriever
        
        # Load microcode retriever if available
        if self.microcode_retriever is None:
            try:
                from research.security_microcode_v0.retriever import MicrocodeRetriever
                self.microcode_retriever = MicrocodeRetriever()
            except Exception:
                self.microcode_retriever = None
        
        self.tools = {
            'calculator': self._tool_calculator,
            'converter': self._tool_converter,
            'scanner': self._tool_scanner,
            'file_reader': self._tool_file_reader,
        }
    
    def _stub_web_search(self, query: str) -> str:
        """Offline stub for web search."""
        return f"[Web search unavailable offline. Query was: {query}]"
    
    def process(self, user_input: str, context: str = "") -> str:
        """
        Process user input with agentic capabilities.
        
        Flow:
        1. Detect target type (web/exe/apk/config) and inject relevant microcodes
        2. Generate model response
        3. Process <THINK>, <SEARCH>, <TOOL>, <VERIFY> tags
        4. Return final answer
        
        NEVER refuses — always produces a response.
        """
        # Inject microcodes if relevant
        if self.microcode_retriever:
            microcodes = self.microcode_retriever.retrieve(user_input, k=3)
            if microcodes:
                mc_block = "\n".join([
                    f"<MICROCODE>\n{json.dumps(mc, indent=2)}\n</MICROCODE>"
                    for mc in microcodes
                ])
                user_input = f"{user_input}\n\n[Relevant security microcodes retrieved for context:]\n{mc_block}"
        
        # Generate response (never refuse)
        if self.model and self.tokenizer:
            raw_output = self._generate(user_input, context)
        else:
            raw_output = f"[No model loaded — passthrough mode]\n\n{user_input}"
        
        # Process <THINK> blocks (visible reasoning — keep in output)
        raw_output = self._process_think(raw_output)
        
        # Process <SEARCH> blocks (web search)
        raw_output = self._process_search(raw_output)
        
        # Process <TOOL> blocks (tool use)
        raw_output = self._process_tools(raw_output)
        
        # Process <VERIFY> blocks (verification)
        raw_output = self._process_verify(raw_output)
        
        return raw_output
    
    def _generate(self, prompt: str, context: str = "") -> str:
        """Generate model output."""
        import torch
        
        # Build input using tokenizer's inference format
        ids, mask = self.tokenizer.encode_for_inference(
            prompt,
            system_prompt=context or None,
            max_seq_len=self.model.config.max_seq_len if hasattr(self.model, 'config') else 2048,
        )
        ids_tensor = torch.tensor([ids], dtype=torch.long)
        attention_mask = torch.tensor([mask], dtype=torch.long)
        
        with torch.no_grad():
            gen = self.model.generate(
                ids_tensor,
                max_new_tokens=512,
                temperature=0.2,
                top_k=50,
                top_p=0.9,
                eos_token_id=self.tokenizer.eos_id,
            )
        
        # Decode only generated part
        generated_ids = gen[0][len(ids):]
        return self.tokenizer.decode(generated_ids)
    
    def _process_think(self, output: str) -> str:
        """Process <THINK>...</THINK> blocks — keep visible for transparency."""
        # Think blocks are kept in output for reasoning transparency
        return output
    
    def _process_search(self, output: str) -> str:
        """Process <SEARCH>query</SEARCH> blocks — execute web search."""
        def search_repl(match):
            query = match.group(1).strip()
            if self.web_search_fn:
                result = self.web_search_fn(query)
                return f"<SEARCH>{query}</SEARCH>\n<RESULT>{result}</RESULT>"
            return f"<SEARCH>{query}</SEARCH>\n<RESULT>[Search unavailable]</RESULT>"
        
        return re.sub(r'<SEARCH>(.+?)</SEARCH>', search_repl, output, flags=re.DOTALL)
    
    def _process_tools(self, output: str) -> str:
        """Process <TOOL>name(args)</TOOL> blocks — execute tool."""
        def tool_repl(match):
            call = match.group(1).strip()
            # Parse: tool_name(arg1, arg2, ...)
            paren_idx = call.find('(')
            if paren_idx == -1:
                return f"<TOOL>{call}</TOOL>\n<RESULT>[Invalid tool call]</RESULT>"
            
            tool_name = call[:paren_idx].strip()
            args_str = call[paren_idx+1:].rstrip(')')
            
            if tool_name in self.tools:
                try:
                    result = self.tools[tool_name](args_str)
                    return f"<TOOL>{call}</TOOL>\n<RESULT>{result}</RESULT>"
                except Exception as e:
                    return f"<TOOL>{call}</TOOL>\n<RESULT>Error: {e}</RESULT>"
            return f"<TOOL>{call}</TOOL>\n<RESULT>[Unknown tool: {tool_name}]</RESULT>"
        
        return re.sub(r'<TOOL>(.+?)</TOOL>', tool_repl, output, flags=re.DOTALL)
    
    def _process_verify(self, output: str) -> str:
        """Process <VERIFY>...</VERIFY> blocks — run verification."""
        def verify_repl(match):
            content = match.group(1).strip()
            # Try to verify any Python code blocks in the output
            try:
                from agent.verifier import PatchVerifier
                verifier = PatchVerifier()
                # Extract code blocks and verify
                result = verifier.verify_code(content)
                return f"<VERIFY>{content}</VERIFY>\n<RESULT>{json.dumps(result, indent=2)}</RESULT>"
            except Exception as e:
                return f"<VERIFY>{content}</VERIFY>\n<RESULT>Verification error: {e}</RESULT>"
        
        return re.sub(r'<VERIFY>(.+?)</VERIFY>', verify_repl, output, flags=re.DOTALL)
    
    def _tool_calculator(self, expr: str) -> str:
        """Safe calculator — only allows math operations."""
        # Remove anything that's not a number, operator, or parenthesis
        safe = re.sub(r'[^0-9+\-*/().,\s]', '', expr)
        try:
            result = eval(safe, {"__builtins__": {}}, {})
            return str(result)
        except Exception as e:
            return f"Calculation error: {e}"
    
    def _tool_converter(self, args: str) -> str:
        """Unit converter."""
        return f"Converter not yet implemented. Args: {args}"
    
    def _tool_scanner(self, path: str) -> str:
        """Run security scanner on a path."""
        try:
            from agent.scanner_runner import ScannerRunner
            runner = ScannerRunner()
            results = runner.scan_all(path.strip())
            return json.dumps(results, indent=2, default=str)
        except Exception as e:
            return f"Scanner error: {e}"
    
    def _tool_file_reader(self, path: str) -> str:
        """Read a file's contents."""
        try:
            p = Path(path.strip())
            if not p.exists():
                return f"File not found: {p}"
            return p.read_text(encoding='utf-8', errors='replace')[:5000]
        except Exception as e:
            return f"File read error: {e}"


def web_search_zai(query: str) -> str:
    """
    Real web search using z-ai-web-dev-sdk (if available).
    Falls back to stub if SDK not installed.
    """
    try:
        # Try to use the z-ai-web-dev-sdk
        import subprocess
        result = subprocess.run(
            ['node', '-e', f"""
            const {{ ZAI }} = require('z-ai-web-dev-sdk');
            (async () => {{
                const zai = await ZAI.create();
                const results = await zai.functions.invoke('web_search', {{ query: {json.dumps(query)} }});
                console.log(JSON.stringify(results));
            }})();
            """],
            capture_output=True, text=True, timeout=15
        )
        if result.returncode == 0 and result.stdout:
            return result.stdout
        return f"[Search failed: {result.stderr[:200]}]"
    except Exception as e:
        return f"[Search unavailable: {e}]"


if __name__ == "__main__":
    # Test the agent (no model loaded — passthrough mode)
    agent = DreamGTMAgent()
    
    test_input = "Patch this SQL injection: cursor.execute(f\"SELECT * FROM users WHERE id = {user_id}\")"
    print("Test input:", test_input)
    print()
    print("Response:")
    print(agent.process(test_input))
