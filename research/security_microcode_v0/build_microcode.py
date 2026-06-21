"""
Security Microcode V0 Builder
==============================
Generates the microcode dataset with S/K/F/V/O/P fields.

S = Source (where untrusted data enters)
K = Sink (where the dangerous operation happens)
F = Flow (how data travels from S to K)
V = Vulnerability (what can go wrong)
O = Patch Operator (the defensive transformation)
P = Proof Obligation (what must hold for the fix to be correct)

Starting categories (12):
1. SQL Injection
2. XSS
3. Path Traversal
4. Command Injection
5. Insecure Upload
6. SSRF
7. Unsafe Eval
8. Weak Password Hashing
9. Insecure CORS
10. Hardcoded Secret
11. Unsafe JWT Decode
12. Missing Authorization
"""
import json
import gzip
import hashlib
from pathlib import Path
from datetime import datetime

BASE = Path(__file__).resolve().parent.parent.parent
OUT_DIR = BASE / "research" / "security_microcode_v0"
OUT_FILE = OUT_DIR / "microcode_dataset.jsonl"


def make_id(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


MICROCODES = [
    # 1. SQL Injection
    {
        "id": "mc_sql_injection_python",
        "category": "sql_injection",
        "cwe": "CWE-89",
        "owasp": "A03:2021-Injection",
        "language": "python",
        "S": "HTTP request parameters, form data, URL query strings",
        "K": "Database cursor.execute() with string-interpolated SQL",
        "F": "request.args['id'] → string format → cursor.execute(sql)",
        "V": "Attacker injects SQL metacharacters via the parameter, altering the query grammar",
        "O": "Replace string interpolation with parameterized queries: cursor.execute(sql, params)",
        "P": "For all user inputs u, the parameterized query passes u as data (not code). The DB driver guarantees u cannot alter SQL grammar.",
        "detection_keywords": ["execute", "f\"", "format(", "%s" + " +", "cursor"],
        "vuln_pattern": "cursor.execute(f\"SELECT * FROM users WHERE id = {user_id}\")",
        "microcode": "cursor.execute(\"SELECT * FROM users WHERE id = %s\", (user_id,))",
        "rationale": "Parameterized queries separate code from data. The DB driver escapes the parameter automatically. This is non-negotiable for any SQL that touches user input.",
    },
    {
        "id": "mc_sql_injection_js",
        "category": "sql_injection",
        "cwe": "CWE-89",
        "owasp": "A03:2021-Injection",
        "language": "javascript",
        "S": "req.body, req.query, req.params",
        "K": "db.query() with template literal SQL",
        "F": "req.body.name → template literal → db.query(sql)",
        "V": "Attacker injects SQL via the request body",
        "O": "Use parameterized queries: db.query(sql, [params])",
        "P": "For all inputs u, db.query passes u via the parameter array, not string interpolation",
        "detection_keywords": ["query", "`", "${", "execute"],
        "vuln_pattern": "db.query(`SELECT * FROM users WHERE name = '${name}'`)",
        "microcode": "db.query('SELECT * FROM users WHERE name = $1', [name])",
        "rationale": "Parameterized queries are the only correct defense against SQL injection. Escaping is fragile and context-dependent.",
    },
    {
        "id": "mc_sql_injection_php",
        "category": "sql_injection",
        "cwe": "CWE-89",
        "owasp": "A03:2021-Injection",
        "language": "php",
        "S": "$_GET, $_POST, $_REQUEST",
        "K": "mysqli_query() with concatenated SQL",
        "F": "$_GET['id'] → string concat → mysqli_query(sql)",
        "V": "Attacker injects SQL via URL parameter",
        "O": "Use prepared statements: mysqli_prepare + bind_param",
        "P": "For all inputs u, mysqli_stmt_bind_param passes u as a parameter, not concatenated SQL",
        "detection_keywords": ["mysqli_query", "$_GET", "$_POST", "."],
        "vuln_pattern": "mysqli_query($conn, \"SELECT * FROM users WHERE id='\" . $_GET['id'] . \"'\")",
        "microcode": "$stmt = mysqli_prepare($conn, \"SELECT * FROM users WHERE id = ?\");\nmysqli_stmt_bind_param($stmt, 's', $_GET['id']);\nmysqli_stmt_execute($stmt);",
        "rationale": "Prepared statements are the PHP standard for SQL injection prevention. They work with all MySQLi/PDO drivers.",
    },

    # 2. XSS
    {
        "id": "mc_xss_python",
        "category": "xss",
        "cwe": "CWE-79",
        "owasp": "A03:2021-Injection",
        "language": "python",
        "S": "User input in forms, URLs, database",
        "K": "HTML response rendering",
        "F": "request.args['name'] → template → HTML response",
        "V": "Attacker injects <script> tags that execute in victim's browser",
        "O": "Use html.escape() or auto-escaping templates",
        "P": "For all inputs u, html.escape(u) contains no HTML metacharacters (<, >, &, ', \")",
        "detection_keywords": ["innerHTML", "render", "f\"", "format", "Response"],
        "vuln_pattern": "return f'<div>{user_input}</div>'",
        "microcode": "import html\nreturn f'<div>{html.escape(user_input)}</div>'",
        "rationale": "Context-aware output encoding is the primary defense against XSS. html.escape converts metacharacters to HTML entities.",
    },
    {
        "id": "mc_xss_js_dom",
        "category": "xss",
        "cwe": "CWE-79",
        "owasp": "A03:2021-Injection",
        "language": "javascript",
        "S": "URL parameters, postMessage, user input",
        "K": "element.innerHTML, document.write()",
        "F": "location.hash → DOM manipulation → innerHTML",
        "V": "Attacker injects HTML/script via URL fragment",
        "O": "Use textContent instead of innerHTML, or sanitize with DOMPurify",
        "P": "For all inputs u, element.textContent = u cannot execute scripts (textContent is parsed as plain text)",
        "detection_keywords": ["innerHTML", "document.write", "outerHTML", "insertAdjacentHTML"],
        "vuln_pattern": "document.getElementById('output').innerHTML = userInput;",
        "microcode": "document.getElementById('output').textContent = userInput;",
        "rationale": "textContent never parses HTML, so it's inherently safe against XSS. Use innerHTML only with sanitized input.",
    },
    {
        "id": "mc_xss_php",
        "category": "xss",
        "cwe": "CWE-79",
        "owasp": "A03:2021-Injection",
        "language": "php",
        "S": "$_GET, $_POST, database",
        "K": "echo, print, printf",
        "F": "$_GET['name'] → echo → HTML output",
        "V": "Attacker injects script tags via URL",
        "O": "Use htmlspecialchars() with ENT_QUOTES and UTF-8",
        "P": "For all inputs u, htmlspecialchars(u, ENT_QUOTES, 'UTF-8') escapes all HTML metacharacters",
        "detection_keywords": ["echo", "print", "$_GET", "$_POST"],
        "vuln_pattern": "echo $_GET['name'];",
        "microcode": "echo htmlspecialchars($_GET['name'], ENT_QUOTES, 'UTF-8');",
        "rationale": "htmlspecialchars with ENT_QUOTES escapes both single and double quotes, preventing attribute injection and script injection.",
    },

    # 3. Path Traversal
    {
        "id": "mc_path_traversal_python",
        "category": "path_traversal",
        "cwe": "CWE-22",
        "owasp": "A01:2021-Broken Access Control",
        "language": "python",
        "S": "User-supplied filename, path parameter",
        "K": "open(), os.path.join(), send_file()",
        "F": "request.args['file'] → open(path) → file read",
        "V": "Attacker uses ../ to escape the intended directory",
        "O": "Validate realpath is within the base directory",
        "P": "For all inputs f, os.path.realpath(base/f) starts with os.path.realpath(base)",
        "detection_keywords": ["open", "send_file", "os.path", "filename"],
        "vuln_pattern": "open(f'uploads/{filename}').read()",
        "microcode": "import os\nbase = os.path.realpath('uploads/')\nsafe_path = os.path.realpath(os.path.join(base, filename))\nif not safe_path.startswith(base + os.sep):\n    raise ValueError('Path traversal detected')\nopen(safe_path).read()",
        "rationale": "Realpath resolves symlinks and ../ sequences. Comparing against the base directory ensures the path stays within bounds.",
    },

    # 4. Command Injection
    {
        "id": "mc_command_injection_python",
        "category": "command_injection",
        "cwe": "CWE-78",
        "owasp": "A03:2021-Injection",
        "language": "python",
        "S": "User input in command arguments",
        "K": "os.system(), subprocess with shell=True",
        "F": "request.args['host'] → os.system(cmd) → shell execution",
        "V": "Attacker injects shell metacharacters (;, |, &&, $())",
        "O": "Use subprocess.run with list args, shell=False",
        "P": "For all inputs u, subprocess.run([cmd, u]) passes u as a single argv element, bypassing shell parsing",
        "detection_keywords": ["os.system", "subprocess", "shell=True", "popen"],
        "vuln_pattern": "os.system(f'ping {host}')",
        "microcode": "import subprocess\nresult = subprocess.run(['ping', host], capture_output=True, text=True, timeout=30)",
        "rationale": "List-form subprocess args bypass the shell entirely. No metacharacter can alter the command structure.",
    },
    {
        "id": "mc_command_injection_js",
        "category": "command_injection",
        "cwe": "CWE-78",
        "owasp": "A03:2021-Injection",
        "language": "javascript",
        "S": "User input in Node.js command execution",
        "K": "child_process.exec()",
        "F": "req.body.cmd → exec(cmd) → shell",
        "V": "Attacker injects shell metacharacters",
        "O": "Use execFile() with array args",
        "P": "For all inputs u, execFile(cmd, [u]) passes u as a single argument, no shell parsing",
        "detection_keywords": ["exec", "child_process", "spawn"],
        "vuln_pattern": "child_process.exec(`ls ${userDir}`, callback);",
        "microcode": "child_process.execFile('ls', [userDir], callback);",
        "rationale": "execFile does not invoke a shell, so metacharacters have no special meaning.",
    },

    # 5. Insecure Upload
    {
        "id": "mc_insecure_upload_python",
        "category": "insecure_upload",
        "cwe": "CWE-434",
        "owasp": "A04:2021-Insecure Design",
        "language": "python",
        "S": "File upload form data",
        "K": "file.save() with user-controlled filename",
        "F": "request.files['upload'] → file.save(path) → filesystem",
        "V": "Attacker uploads .php/.py/.jsp file, achieving RCE",
        "O": "Validate extension, generate random filename, store outside webroot",
        "P": "For all uploads u: ext(u) ∈ ALLOWED_EXTENSIONS, filename = random, path ∉ webroot",
        "detection_keywords": ["file.save", "upload", "filename", "save_as"],
        "vuln_pattern": "file.save(f'uploads/{file.filename}')",
        "microcode": "import os, secrets\nALLOWED = {'png', 'jpg', 'jpeg', 'gif', 'pdf'}\nMAX_SIZE = 10 * 1024 * 1024\next = file.filename.rsplit('.', 1)[-1].lower()\nif ext not in ALLOWED: raise ValueError('Type not allowed')\nif len(file.read()) > MAX_SIZE: raise ValueError('Too large')\nfile.seek(0)\nsafe_name = secrets.token_hex(16) + '.' + ext\nfile.save(os.path.join('/var/uploads', safe_name))  # outside webroot",
        "rationale": "Extension allowlist prevents executable upload. Random filename prevents overwrite and path traversal. Storing outside webroot prevents direct execution.",
    },

    # 6. SSRF
    {
        "id": "mc_ssrf_python",
        "category": "ssrf",
        "cwe": "CWE-918",
        "owasp": "A10:2021-SSRF",
        "language": "python",
        "S": "User-supplied URL",
        "K": "requests.get(), urllib.urlopen()",
        "F": "request.args['url'] → requests.get(url) → server-side request",
        "V": "Attacker accesses internal services (169.254.169.254, localhost, file://)",
        "O": "Validate URL scheme, hostname, and block internal IPs",
        "P": "For all URLs u: scheme(u) ∈ {http, https}, hostname(u) ∉ internal_ranges, redirects disabled",
        "detection_keywords": ["requests.get", "urlopen", "url", "fetch"],
        "vuln_pattern": "import requests\nr = requests.get(user_url)",
        "microcode": "import requests\nfrom urllib.parse import urlparse\nimport ipaddress\n\nparsed = urlparse(user_url)\nif parsed.scheme not in ('http', 'https'): raise ValueError('Bad scheme')\ntry:\n    ip = ipaddress.ip_address(parsed.hostname)\n    if ip.is_private or ip.is_loopback or ip.is_link_local:\n        raise ValueError('Internal IP blocked')\nexcept ipaddress.AddressValueError:\n    pass  # hostname is a domain, not IP\nr = requests.get(user_url, timeout=10, allow_redirects=False, max_redirects=0)",
        "rationale": "SSRF prevention requires blocking internal IPs, validating scheme, and disabling redirects (which can bypass hostname checks).",
    },

    # 7. Unsafe Eval
    {
        "id": "mc_unsafe_eval_python",
        "category": "unsafe_eval",
        "cwe": "CWE-95",
        "owasp": "A03:2021-Injection",
        "language": "python",
        "S": "User input, serialized data",
        "K": "eval(), exec(), compile()",
        "F": "request.args['expr'] → eval(expr) → arbitrary code execution",
        "V": "Attacher executes arbitrary Python code",
        "O": "Use ast.literal_eval() for literals, or a safe parser",
        "P": "For all inputs u, ast.literal_eval(u) only accepts Python literals (str, int, float, list, dict, tuple, bool, None)",
        "detection_keywords": ["eval", "exec", "compile", "__import__"],
        "vuln_pattern": "result = eval(user_expression)",
        "microcode": "import ast\nresult = ast.literal_eval(user_expression)  # Only accepts literals",
        "rationale": "ast.literal_eval only parses Python literal structures. It cannot execute function calls or access attributes.",
    },
    {
        "id": "mc_unsafe_eval_js",
        "category": "unsafe_eval",
        "cwe": "CWE-95",
        "owasp": "A03:2021-Injection",
        "language": "javascript",
        "S": "User input, template strings",
        "K": "eval(), new Function(), setTimeout(string)",
        "F": "userInput → eval() → arbitrary JS execution",
        "V": "Attacker runs arbitrary JavaScript",
        "O": "Use JSON.parse() for data, or a sandboxed interpreter",
        "P": "For all inputs u, JSON.parse(u) only accepts JSON. No function calls or expressions are evaluated.",
        "detection_keywords": ["eval", "Function", "setTimeout", "setInterval"],
        "vuln_pattern": "eval(userInput);",
        "microcode": "JSON.parse(userInput);  // Safe: only parses JSON data",
        "rationale": "JSON.parse cannot execute code. It only accepts JSON syntax, which has no functions or expressions.",
    },

    # 8. Weak Password Hashing
    {
        "id": "mc_weak_hash_python",
        "category": "weak_crypto",
        "cwe": "CWE-327",
        "owasp": "A02:2021-Cryptographic Failures",
        "language": "python",
        "S": "User password at registration/login",
        "K": "hashlib.md5(), hashlib.sha1()",
        "F": "password → md5(password) → stored hash",
        "V": "MD5/SHA1 are fast, vulnerable to brute-force and rainbow tables",
        "O": "Use PBKDF2-HMAC-SHA256 with 100K+ iterations + random salt, or bcrypt/argon2",
        "P": "For all passwords p: hash = PBKDF2(p, salt, 100000). Time to crack ≥ 10^6 years on commodity hardware.",
        "detection_keywords": ["md5", "sha1", "hashlib", "hexdigest"],
        "vuln_pattern": "import hashlib\nhashed = hashlib.md5(password.encode()).hexdigest()",
        "microcode": "import hashlib, os\nsalt = os.urandom(32)\nhashed = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000).hex()\n# Store: salt + hashed\n# Or better: use argon2-cffi\n# from argon2 import PasswordHasher\n# ph = PasswordHasher()\n# hashed = ph.hash(password)",
        "rationale": "PBKDF2 with 100K iterations makes brute-force ~100,000x slower than MD5. Argon2id is even better (memory-hard).",
    },

    # 9. Insecure CORS
    {
        "id": "mc_insecure_cors",
        "category": "insecure_cors",
        "cwe": "CWE-942",
        "owasp": "A05:2021-Security Misconfiguration",
        "language": "python",
        "S": "HTTP Origin header",
        "K": "Access-Control-Allow-Origin response header",
        "F": "request.Origin → ACAO = '*' → any site can read responses",
        "V": "Attacker's website can make authenticated cross-origin requests",
        "O": "Allowlist specific origins, never use '*' with credentials",
        "P": "For all origins o: ACAO = o only if o ∈ ALLOWED_ORIGINS. ACAO ≠ '*' when credentials are allowed.",
        "detection_keywords": ["Access-Control-Allow-Origin", "CORS", " ACAO"],
        "vuln_pattern": "response.headers['Access-Control-Allow-Origin'] = '*'",
        "microcode": "ALLOWED_ORIGINS = {'https://example.com', 'https://app.example.com'}\norigin = request.headers.get('Origin')\nif origin in ALLOWED_ORIGINS:\n    response.headers['Access-Control-Allow-Origin'] = origin\n    response.headers['Vary'] = 'Origin'\n    response.headers['Access-Control-Allow-Credentials'] = 'true'\n# Never set ACAO='*' with Allow-Credentials: true",
        "rationale": "CORS allowlist ensures only trusted origins can make cross-origin requests. Reflecting the Origin header without validation is equivalent to '*'.",
    },

    # 10. Hardcoded Secret
    {
        "id": "mc_hardcoded_secret",
        "category": "hardcoded_secret",
        "cwe": "CWE-798",
        "owasp": "A07:2021-Identification & Auth Failures",
        "language": "python",
        "S": "Source code, config files",
        "K": "API_KEY = '...', PASSWORD = '...'",
        "F": "secret in code → committed to git → leaked",
        "V": "Anyone with repo access gets the secret",
        "O": "Use environment variables or a secret manager",
        "P": "For all secrets s: s ∈ env_vars OR s ∈ secret_manager. s ∉ source_code. s ∉ git_history.",
        "detection_keywords": ["API_KEY", "SECRET", "PASSWORD", "TOKEN", "sk-"],
        "vuln_pattern": "API_KEY = 'sk-1234567890abcdef'",
        "microcode": "import os\nAPI_KEY = os.environ.get('API_KEY')\nif not API_KEY:\n    raise RuntimeError('API_KEY environment variable not set')\n# Or use a secret manager:\n# from google.cloud import secretmanager\n# client = secretmanager.SecretManagerServiceClient()\n# response = client.access_secret_version(name='projects/.../secrets/api-key/versions/latest')\n# API_KEY = response.payload.data.decode('UTF-8')",
        "rationale": "Environment variables keep secrets out of source code. Secret managers add rotation, audit logging, and access control.",
    },

    # 11. Unsafe JWT Decode
    {
        "id": "mc_unsafe_jwt",
        "category": "unsafe_jwt",
        "cwe": "CWE-345",
        "owasp": "A02:2021-Cryptographic Failures",
        "language": "python",
        "S": "Authorization header, cookie",
        "K": "jwt.decode() without verification",
        "F": "token → jwt.decode(token, verify=False) → unverified claims",
        "V": "Attacker forges a JWT with admin claims, server accepts it",
        "O": "Always verify signature, specify algorithms explicitly, validate claims",
        "P": "For all tokens t: jwt.decode(t, key, algorithms=['HS256']) verifies signature. alg=none is rejected.",
        "detection_keywords": ["jwt", "decode", "verify=False", "token"],
        "vuln_pattern": "import jwt\ndata = jwt.decode(token, verify=False)",
        "microcode": "import jwt, os\nSECRET = os.environ.get('JWT_SECRET')\ntry:\n    data = jwt.decode(token, SECRET, algorithms=['HS256'])\nexcept jwt.InvalidTokenError:\n    raise ValueError('Invalid JWT')\n# Verify claims:\nif data['exp'] < time.time():\n    raise ValueError('Token expired')",
        "rationale": "Explicit algorithms list prevents the 'alg=none' attack. Always verify signature and expiry. Never accept unsigned tokens.",
    },

    # 12. Missing Authorization
    {
        "id": "mc_missing_authz",
        "category": "missing_authz",
        "cwe": "CWE-862",
        "owasp": "A01:2021-Broken Access Control",
        "language": "python",
        "S": "HTTP request to protected endpoint",
        "K": "Route handler without auth check",
        "F": "request → handler → data access (no auth check)",
        "V": "Any user can access any other user's data (IDOR)",
        "O": "Add authorization decorator, check ownership/role",
        "P": "For all requests r to protected resource res: authn(r) ∧ authz(r.user, res). Unauthorized → 403.",
        "detection_keywords": ["@app.route", "@route", "def get_", "def update_"],
        "vuln_pattern": "@app.route('/admin')\ndef admin():\n    return render_template('admin.html')",
        "microcode": "from functools import wraps\nfrom flask import abort, session\n\ndef require_role(role):\n    def decorator(f):\n        @wraps(f)\n        def decorated(*args, **kwargs):\n            user = session.get('user')\n            if not user or user.get('role') != role:\n                abort(403)\n            return f(*args, **kwargs)\n        return decorated\n    return decorator\n\n@app.route('/admin')\n@require_role('admin')\ndef admin():\n    return render_template('admin.html')",
        "rationale": "Authorization must be checked on every request to a protected resource. Decorators ensure no route is accidentally left unprotected.",
    },
]


def main():
    print(f"[{datetime.now():%H:%M:%S}] Security Microcode V0 Builder")
    print("=" * 70)
    print(f"Generating {len(MICROCODES)} microcodes...")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with OUT_FILE.open("w", encoding="utf-8") as f:
        for mc in MICROCODES:
            # Add composed_with field
            mc["composed_with"] = []
            # Link related microcodes
            for other in MICROCODES:
                if other["id"] != mc["id"] and other["category"] == mc["category"]:
                    mc["composed_with"].append(other["id"])
            f.write(json.dumps(mc, ensure_ascii=False) + "\n")

    size_kb = OUT_FILE.stat().st_size / 1024
    print(f"\n✅ Wrote {len(MICROCODES)} microcodes to {OUT_FILE}")
    print(f"   Size: {size_kb:.0f} KB")
    print(f"\nCategories covered:")
    cats = set(mc["category"] for mc in MICROCODES)
    for c in sorted(cats):
        count = sum(1 for mc in MICROCODES if mc["category"] == c)
        print(f"  {c}: {count} microcode(s)")
    print(f"\nLanguages covered:")
    langs = set(mc["language"] for mc in MICROCODES)
    for l in sorted(langs):
        count = sum(1 for mc in MICROCODES if mc["language"] == l)
        print(f"  {l}: {count} microcode(s)")


if __name__ == "__main__":
    main()
