# Comparison of AI Security Analysis Agents

## Description

- **AI Security Analysis Agents** are tools that automatically detect
  vulnerabilities, suggest security hardening measures, and validate compliance with
  security policies with minimal human intervention
- They range from vulnerability scanners with AI-enhanced analysis (e.g., `Snyk` with
  LLM backend, `Checkmarx`, `Semgrep` with security rules) to full autonomous
  security reasoning agents (e.g., `CodeQL` with LLM interpretation,
  `GitHub Advanced Security` features)
- Key capabilities include identifying OWASP Top 10 vulnerabilities (injection,
  broken authentication, XSS, CSRF, insecure deserialization, weak crypto, access
  control flaws), detecting supply-chain risks and dependency vulnerabilities,
  suggesting secure API alternatives and hardening patterns, validating access
  controls and data flow, explaining vulnerability impact and attack scenarios, and
  recommending remediation steps
- Agents differ in false positive rates (flagging non-vulnerabilities as bugs: erodes
  trust), coverage of vulnerability classes (broad vs. specialized), ability to trace
  taint flow and data dependencies across files and modules, explainability of
  findings (why is this a vulnerability?), and practical integration with developer
  workflows (IDE, CI/CD, GitHub)
- This project teaches students critical thinking about security claims in AI
  tools—false negatives can be catastrophic (real vulnerability missed), and false
  positives create alert fatigue (developers ignore tool)

## Comparison of Security Analysis Agents

| Type            | Name                   | Description                                                                        | Website                                        | Strength                   |
| --------------- | ---------------------- | ---------------------------------------------------------------------------------- | ---------------------------------------------- | -------------------------- |
| Cloud scanner   | Snyk                   | AI-powered vulnerability scanner for code and dependencies with LLM explanations   | https://snyk.io                                | Fast, developer-friendly   |
| Static analysis | Semgrep                | Pattern-matching security rules with LLM-enhanced explanations and fix suggestions | https://semgrep.dev                            | Customizable, low false +  |
| Enterprise      | Checkmarx              | Static code analysis for OWASP Top 10, CWE, SANS 25 vulnerabilities                | https://checkmarx.com                          | Enterprise, comprehensive  |
| CodeQL + LLM    | GitHub Code Scanning   | GitHub-integrated security scanning with LLM explanations of findings              | https://github.com/advanced-security           | GitHub-native, free tier   |
| Specialized     | Fortify SCA            | Software composition analysis with vulnerability database and LLM explanations     | https://www.microfocus.com/fortify             | Legacy systems, compliance |
| Open-source     | OWASP Dependency-Check | Detects known vulnerabilities in dependencies; enhanced with LLM explanations      | https://owasp.org/www-project-dependency-check | Free, open-source          |

## Vulnerability Detection Capabilities

| Level                       | Capability                       | Example behaviors                                    |
| --------------------------- | -------------------------------- | ---------------------------------------------------- |
| L0 -- Flag known CVEs       | Identify known vulnerabilities   | "lodash < 4.17.21 has CVE-2021-23337"                |
| L1 -- Detect patterns       | Find OWASP Top 10 patterns       | "SQL injection risk: user input in query"            |
| L2 -- Trace data flow       | Track taint across files         | "Untrusted input from API used in SQL query at L42"  |
| L3 -- Explain & suggest fix | Propose remediation              | "Use parameterized queries instead of string concat" |
| L4 -- Autonomous hardening  | Trace & auto-fix vulnerabilities | Apply security patch, verify fix, run tests          |

## Project Objective

Design a controlled empirical study that benchmarks at least three AI security
analysis agents on a curated dataset of deliberately vulnerable and secure code
examples. The project aims to answer: _Which agents detect the most real
vulnerabilities, avoid false positives, provide clear explanations, and suggest
working fixes?_ Students will create a security benchmark (intentional
vulnerabilities + real CVEs + secure patterns), apply each agent to detect
vulnerabilities, and systematically compare: true positive rate (real vulnerabilities
caught), false positive rate (false alarms), vulnerability classification accuracy
(OWASP category), explanation quality, and remediation correctness

## Tasks

- **Security Benchmark Creation**: Curate or create 40–50 code snippets covering:
  - 15–20 intentional vulnerabilities (seeded bugs following OWASP Top 10)
  - 10–15 real CVEs from CVE database with proof-of-concept code
  - 15–20 secure code patterns (common mistakes but actually safe)
  - Document: vulnerability type (OWASP category), attack scenario, severity
    (critical/high/medium/low), correct remediation
- **Agent Setup & Execution**: Install and configure at least three security agents
  (e.g., Snyk, Semgrep, GitHub Code Scanning); run each agent on all benchmark
  snippets; record: findings reported, severity assigned, confidence score, and
  explanation provided
- **Vulnerability Detection Accuracy**: For each reported finding, measure: (a) true
  positive (is it a real vulnerability?), (b) false positive (is the code actually
  safe?), (c) missed vulnerability (did agent miss a real bug?), (d) correct OWASP
  classification
- **False Positive Analysis**: Measure false positive rate (FP / (FP + TP)); identify
  which vulnerability classes have highest false alarm rate; document examples of
  false positives to understand agent confusion patterns
- **Severity Rating Accuracy**: Compare agent-assigned severity (critical/high/
  medium/low) with consensus severity from security experts; measure: (a) is the
  rating appropriate for the vulnerability?, (b) does agent over/under-estimate risk?
- **Explanation Quality**: Extract agent's explanation of each vulnerability; score
  on: (a) technical accuracy (does it explain the attack?), (b) clarity for
  developers, (c) completeness (covers impact + attack scenario), (d) usefulness for
  remediation
- **Remediation Suggestion Correctness**: For each vulnerability, extract agent's fix
  suggestion; measure: (a) does suggested fix eliminate the vulnerability? (b) are
  there side effects or performance regression? (c) is the fix idiomatic and
  maintainable?
- **Data Flow Tracing**: Test multi-file vulnerabilities where untrusted input flows
  from API → business logic → database; measure which agents correctly trace taint
  across module boundaries
- **Comparative Scorecard**: Build a rubric weighing: true positive rate, false
  positive rate, severity accuracy, explanation quality, and fix correctness; rank
  agents and identify specialization (e.g., which agent best detects injection vs
  auth flaws)

## Vulnerability Benchmark Categories

### OWASP Top 10 (2021)

1. **Broken Access Control** (vertical/horizontal privilege escalation)

   ```python
   # Vulnerable: No permission check
   def delete_user(user_id):
       database.delete_user(user_id)

   # Secure: Check caller has permission
   def delete_user(user_id):
       if not current_user.is_admin():
           raise PermissionError()
       database.delete_user(user_id)
   ```

2. **Cryptographic Failures** (weak hashing, bad RNG, hardcoded keys)

   ```python
   # Vulnerable: Weak hash
   import hashlib
   password_hash = hashlib.md5(password).hexdigest()

   # Secure: Strong hash with salt
   import bcrypt
   password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
   ```

3. **Injection** (SQL, OS command, LDAP)

   ```python
   # Vulnerable: String concatenation
   query = f"SELECT * FROM users WHERE name = '{user_input}'"
   database.execute(query)

   # Secure: Parameterized query
   query = "SELECT * FROM users WHERE name = ?"
   database.execute(query, [user_input])
   ```

4. **Insecure Design** (missing security controls, inadequate threat modeling)
5. **Security Misconfiguration** (default credentials, debug mode, exposed configs)
6. **Vulnerable Components** (outdated dependencies, known CVEs)
7. **Authentication Failures** (weak password, session fixation, no MFA)
8. **Data Integrity Failures** (insecure deserialization, unsigned tokens)
9. **Logging & Monitoring Failures** (not logging security events)
10. **SSRF** (Server-Side Request Forgery)

### Supply Chain Risks

- Outdated dependencies with known CVEs
- Transitive dependencies pulling malicious code
- Unverified package sources

### Crypto Flaws

- Hardcoded encryption keys
- Weak random number generation
- Using deprecated algorithms (MD5, SHA1)
- Insecure key management

## Sample Vulnerability Scenarios

### SQL Injection

```python
def find_user_by_email(email: str):
    # Vulnerable
    query = f"SELECT * FROM users WHERE email = '{email}'"
    return database.execute(query)

def find_user_by_email_secure(email: str):
    # Secure
    query = "SELECT * FROM users WHERE email = ?"
    return database.execute(query, [email])
```

### Cross-Site Scripting (XSS)

```python
# Vulnerable: Unescaped user input in HTML
html = f"<h1>Welcome {user_input}</h1>"

# Secure: HTML escape
from html import escape
html = f"<h1>Welcome {escape(user_input)}</h1>"
```

### Insecure Deserialization

```python
import pickle
# Vulnerable: Arbitrary code execution via pickle
data = pickle.loads(user_input)

# Secure: Use json.loads with restricted types
import json
data = json.loads(user_input)
```

### Path Traversal

```python
import os
# Vulnerable: No path validation
file_path = os.path.join("/var/files", user_input)
with open(file_path, 'r') as f:
    return f.read()

# Secure: Validate and normalize path
from pathlib import Path
base_dir = Path("/var/files").resolve()
file_path = (base_dir / user_input).resolve()
if base_dir not in file_path.parents:
    raise ValueError("Invalid path")
```

## Bonus Ideas

- **Obfuscation Robustness**: Intentionally obfuscate vulnerabilities (split into
  multiple statements, indirect data flow) and measure which agents still detect
  them; test AI robustness vs. adversarial vulnerability hiding
- **Real CVE Dataset**: Source real vulnerabilities from CVE database
  (https://cve.mitre.org) with proof-of-concept code; measure which agents correctly
  identify published CVEs
- **Performance Regression**: Measure wall-clock time and API cost to analyze
  codebases of different sizes (1k, 10k, 100k LOC); compare agent efficiency
- **Fix Verification**: For each remediation suggested by an agent, verify: (a) does
  fix eliminate vulnerability? (b) are there regression tests in the original
  codebase that would pass with the fix? (c) does fix introduce performance issues?
- **Compliance Checking**: Test whether agents can validate compliance with security
  frameworks (OWASP Top 10, CWE Top 25, NIST Cybersecurity Framework)
- **Supply Chain Analysis**: Create a project with intentionally outdated
  dependencies (with known CVEs); measure whether agents detect and suggest upgrades
- **Taint Analysis**: Create multi-file vulnerabilities where untrusted input flows
  through multiple modules; measure which agents successfully trace taint across
  boundaries
- **False Positive Investigation**: For each false positive, investigate why the
  agent flagged it; identify systematic patterns (e.g., agent overly cautious about
  string operations)
- **Developer Trust**: Survey developers on which agent explanations are most
  trusted; correlate trust with accuracy metrics
- **Cost Analysis**: Compare API costs for security scanning; calculate
  cost-per-finding and cost-per-vulnerability-fixed

## Useful Resources

- **Vulnerability Datasets**:
  - CVE Mitre Database: https://cve.mitre.org
  - CWE Top 25: https://cwe.mitre.org/top25
  - OWASP Top 10: https://owasp.org/www-project-top-ten
  - Vulnerable Code Examples:
    https://github.com/payloadbox/sql-injection-payload-list
- **Security Analysis Tools**:
  - Snyk Documentation: https://docs.snyk.io
  - Semgrep Documentation: https://semgrep.dev/docs
  - GitHub Code Scanning: https://docs.github.com/en/code-security/code-scanning
  - OWASP Dependency-Check: https://owasp.org/www-project-dependency-check
- **Testing & Verification**:
  - Pytest: https://docs.pytest.org
  - Security testing frameworks:
    https://github.com/msabegun/awesome-api-security-testing
  - Bandit (Python security linter): https://bandit.readthedocs.io
- **References**:
  - _The Web Application Hacker's Handbook_ by Stuttard & Pinto
  - OWASP Testing Guide: https://owasp.org/www-project-web-security-testing-guide
  - PortSwigger Web Security Academy: https://portswigger.net/web-security
- **Agent APIs**:
  - Snyk API: https://snyk.io/docs/api
  - GitHub GraphQL API: https://docs.github.com/en/graphql
  - Semgrep API: https://semgrep.dev/api