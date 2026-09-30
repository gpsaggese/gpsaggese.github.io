# Comparison of AI Security Analysis Agents

## Status

- **Status:**: draft
- **Complete Specs:**: 80%

## Core Idea

- **AI Security Analysis Agents** are tools that automatically:
  - Detect vulnerabilities
  - Suggest security hardening measures
  - Validate compliance with security policies
  - Work with minimal human intervention
- Agents range from AI-enhanced scanners to full autonomous security reasoning agents
  - Vulnerability scanners with AI-enhanced analysis
    - E.g., `Snyk` with LLM backend, `Checkmarx`, `Semgrep` with security rules
  - Autonomous security reasoning agents
    - E.g., `CodeQL` with LLM interpretation, `GitHub Advanced Security` features
- Key capabilities:
  - Identify OWASP Top 10 vulnerabilities
    - Injection
    - Broken authentication
    - XSS
    - CSRF
    - Insecure deserialization
    - Weak crypto
    - Access control flaws
  - Detect supply-chain risks and dependency vulnerabilities
  - Suggest secure API alternatives and hardening patterns
  - Validate access controls and data flow
  - Explain vulnerability impact and attack scenarios
  - Recommend remediation steps
- Agents differ in:
  - False positive rates: flagging non-vulnerabilities as bugs erodes trust
  - Coverage of vulnerability classes (broad vs. specialized)
  - Ability to trace taint flow and data dependencies across files and modules
  - Explainability of findings: why is this a vulnerability?
  - Practical integration with developer workflows (IDE, CI/CD, GitHub)
- Critical thinking about security claims in AI tools is needed
  - False negatives can be catastrophic: a real vulnerability is missed
  - False positives create alert fatigue: developers ignore the tool
- Project objective: design a controlled empirical study that benchmarks at least
  three AI security analysis agents on a curated dataset of deliberately vulnerable
  and secure code examples
  - Create a security benchmark
    - Intentional vulnerabilities
    - Real CVEs
    - Secure patterns
  - Apply each agent to detect vulnerabilities
  - Systematically compare:
    - True positive rate (real vulnerabilities caught)
    - False positive rate (false alarms)
    - Vulnerability classification accuracy (OWASP category)
    - Explanation quality
    - Remediation correctness

## Formalization

### Comparison of Security Analysis Agents

| Type            | Name                   | Description                                                                        | Website                                        | Strength                   |
| --------------- | ---------------------- | ---------------------------------------------------------------------------------- | ---------------------------------------------- | -------------------------- |
| Cloud scanner   | Snyk                   | AI-powered vulnerability scanner for code and dependencies with LLM explanations   | https://snyk.io                                | Fast, developer-friendly   |
| Static analysis | Semgrep                | Pattern-matching security rules with LLM-enhanced explanations and fix suggestions | https://semgrep.dev                            | Customizable, low false +  |
| Enterprise      | Checkmarx              | Static code analysis for OWASP Top 10, CWE, SANS 25 vulnerabilities                | https://checkmarx.com                          | Enterprise, comprehensive  |
| CodeQL + LLM    | GitHub Code Scanning   | GitHub-integrated security scanning with LLM explanations of findings              | https://github.com/advanced-security           | GitHub-native, free tier   |
| Specialized     | Fortify SCA            | Software composition analysis with vulnerability database and LLM explanations     | https://www.microfocus.com/fortify             | Legacy systems, compliance |
| Open-source     | OWASP Dependency-Check | Detects known vulnerabilities in dependencies; enhanced with LLM explanations      | https://owasp.org/www-project-dependency-check | Free, open-source          |

### Vulnerability Detection Capabilities

| Level                         | Capability                         | Example behaviors                                    |
| ----------------------------- | ---------------------------------- | ---------------------------------------------------- |
| L0 -- Flag known CVEs         | Identify known vulnerabilities     | "lodash < 4.17.21 has CVE-2021-23337"                |
| L1 -- Detect patterns         | Find OWASP Top 10 patterns         | "SQL injection risk: user input in query"            |
| L2 -- Trace data flow         | Track taint across files           | "Untrusted input from API used in SQL query at L42"  |
| L3 -- Explain and suggest fix | Propose remediation                | "Use parameterized queries instead of string concat" |
| L4 -- Autonomous hardening    | Trace and auto-fix vulnerabilities | Apply security patch, verify fix, run tests          |

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
9. **Logging and Monitoring Failures** (not logging security events)
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

## Key Examples

- **SQL Injection**:
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

- **Cross-Site Scripting (XSS)**:
  ```python
  # Vulnerable: Unescaped user input in HTML
  html = f"<h1>Welcome {user_input}</h1>"

  # Secure: HTML escape
  from html import escape
  html = f"<h1>Welcome {escape(user_input)}</h1>"
  ```

- **Insecure Deserialization**:
  ```python
  import pickle
  # Vulnerable: Arbitrary code execution via pickle
  data = pickle.loads(user_input)

  # Secure: Use json.loads with restricted types
  import json
  data = json.loads(user_input)
  ```

- **Path Traversal**:
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

## Questions

1. _Which agents detect the most real vulnerabilities, avoid false positives,
   provide clear explanations, and suggest working fixes?_
2. What would a fair comparison look like? Precision and recall against
   seeded vulnerabilities with known ground truth are needed, since an agent
   that flags every line has perfect recall and is useless.
3. If an agent reliably reaches autonomous hardening (L4), does human security
   review become a verification step, and how often does its patch introduce a
   new vulnerability?

## Research Topics

- **Obfuscation Robustness**: intentionally obfuscate vulnerabilities
  - E.g., split into multiple statements, indirect data flow
  - Measure which agents still detect them
  - Test AI robustness vs. adversarial vulnerability hiding
- **Real CVE Dataset**: source real vulnerabilities from the CVE database
  (https://cve.mitre.org) with proof-of-concept code
  - Measure which agents correctly identify published CVEs
- **Performance Regression**: measure wall-clock time and API cost to analyze
  codebases of different sizes (1k, 10k, 100k LOC)
  - Compare agent efficiency
- **Fix Verification**: for each remediation suggested by an agent, verify:
  - Does the fix eliminate the vulnerability?
  - Would regression tests in the original codebase pass with the fix?
  - Does the fix introduce performance issues?
- **Compliance Checking**: test whether agents can validate compliance with
  security frameworks
  - OWASP Top 10
  - CWE Top 25
  - NIST Cybersecurity Framework
- **Supply Chain Analysis**: create a project with intentionally outdated
  dependencies (with known CVEs)
  - Measure whether agents detect them and suggest upgrades
- **Taint Analysis**: create multi-file vulnerabilities where untrusted input flows
  through multiple modules
  - Measure which agents successfully trace taint across boundaries
- **False Positive Investigation**: for each false positive, investigate why the
  agent flagged it
  - Identify systematic patterns
  - E.g., agent overly cautious about string operations
- **Developer Trust**: survey developers on which agent explanations are most
  trusted
  - Correlate trust with accuracy metrics
- **Cost Analysis**: compare API costs for security scanning
  - Calculate cost-per-finding
  - Calculate cost-per-vulnerability-fixed

## Next steps

- [ ] Look for related research (what has already been done)
- [ ] Finalize the implementation plan
- [ ] GP to review / approve the plan
- [ ] Hack a quick end-to-end prototype (e.g., in 1-2 days) to show that you
      understood the problem and can make progress
- [ ] Break the problem down in phases and milestones
- [ ] Execute one step at the time

## Implementation plan

- Milestone 1: create the security benchmark
  - **Security Benchmark Creation**: curate or create 40-50 code snippets covering:
    - 15-20 intentional vulnerabilities (seeded bugs following OWASP Top 10)
    - 10-15 real CVEs from the CVE database with proof-of-concept code
    - 15-20 secure code patterns (common mistakes but actually safe)
  - Document for each snippet:
    - Vulnerability type (OWASP category)
    - Attack scenario
    - Severity (critical/high/medium/low)
    - Correct remediation

- Milestone 2: set up and run the agents
  - **Agent Setup and Execution**: install and configure at least three security
    agents
    - E.g., Snyk, Semgrep, GitHub Code Scanning
  - Run each agent on all benchmark snippets
  - Record:
    - Findings reported
    - Severity assigned
    - Confidence score
    - Explanation provided

- Milestone 3: evaluate detection quality
  - **Vulnerability Detection Accuracy**: for each reported finding, measure:
    - True positive: is it a real vulnerability?
    - False positive: is the code actually safe?
    - Missed vulnerability: did the agent miss a real bug?
    - Correct OWASP classification
  - **False Positive Analysis**: measure false positive rate as $FP / (FP + TP)$
    - Identify which vulnerability classes have the highest false alarm rate
    - Document examples of false positives to understand agent confusion patterns
  - **Severity Rating Accuracy**: compare agent-assigned severity
    (critical/high/medium/low) with consensus severity from security experts
    - Is the rating appropriate for the vulnerability?
    - Does the agent over/under-estimate risk?
  - **Data Flow Tracing**: test multi-file vulnerabilities where untrusted input
    flows from API -> business logic -> database
    - Measure which agents correctly trace taint across module boundaries

- Milestone 4: evaluate explanations and fixes, then build the scorecard
  - **Explanation Quality**: extract the agent's explanation of each vulnerability
    and score on:
    - Technical accuracy: does it explain the attack?
    - Clarity for developers
    - Completeness: covers impact and attack scenario
    - Usefulness for remediation
  - **Remediation Suggestion Correctness**: for each vulnerability, extract the
    agent's fix suggestion and measure:
    - Does the suggested fix eliminate the vulnerability?
    - Are there side effects or performance regression?
    - Is the fix idiomatic and maintainable?
  - **Comparative Scorecard**: build a rubric weighing:
    - True positive rate
    - False positive rate
    - Severity accuracy
    - Explanation quality
    - Fix correctness
  - Rank agents and identify specialization
    - E.g., which agent best detects injection vs auth flaws

## References

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
- **Testing and Verification**:
  - Pytest: https://docs.pytest.org
  - Security testing frameworks:
    https://github.com/msabegun/awesome-api-security-testing
  - Bandit (Python security linter): https://bandit.readthedocs.io
- **Books and Guides**:
  - _The Web Application Hacker's Handbook_ by Stuttard and Pinto
  - OWASP Testing Guide: https://owasp.org/www-project-web-security-testing-guide
  - PortSwigger Web Security Academy: https://portswigger.net/web-security
- **Agent APIs**:
  - Snyk API: https://snyk.io/docs/api
  - GitHub GraphQL API: https://docs.github.com/en/graphql
  - Semgrep API: https://semgrep.dev/api
