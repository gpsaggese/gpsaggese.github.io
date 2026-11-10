"""
id: SEED-002
type: seeded
vulnerable: true
cwe: 79
owasp: A05:2025 Injection
severity: medium
bug_lines: [39]
attack_scenario: Attacker shares a link with name=<script>...</script>; the script runs in the victim's browser and can read the session
remediation: Escape the value before putting it in HTML, e.g., html.escape(name), or render with an autoescaping template
source: Written by us, based on the CWE-79 page and the project brief key example
cve_id:
fix_commit:
pair_id: P-002
labelled_by: draft
reviewed_by:
first_pass_agree:
"""

from flask import Flask, request

app = Flask(__name__)

PAGE = """<!doctype html>
<html>
  <head><title>Dashboard</title></head>
  <body>
    <h1>{greeting}</h1>
    <p>You have {count} unread messages.</p>
  </body>
</html>
"""


@app.route("/dashboard")
def dashboard():
    name = request.args.get("name", "guest")
    count = int(request.args.get("unread", "0"))
    greeting = f"Welcome back, {name}"  # {!LINE}
    return PAGE.format(greeting=greeting, count=count)
