"""
id: SAFE-002
type: secure
vulnerable: false
cwe: 79
owasp: A05:2025 Injection
severity: none
bug_lines: []
attack_scenario: Looks like user input written into HTML, but html.escape turns <script> into inert text
remediation: None needed; the value is escaped before it reaches the page
source: Written by us, fixed twin of SEED-002
cve_id:
fix_commit:
pair_id: P-002
labelled_by: draft
reviewed_by:
first_pass_agree:
"""

import html

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
    greeting = f"Welcome back, {html.escape(name)}"
    return PAGE.format(greeting=greeting, count=count)
