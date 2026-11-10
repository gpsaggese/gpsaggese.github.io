"""
id: SEED-004
type: seeded
vulnerable: true
cwe: 22
owasp: A01:2025 Broken Access Control
severity: high
bug_lines: [34]
attack_scenario: Attacker requests name=../config/secrets.env and reads files outside the reports folder
remediation: Resolve the path with os.path.realpath and reject it unless it stays inside REPORT_DIR
source: Written by us, based on the CWE-22 page and the project brief key example
cve_id:
fix_commit:
pair_id: P-004
labelled_by: draft
reviewed_by:
first_pass_agree:
"""

import os

from flask import Flask, abort, request

app = Flask(__name__)

REPORT_DIR = os.environ.get("REPORT_DIR", "/srv/app/reports")


@app.route("/reports/download")
def download_report():
    name = request.args.get("name", "")
    if not name.endswith(".txt") and not name.endswith(".env"):
        abort(400)
    path = os.path.join(REPORT_DIR, name)  # {!LINE}
    if not os.path.isfile(path):
        abort(404)
    with open(path, encoding="utf-8") as handle:
        content = handle.read()
    return content, 200, {"Content-Type": "text/plain"}
