"""
id: SAFE-004
type: secure
vulnerable: false
cwe: 22
owasp: A01:2025 Broken Access Control
severity: none
bug_lines: []
attack_scenario: Looks like a file path built from user input, but the resolved path must stay inside REPORT_DIR so ../ is rejected
remediation: None needed; realpath plus a commonpath check confines reads to REPORT_DIR
source: Written by us, fixed twin of SEED-004
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
    base = os.path.realpath(REPORT_DIR)
    path = os.path.realpath(os.path.join(base, name))
    if os.path.commonpath([base, path]) != base:
        abort(404)
    if not os.path.isfile(path):
        abort(404)
    with open(path, encoding="utf-8") as handle:
        content = handle.read()
    return content, 200, {"Content-Type": "text/plain"}
