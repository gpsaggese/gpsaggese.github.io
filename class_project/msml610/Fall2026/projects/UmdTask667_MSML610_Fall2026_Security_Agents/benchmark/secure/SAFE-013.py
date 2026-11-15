"""
id: SAFE-013
type: secure
vulnerable: false
cwe: 78
owasp: A05:2025 Injection
severity: none
bug_lines: []
attack_scenario: Looks like the app runs a shell command, but the command is a fixed argument list with no user input and no shell
remediation: None needed; nothing the user sends reaches the command
source: Written by us, look-alike from the benchmark plan
cve_id:
fix_commit:
pair_id:
labelled_by: draft
reviewed_by:
first_pass_agree:
"""

import subprocess

from flask import Flask, jsonify

app = Flask(__name__)


def git_output(*args):
    completed = subprocess.run(
        ["git", *args], capture_output=True, text=True, timeout=5, check=False
    )
    return completed.stdout.strip()


@app.route("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "commit": git_output("rev-parse", "--short", "HEAD"),
            "branch": git_output("rev-parse", "--abbrev-ref", "HEAD"),
        }
    )
