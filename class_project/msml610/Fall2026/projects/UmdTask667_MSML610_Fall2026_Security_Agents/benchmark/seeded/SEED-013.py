"""
id: SEED-013
type: seeded
vulnerable: true
cwe: 78
owasp: A05:2025 Injection
severity: critical
bug_lines: [32, 33]
attack_scenario: Attacker sends host=127.0.0.1;cat /etc/passwd and the shell runs the second command
remediation: Validate the host with ipaddress.ip_address and call subprocess.run with an argument list and no shell
source: Written by us, based on the CWE-78 page
cve_id:
fix_commit:
pair_id: P-005
labelled_by: draft
reviewed_by:
first_pass_agree:
"""

import subprocess

from flask import Flask, jsonify, request

app = Flask(__name__)


@app.route("/tools/ping")
def ping_host():
    host = request.args.get("host", "").strip()
    if not host:
        return jsonify({"error": "host is required"}), 400
    command = f"ping -c 1 {host}"  # {!LINE}
    completed = subprocess.run(  # {!LINE}
        command, shell=True, capture_output=True, text=True, timeout=10
    )
    return jsonify(
        {
            "host": host,
            "reachable": completed.returncode == 0,
            "output": completed.stdout,
        }
    )
