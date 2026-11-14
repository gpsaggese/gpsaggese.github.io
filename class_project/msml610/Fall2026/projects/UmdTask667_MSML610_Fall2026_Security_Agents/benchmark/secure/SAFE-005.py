"""
id: SAFE-005
type: secure
vulnerable: false
cwe: 78
owasp: A05:2025 Injection
severity: none
bug_lines: []
attack_scenario: Looks like a shell command built from user input, but the host must parse as an IP address and the command is an argument list with no shell
remediation: None needed; ipaddress rejects anything that is not an IP and no shell interprets the arguments
source: Written by us, fixed twin of SEED-013
cve_id:
fix_commit:
pair_id: P-005
labelled_by: draft
reviewed_by:
first_pass_agree:
"""

import ipaddress
import subprocess

from flask import Flask, jsonify, request

app = Flask(__name__)


@app.route("/tools/ping")
def ping_host():
    host = request.args.get("host", "").strip()
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return jsonify({"error": "host must be an IP address"}), 400
    completed = subprocess.run(
        ["ping", "-c", "1", str(address)], capture_output=True, text=True, timeout=10
    )
    return jsonify(
        {
            "host": host,
            "reachable": completed.returncode == 0,
            "output": completed.stdout,
        }
    )
