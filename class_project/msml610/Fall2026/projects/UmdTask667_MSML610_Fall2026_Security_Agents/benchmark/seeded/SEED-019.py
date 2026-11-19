"""
id: SEED-019
type: seeded
vulnerable: true
cwe: 532
owasp: A09:2025 Security Logging and Alerting Failures
severity: medium
bug_lines: [35]
attack_scenario: Anyone with access to the log files or the log pipeline reads users' passwords
remediation: Never log secrets; log the username and the outcome only
source: Written by us, based on the CWE-532 page
cve_id:
fix_commit:
pair_id:
labelled_by: draft
reviewed_by:
first_pass_agree:
"""

import logging

from flask import Flask, jsonify, request
from werkzeug.security import check_password_hash, generate_password_hash

app = Flask(__name__)
logger = logging.getLogger("auth")

USERS = {"ana": generate_password_hash("correct horse battery")}


@app.route("/login", methods=["POST"])
def login():
    username = request.form.get("username", "")
    password = request.form.get("password", "")
    logger.info("login attempt user=%s password=%s", username, password)  # {!LINE}
    stored = USERS.get(username)
    if stored is None or not check_password_hash(stored, password):
        logger.warning("login failed for user=%s", username)
        return jsonify({"ok": False}), 401
    logger.info("login ok for user=%s", username)
    return jsonify({"ok": True})
