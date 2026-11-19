"""
id: SAFE-017
type: secure
vulnerable: false
cwe: 532
owasp: A09:2025 Security Logging and Alerting Failures
severity: none
bug_lines: []
attack_scenario: Looks like credentials written to logs, but only the username, password length, and outcome are logged
remediation: None needed; no secret value reaches the log
source: Written by us, look-alike of SEED-019
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
    logger.info("login attempt user=%s password_len=%d", username, len(password))
    stored = USERS.get(username)
    if stored is None or not check_password_hash(stored, password):
        logger.warning("login failed for user=%s", username)
        return jsonify({"ok": False}), 401
    logger.info("login ok for user=%s", username)
    return jsonify({"ok": True})
