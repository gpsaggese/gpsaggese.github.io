"""
id: SEED-016
type: seeded
vulnerable: true
cwe: 798
owasp: A07:2025 Authentication Failures
severity: high
bug_lines: [29]
attack_scenario: Anyone with read access to the repo logs into the ops console with the password in the source
remediation: Read the admin credential hash from a secret store or environment and compare with check_password_hash
source: Written by us, based on the CWE-798 page
cve_id:
fix_commit:
pair_id:
labelled_by: draft
reviewed_by:
first_pass_agree:
"""

import hmac
import os

from flask import Flask, abort, jsonify, request, session

app = Flask(__name__)
app.secret_key = os.environ["FLASK_SECRET_KEY"]

ADMIN_USER = "ops-admin"
ADMIN_PASSWORD = "Winter2024!ops"  # {!LINE}


@app.route("/ops/login", methods=["POST"])
def ops_login():
    username = request.form.get("username", "")
    password = request.form.get("password", "")
    user_ok = hmac.compare_digest(username, ADMIN_USER)
    password_ok = hmac.compare_digest(password, ADMIN_PASSWORD)
    if not (user_ok and password_ok):
        abort(401)
    session["ops"] = True
    return jsonify({"user": username})
