"""
id: SEED-011
type: seeded
vulnerable: true
cwe: 321
owasp: A04:2025 Cryptographic Failures
severity: high
bug_lines: [23]
attack_scenario: Anyone who reads the source signs their own session cookie with the known key and becomes any user
remediation: Load SECRET_KEY from an environment variable or secret manager and rotate it
source: Written by us, based on the CWE-321 page
cve_id:
fix_commit:
pair_id:
labelled_by: draft
reviewed_by:
first_pass_agree:
"""

from flask import Flask, abort, jsonify, request, session

app = Flask(__name__)
app.config["SECRET_KEY"] = "f3a9c1d2-orders-2024"  # {!LINE}

ACCOUNTS = {"ana": "staff", "raj": "manager"}


@app.route("/login", methods=["POST"])
def login():
    username = request.form.get("username", "")
    if username not in ACCOUNTS:
        abort(401)
    session["user"] = username
    return jsonify({"user": username})


@app.route("/reports/monthly")
def monthly_report():
    user = session.get("user")
    if user is None or ACCOUNTS.get(user) != "manager":
        abort(403)
    return jsonify({"revenue": 48210, "orders": 1310})
