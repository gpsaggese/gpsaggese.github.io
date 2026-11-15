"""
id: SEED-005
type: seeded
vulnerable: true
cwe: 862
owasp: A01:2025 Broken Access Control
severity: high
bug_lines: [51]
attack_scenario: An anonymous user sends DELETE /admin/users/u2 and removes an account without logging in
remediation: Protect the route with the same @login_required decorator and an admin role check
source: Written by us, based on the CWE-862 page
cve_id:
fix_commit:
pair_id:
labelled_by: draft
reviewed_by:
first_pass_agree:
"""

import os
from functools import wraps

from flask import Flask, abort, jsonify, session

app = Flask(__name__)
app.secret_key = os.environ["FLASK_SECRET_KEY"]

USERS = {
    "u1": {"name": "Ana", "role": "admin"},
    "u2": {"name": "Raj", "role": "staff"},
}


def login_required(view):
    @wraps(view)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            abort(401)
        return view(*args, **kwargs)

    return wrapper


@app.route("/profile")
@login_required
def profile():
    return jsonify(USERS[session["user_id"]])


@app.route("/admin/users/<user_id>", methods=["DELETE"])
def delete_user(user_id):  # {!LINE}
    if user_id not in USERS:
        abort(404)
    USERS.pop(user_id)
    return jsonify({"deleted": user_id})
