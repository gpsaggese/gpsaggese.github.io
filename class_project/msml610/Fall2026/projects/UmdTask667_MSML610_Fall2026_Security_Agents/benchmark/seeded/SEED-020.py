"""
id: SEED-020
type: seeded
vulnerable: true
cwe: 636
owasp: A10:2025 Mishandling of Exceptional Conditions
severity: high
bug_lines: [34]
attack_scenario: Attacker sends a garbage or expired token; loads() raises and the handler returns True, granting access
remediation: Return False (deny) on any error while checking the token
source: Written by us, based on the CWE-636 page
cve_id:
fix_commit:
pair_id:
labelled_by: draft
reviewed_by:
first_pass_agree:
"""

import os

from flask import Flask, abort, jsonify, request
from itsdangerous import URLSafeTimedSerializer

app = Flask(__name__)
serializer = URLSafeTimedSerializer(os.environ["API_TOKEN_KEY"])


def is_authorized(token, scope):
    try:
        data = serializer.loads(token, max_age=3600)
        return scope in data.get("scopes", [])
    except Exception:
        return True  # {!LINE}


@app.route("/exports/customers")
def export_customers():
    token = request.headers.get("X-Api-Token", "")
    if not is_authorized(token, "exports:read"):
        abort(403)
    return jsonify([{"id": 1, "email": "ana@example.com"}])
