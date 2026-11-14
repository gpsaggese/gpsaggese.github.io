"""
id: SAFE-009
type: secure
vulnerable: false
cwe: 639
owasp: A01:2025 Broken Access Control
severity: none
bug_lines: []
attack_scenario: Looks like an object fetched by a user-controlled id, but the owner must match the logged-in user
remediation: None needed; other users' invoices return 404
source: Written by us, fixed twin of SEED-006
cve_id:
fix_commit:
pair_id: P-009
labelled_by: draft
reviewed_by:
first_pass_agree:
"""

import os

from flask import Flask, abort, jsonify, session

app = Flask(__name__)
app.secret_key = os.environ["FLASK_SECRET_KEY"]

INVOICES = {
    101: {"owner": "ana", "amount": 120.0, "status": "paid"},
    102: {"owner": "raj", "amount": 75.5, "status": "due"},
}


@app.route("/invoices/<int:invoice_id>")
def get_invoice(invoice_id):
    user = session.get("user")
    if user is None:
        abort(401)
    invoice = INVOICES.get(invoice_id)
    if invoice is None or invoice["owner"] != user:
        abort(404)
    return jsonify(
        {
            "id": invoice_id,
            "amount": invoice["amount"],
            "status": invoice["status"],
        }
    )
