"""
id: SEED-006
type: seeded
vulnerable: true
cwe: 639
owasp: A01:2025 Broken Access Control
severity: high
bug_lines: [39]
attack_scenario: Logged-in user ana requests /invoices/102 and reads raj's invoice by changing the id
remediation: Return 404 unless invoice['owner'] equals the logged-in user
source: Written by us, based on the CWE-639 page
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
    if invoice is None:  # {!LINE}
        abort(404)
    return jsonify(
        {
            "id": invoice_id,
            "amount": invoice["amount"],
            "status": invoice["status"],
        }
    )
