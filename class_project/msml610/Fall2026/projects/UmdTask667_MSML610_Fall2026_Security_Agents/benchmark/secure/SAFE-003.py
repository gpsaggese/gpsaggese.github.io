"""
id: SAFE-003
type: secure
vulnerable: false
cwe: 502
owasp: A08:2025 Software or Data Integrity Failures
severity: none
bug_lines: []
attack_scenario: Looks like decoding an object from a client cookie, but json.loads only builds plain data and cannot run code
remediation: None needed; JSON parsing cannot create arbitrary objects
source: Written by us, fixed twin of SEED-003
cve_id:
fix_commit:
pair_id: P-003
labelled_by: draft
reviewed_by:
first_pass_agree:
"""

import base64
import json

from flask import Flask, jsonify, request

app = Flask(__name__)

PRICES = {"apple": 0.5, "bread": 2.25, "milk": 1.1}


@app.route("/cart/total")
def cart_total():
    cookie = request.cookies.get("cart")
    if cookie is None:
        return jsonify({"total": 0.0, "items": {}})
    raw = base64.urlsafe_b64decode(cookie.encode("ascii"))
    items = json.loads(raw)
    total = 0.0
    for item, quantity in items.items():
        total += PRICES.get(item, 0.0) * int(quantity)
    return jsonify({"total": round(total, 2), "items": items})
