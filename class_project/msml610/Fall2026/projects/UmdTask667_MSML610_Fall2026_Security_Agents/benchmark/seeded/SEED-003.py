"""
id: SEED-003
type: seeded
vulnerable: true
cwe: 502
owasp: A08:2025 Software or Data Integrity Failures
severity: critical
bug_lines: [36]
attack_scenario: Attacker sets the cart cookie to a crafted pickle whose __reduce__ runs os.system, getting code execution on the server
remediation: Store the cart as JSON and parse it with json.loads, or sign the cookie and never unpickle client data
source: Written by us, based on the CWE-502 page and the project brief key example
cve_id:
fix_commit:
pair_id: P-003
labelled_by: draft
reviewed_by:
first_pass_agree:
"""

import base64
import pickle

from flask import Flask, jsonify, request

app = Flask(__name__)

PRICES = {"apple": 0.5, "bread": 2.25, "milk": 1.1}


@app.route("/cart/total")
def cart_total():
    cookie = request.cookies.get("cart")
    if cookie is None:
        return jsonify({"total": 0.0, "items": {}})
    raw = base64.urlsafe_b64decode(cookie.encode("ascii"))
    items = pickle.loads(raw)  # {!LINE}
    total = 0.0
    for item, quantity in items.items():
        total += PRICES.get(item, 0.0) * int(quantity)
    return jsonify({"total": round(total, 2), "items": items})
