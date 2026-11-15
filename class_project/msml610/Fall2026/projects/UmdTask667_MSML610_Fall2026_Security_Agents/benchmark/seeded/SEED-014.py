"""
id: SEED-014
type: seeded
vulnerable: true
cwe: 95
owasp: A05:2025 Injection
severity: critical
bug_lines: [36]
attack_scenario: Attacker sends formula=__import__('os').system('id') and the server runs it
remediation: Parse the formula with a small expression parser (e.g., ast with an allowlist of nodes) instead of eval
source: Written by us, based on the CWE-95 page
cve_id:
fix_commit:
pair_id:
labelled_by: draft
reviewed_by:
first_pass_agree:
"""

from flask import Flask, jsonify, request

app = Flask(__name__)

BASE_PRICES = {"basic": 9.0, "pro": 29.0, "team": 79.0}


@app.route("/pricing/quote", methods=["POST"])
def quote():
    data = request.get_json(force=True)
    plan = data.get("plan", "basic")
    seats = int(data.get("seats", 1))
    formula = data.get("formula", "base * seats")
    if plan not in BASE_PRICES:
        return jsonify({"error": "unknown plan"}), 400
    variables = {"base": BASE_PRICES[plan], "seats": seats}
    price = eval(formula, {}, variables)  # {!LINE}
    return jsonify({"plan": plan, "seats": seats, "price": round(float(price), 2)})
