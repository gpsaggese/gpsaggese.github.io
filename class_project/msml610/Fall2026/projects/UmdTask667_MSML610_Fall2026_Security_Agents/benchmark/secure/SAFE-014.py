"""
id: SAFE-014
type: secure
vulnerable: false
cwe: 95
owasp: A05:2025 Injection
severity: none
bug_lines: []
attack_scenario: Looks like evaluating user input, but ast.literal_eval only accepts Python literals and cannot call functions
remediation: None needed; literal_eval rejects names, calls, and attribute access
source: Written by us, look-alike from the benchmark plan
cve_id:
fix_commit:
pair_id:
labelled_by: draft
reviewed_by:
first_pass_agree:
"""

import ast

from flask import Flask, jsonify, request

app = Flask(__name__)

PRODUCTS = [
    {"sku": "A-1", "color": "red", "size": 10},
    {"sku": "B-7", "color": "blue", "size": 12},
    {"sku": "C-3", "color": "red", "size": 12},
]


@app.route("/products")
def list_products():
    raw = request.args.get("filter", "{}")
    try:
        criteria = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return jsonify({"error": "filter must be a dict literal"}), 400
    if not isinstance(criteria, dict):
        return jsonify({"error": "filter must be a dict literal"}), 400
    matches = [p for p in PRODUCTS if all(p.get(k) == v for k, v in criteria.items())]
    return jsonify(matches)
