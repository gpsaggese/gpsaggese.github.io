"""
id: SAFE-018
type: secure
vulnerable: false
cwe: 611
owasp: A02:2025 Security Misconfiguration
severity: none
bug_lines: []
attack_scenario: Looks like XML parsed from a request body, but defusedxml refuses DTDs and external entities
remediation: None needed; defusedxml raises on entity declarations
source: Written by us, look-alike of SEED-009
cve_id:
fix_commit:
pair_id:
labelled_by: draft
reviewed_by:
first_pass_agree:
"""

from defusedxml import ElementTree
from flask import Flask, jsonify, request

app = Flask(__name__)


@app.route("/orders/import", methods=["POST"])
def import_order():
    root = ElementTree.fromstring(request.get_data())
    order = {
        "customer": root.findtext("customer", default=""),
        "items": [],
    }
    for item in root.iterfind("item"):
        order["items"].append(
            {"sku": item.get("sku"), "qty": int(item.get("qty", "1"))}
        )
    return jsonify(order)
