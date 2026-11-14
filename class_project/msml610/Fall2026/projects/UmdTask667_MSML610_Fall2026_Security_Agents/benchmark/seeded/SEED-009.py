"""
id: SEED-009
type: seeded
vulnerable: true
cwe: 611
owasp: A02:2025 Security Misconfiguration
severity: high
bug_lines: [28]
attack_scenario: Attacker posts an order whose DTD defines an entity pointing to file:///etc/passwd; the file content comes back in the customer field
remediation: Parse with defusedxml, or create the lxml parser with resolve_entities=False and load_dtd=False
source: Written by us, based on the CWE-611 page
cve_id:
fix_commit:
pair_id:
labelled_by: draft
reviewed_by:
first_pass_agree:
"""

from flask import Flask, jsonify, request
from lxml import etree

app = Flask(__name__)


@app.route("/orders/import", methods=["POST"])
def import_order():
    parser = etree.XMLParser(resolve_entities=True, load_dtd=True)  # {!LINE}
    root = etree.fromstring(request.get_data(), parser)
    order = {
        "customer": root.findtext("customer", default=""),
        "items": [],
    }
    for item in root.iterfind("item"):
        order["items"].append(
            {"sku": item.get("sku"), "qty": int(item.get("qty", "1"))}
        )
    return jsonify(order)
