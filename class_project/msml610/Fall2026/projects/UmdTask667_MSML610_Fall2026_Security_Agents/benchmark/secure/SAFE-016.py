"""
id: SAFE-016
type: secure
vulnerable: false
cwe: 918
owasp: A01:2025 Broken Access Control
severity: none
bug_lines: []
attack_scenario: Looks like the server fetches a URL chosen by the user, but the user only picks a key and the URL comes from a fixed map
remediation: None needed; the user cannot make the server call an arbitrary host
source: Written by us, look-alike from the benchmark plan
cve_id:
fix_commit:
pair_id:
labelled_by: draft
reviewed_by:
first_pass_agree:
"""

import requests
from flask import Flask, abort, jsonify, request

app = Flask(__name__)

RATE_SOURCES = {
    "ecb": "https://data-api.ecb.europa.eu/service/data/EXR/D.USD.EUR.SP00.A",
    "fed": "https://api.example-fed.gov/rates/usd",
    "boe": "https://api.example-boe.co.uk/rates/gbp",
}


@app.route("/rates")
def get_rates():
    source = request.args.get("source", "ecb")
    url = RATE_SOURCES.get(source)
    if url is None:
        abort(400)
    response = requests.get(url, timeout=5)
    response.raise_for_status()
    return jsonify({"source": source, "status": response.status_code})
