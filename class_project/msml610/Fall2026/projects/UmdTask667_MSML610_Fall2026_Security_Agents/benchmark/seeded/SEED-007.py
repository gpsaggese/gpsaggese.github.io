"""
id: SEED-007
type: seeded
vulnerable: true
cwe: 918
owasp: A01:2025 Broken Access Control
severity: high
bug_lines: [33]
attack_scenario: Attacker asks for url=http://169.254.169.254/latest/meta-data/ and the server fetches cloud credentials from the internal metadata service
remediation: Allow only https URLs whose resolved host is on an allowlist and not a private or link-local address
source: Written by us, based on the CWE-918 page
cve_id:
fix_commit:
pair_id:
labelled_by: draft
reviewed_by:
first_pass_agree:
"""

import requests
from flask import Flask, jsonify, request

app = Flask(__name__)

MAX_CHARS = 200_000


@app.route("/preview")
def link_preview():
    url = request.args.get("url", "")
    if not url.startswith(("http://", "https://")):
        return jsonify({"error": "url must be http or https"}), 400
    response = requests.get(url, timeout=5)  # {!LINE}
    body = response.text[:MAX_CHARS]
    title = ""
    start = body.find("<title>")
    end = body.find("</title>")
    if start != -1 and end > start:
        title = body[start + len("<title>") : end].strip()
    return jsonify({"url": url, "status": response.status_code, "title": title})
