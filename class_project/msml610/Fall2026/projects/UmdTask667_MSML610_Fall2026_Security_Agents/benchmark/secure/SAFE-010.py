"""
id: SAFE-010
type: secure
vulnerable: false
cwe: 328
owasp: A04:2025 Cryptographic Failures
severity: none
bug_lines: []
attack_scenario: Looks like MD5 used for security, but it only builds a cache key and is marked usedforsecurity=False
remediation: None needed; collisions only cause a cache miss
source: Written by us, look-alike from the benchmark plan
cve_id:
fix_commit:
pair_id:
labelled_by: draft
reviewed_by:
first_pass_agree:
"""

import hashlib
import json

from flask import Flask, jsonify, request

app = Flask(__name__)
CACHE = {}


def cache_key(params):
    payload = json.dumps(params, sort_keys=True).encode("utf-8")
    return hashlib.md5(payload, usedforsecurity=False).hexdigest()


def run_search(params):
    return [{"sku": "A-1", "score": 0.9}, {"sku": "B-7", "score": 0.4}]


@app.route("/search")
def search():
    params = {"q": request.args.get("q", ""), "page": request.args.get("page", "1")}
    key = cache_key(params)
    if key not in CACHE:
        CACHE[key] = run_search(params)
    return jsonify({"results": CACHE[key], "cached_key": key})
