"""
id: SAFE-007
type: secure
vulnerable: false
cwe: 502
owasp: A08:2025 Software or Data Integrity Failures
severity: none
bug_lines: []
attack_scenario: Looks like YAML loaded from a request body, but yaml.safe_load only builds plain types and rejects python/object tags
remediation: None needed; safe_load cannot construct arbitrary Python objects
source: Written by us, fixed twin of SEED-018
cve_id:
fix_commit:
pair_id: P-007
labelled_by: draft
reviewed_by:
first_pass_agree:
"""

import yaml
from flask import Flask, jsonify, request

app = Flask(__name__)

ALLOWED_STEPS = {"checkout", "build", "test", "deploy"}


@app.route("/pipelines/validate", methods=["POST"])
def validate_pipeline():
    text = request.get_data(as_text=True)
    config = yaml.safe_load(text)
    if not isinstance(config, dict):
        return jsonify({"valid": False, "errors": ["not a mapping"]}), 400
    errors = []
    for index, step in enumerate(config.get("steps", [])):
        if step.get("uses") not in ALLOWED_STEPS:
            errors.append(f"step {index}: unknown action {step.get('uses')}")
    return jsonify({"valid": not errors, "errors": errors})
