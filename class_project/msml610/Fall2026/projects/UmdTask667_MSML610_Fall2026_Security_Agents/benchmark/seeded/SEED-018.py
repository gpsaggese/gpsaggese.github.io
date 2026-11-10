"""
id: SEED-018
type: seeded
vulnerable: true
cwe: 502
owasp: A08:2025 Software or Data Integrity Failures
severity: critical
bug_lines: [31]
attack_scenario: Attacker uploads a pipeline file with !!python/object/apply:os.system and the loader runs the command
remediation: Load the file with yaml.safe_load
source: Written by us, based on the CWE-502 page and the PyYAML documentation
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
    config = yaml.load(text, Loader=yaml.Loader)  # {!LINE}
    if not isinstance(config, dict):
        return jsonify({"valid": False, "errors": ["not a mapping"]}), 400
    errors = []
    for index, step in enumerate(config.get("steps", [])):
        if step.get("uses") not in ALLOWED_STEPS:
            errors.append(f"step {index}: unknown action {step.get('uses')}")
    return jsonify({"valid": not errors, "errors": errors})
