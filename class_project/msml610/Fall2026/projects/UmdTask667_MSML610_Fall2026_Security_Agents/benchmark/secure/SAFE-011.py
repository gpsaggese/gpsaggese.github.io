"""
id: SAFE-011
type: secure
vulnerable: false
cwe: 338
owasp: A04:2025 Cryptographic Failures
severity: none
bug_lines: []
attack_scenario: Looks like the random module used where unpredictability matters, but it only shuffles quiz questions
remediation: None needed; predicting the question order gives no advantage
source: Written by us, look-alike from the benchmark plan
cve_id:
fix_commit:
pair_id:
labelled_by: draft
reviewed_by:
first_pass_agree:
"""

import random

from flask import Flask, jsonify, request

app = Flask(__name__)

QUESTIONS = [
    {"id": 1, "text": "What does HTTP stand for?"},
    {"id": 2, "text": "Which port does HTTPS use by default?"},
    {"id": 3, "text": "What is a primary key?"},
    {"id": 4, "text": "What does JSON stand for?"},
    {"id": 5, "text": "Name one HTTP method that is idempotent."},
]


@app.route("/quiz/start")
def start_quiz():
    count = min(int(request.args.get("count", "3")), len(QUESTIONS))
    selected = random.sample(QUESTIONS, k=count)
    random.shuffle(selected)
    return jsonify({"questions": selected})
