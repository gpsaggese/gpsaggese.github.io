"""
id: SEED-012
type: seeded
vulnerable: true
cwe: 338
owasp: A04:2025 Cryptographic Failures
severity: high
bug_lines: [34]
attack_scenario: random uses a Mersenne Twister; an attacker who collects enough outputs predicts the next reset token and takes over the account
remediation: Generate tokens with secrets.token_urlsafe() or secrets.choice()
source: Written by us, based on the CWE-338 page
cve_id:
fix_commit:
pair_id: P-008
labelled_by: draft
reviewed_by:
first_pass_agree:
"""

import random
import string
import time

from flask import Flask, jsonify, request

app = Flask(__name__)

RESET_TOKENS = {}
TOKEN_TTL_SECONDS = 900


def make_token(length=24):
    alphabet = string.ascii_letters + string.digits
    return "".join(random.choice(alphabet) for _ in range(length))  # {!LINE}


def send_reset_email(address, link):
    app.logger.info("queued reset email for %s", address)


@app.route("/password/forgot", methods=["POST"])
def forgot_password():
    email = request.form.get("email", "")
    token = make_token()
    RESET_TOKENS[token] = (email, time.time() + TOKEN_TTL_SECONDS)
    send_reset_email(email, f"https://shop.example.com/reset?token={token}")
    return jsonify({"sent": True})
