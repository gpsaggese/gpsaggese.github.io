"""
id: SAFE-008
type: secure
vulnerable: false
cwe: 338
owasp: A04:2025 Cryptographic Failures
severity: none
bug_lines: []
attack_scenario: Looks like a random reset token, but secrets.choice draws from the OS CSPRNG so tokens cannot be predicted
remediation: None needed; the secrets module is designed for security tokens
source: Written by us, fixed twin of SEED-012
cve_id:
fix_commit:
pair_id: P-008
labelled_by: draft
reviewed_by:
first_pass_agree:
"""

import secrets
import string
import time

from flask import Flask, jsonify, request

app = Flask(__name__)

RESET_TOKENS = {}
TOKEN_TTL_SECONDS = 900


def make_token(length=24):
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


def send_reset_email(address, link):
    app.logger.info("queued reset email for %s", address)


@app.route("/password/forgot", methods=["POST"])
def forgot_password():
    email = request.form.get("email", "")
    token = make_token()
    RESET_TOKENS[token] = (email, time.time() + TOKEN_TTL_SECONDS)
    send_reset_email(email, f"https://shop.example.com/reset?token={token}")
    return jsonify({"sent": True})
