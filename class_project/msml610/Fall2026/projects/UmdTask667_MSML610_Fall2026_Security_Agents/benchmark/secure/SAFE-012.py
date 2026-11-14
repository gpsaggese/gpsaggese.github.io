"""
id: SAFE-012
type: secure
vulnerable: false
cwe: 798
owasp: A07:2025 Authentication Failures
severity: none
bug_lines: []
attack_scenario: Looks like a hard-coded password, but the constants are UI labels and a field name, not credentials
remediation: None needed; no secret is stored in the code
source: Written by us, look-alike from the benchmark plan
cve_id:
fix_commit:
pair_id:
labelled_by: draft
reviewed_by:
first_pass_agree:
"""

from flask import Flask, render_template_string

app = Flask(__name__)

PASSWORD_PROMPT = "Enter your password:"
PASSWORD_FIELD = "password"
PASSWORD_HINT = "At least 12 characters, including a number."

LOGIN_FORM = """
<form method="post" action="/login">
  <label>Email <input name="email" type="email"></label>
  <label>{{ prompt }} <input name="{{ field }}" type="password"></label>
  <small>{{ hint }}</small>
  <button type="submit">Sign in</button>
</form>
"""


@app.route("/login", methods=["GET"])
def login_form():
    return render_template_string(
        LOGIN_FORM, prompt=PASSWORD_PROMPT, field=PASSWORD_FIELD, hint=PASSWORD_HINT
    )
