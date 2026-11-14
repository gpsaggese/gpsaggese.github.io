"""
id: SAFE-006
type: secure
vulnerable: false
cwe: 1336
owasp: A05:2025 Injection
severity: none
bug_lines: []
attack_scenario: Looks like user input in a Jinja2 template, but the template text is constant and the recipient is passed as an autoescaped variable
remediation: None needed; Jinja2 treats the recipient as data, not template code
source: Written by us, fixed twin of SEED-015
cve_id:
fix_commit:
pair_id: P-006
labelled_by: draft
reviewed_by:
first_pass_agree:
"""

from flask import Flask, render_template_string, request

app = Flask(__name__)

OCCASIONS = {
    "birthday": "Happy birthday!",
    "thanks": "Thank you so much!",
    "welcome": "Welcome to the team!",
}


@app.route("/card")
def greeting_card():
    recipient = request.args.get("to", "friend")
    occasion = request.args.get("occasion", "birthday")
    message = OCCASIONS.get(occasion, OCCASIONS["birthday"])
    template = (
        "<div class='card'><h2>Dear {{ recipient }},</h2>"
        "<p>{{ message }}</p></div>"
    )
    return render_template_string(template, recipient=recipient, message=message)
