"""
id: SEED-015
type: seeded
vulnerable: true
cwe: 1336
owasp: A05:2025 Injection
severity: critical
bug_lines: [37]
attack_scenario: Attacker sends to={{ config.items() }} (or a payload reaching os.popen) and Jinja2 evaluates it on the server
remediation: Keep the template constant and pass the recipient as a variable: render_template_string('... {{ recipient }} ...', recipient=recipient)
source: Written by us, based on the CWE-1336 page
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
        "<div class='card'><h2>Dear " + recipient + ",</h2>"  # {!LINE}
        "<p>{{ message }}</p></div>"
    )
    return render_template_string(template, message=message)
