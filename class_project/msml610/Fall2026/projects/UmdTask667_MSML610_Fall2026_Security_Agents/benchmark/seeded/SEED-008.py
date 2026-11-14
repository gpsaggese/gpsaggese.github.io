"""
id: SEED-008
type: seeded
vulnerable: true
cwe: 489
owasp: A02:2025 Security Misconfiguration
severity: high
bug_lines: [49]
attack_scenario: The app listens on 0.0.0.0 with the Werkzeug debugger on; anyone who triggers an error gets an interactive Python console
remediation: Never pass debug=True in deployed code; read it from config that defaults to off and bind to 127.0.0.1 in development
source: Written by us, based on the CWE-489 page
cve_id:
fix_commit:
pair_id:
labelled_by: draft
reviewed_by:
first_pass_agree:
"""

import sqlite3

from flask import Flask, g, jsonify

app = Flask(__name__)
DATABASE = "inventory.db"


def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DATABASE)
    return g.db


@app.teardown_appcontext
def close_db(exc):
    db = g.pop("db", None)
    if db is not None:
        db.close()


@app.route("/items")
def list_items():
    rows = get_db().execute("SELECT sku, name, stock FROM items").fetchall()
    items = [{"sku": r[0], "name": r[1], "stock": r[2]} for r in rows]
    return jsonify(items)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)  # {!LINE}
