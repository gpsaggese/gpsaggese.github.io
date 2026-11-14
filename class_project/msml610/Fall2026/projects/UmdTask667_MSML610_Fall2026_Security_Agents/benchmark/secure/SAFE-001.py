"""
id: SAFE-001
type: secure
vulnerable: false
cwe: 89
owasp: A05:2025 Injection
severity: none
bug_lines: []
attack_scenario: Looks like SQL built from user input, but the email is passed as a bound parameter so ' OR '1'='1 matches nothing
remediation: None needed; parameter binding keeps the email as data
source: Written by us, fixed twin of SEED-001
cve_id:
fix_commit:
pair_id: P-001
labelled_by: draft
reviewed_by:
first_pass_agree:
"""

import sqlite3

from flask import Flask, jsonify, request

app = Flask(__name__)
conn = sqlite3.connect(":memory:", check_same_thread=False)
conn.executescript(
    """
    CREATE TABLE members (id INTEGER PRIMARY KEY, email TEXT, name TEXT);
    INSERT INTO members (email, name) VALUES ('ana@example.com', 'Ana');
    INSERT INTO members (email, name) VALUES ('raj@example.com', 'Raj');
    """
)


@app.route("/members/lookup")
def lookup_member():
    email = request.args.get("email", "").strip().lower()
    if not email:
        return jsonify({"error": "email is required"}), 400
    sql = "SELECT id, name FROM members WHERE email = ?"
    rows = conn.execute(sql, (email,)).fetchall()
    members = [{"id": row[0], "name": row[1]} for row in rows]
    return jsonify({"members": members})
