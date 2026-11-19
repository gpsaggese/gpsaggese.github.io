"""
id: SEED-001
type: seeded
vulnerable: true
cwe: 89
owasp: A05:2025 Injection
severity: high
bug_lines: [40, 41]
attack_scenario: Attacker sends email=' OR '1'='1 and the lookup returns every member instead of one
remediation: Pass the email as a bound parameter, e.g., conn.execute("... WHERE email = ?", (email,))
source: Written by us, based on the CWE-89 page and the project brief key example
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
    sql = f"SELECT id, name FROM members WHERE email = '{email}'"  # {!LINE}
    rows = conn.execute(sql).fetchall()  # {!LINE}
    members = [{"id": row[0], "name": row[1]} for row in rows]
    return jsonify({"members": members})
