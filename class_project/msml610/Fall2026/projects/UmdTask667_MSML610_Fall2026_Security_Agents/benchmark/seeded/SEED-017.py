"""
id: SEED-017
type: seeded
vulnerable: true
cwe: 256
owasp: A07:2025 Authentication Failures
severity: high
bug_lines: [37]
attack_scenario: Anyone who reads the users table (backup, SQL error, insider) gets every password in clear text and reuses them elsewhere
remediation: Store generate_password_hash(password) and verify with check_password_hash
source: Written by us, based on the CWE-256 page
cve_id:
fix_commit:
pair_id:
labelled_by: draft
reviewed_by:
first_pass_agree:
"""

import sqlite3

from flask import Flask, jsonify, request

app = Flask(__name__)
conn = sqlite3.connect(":memory:", check_same_thread=False)
conn.execute("CREATE TABLE users (email TEXT PRIMARY KEY, password TEXT)")


@app.route("/register", methods=["POST"])
def register():
    email = request.form.get("email", "").lower()
    password = request.form.get("password", "")
    if "@" not in email or len(password) < 10:
        return jsonify({"error": "invalid email or password"}), 400
    conn.execute(
        "INSERT INTO users (email, password) VALUES (?, ?)",
        (email, password),  # {!LINE}
    )
    conn.commit()
    return jsonify({"registered": email}), 201


@app.route("/login", methods=["POST"])
def login():
    email = request.form.get("email", "").lower()
    row = conn.execute(
        "SELECT password FROM users WHERE email = ?", (email,)
    ).fetchone()
    ok = row is not None and row[0] == request.form.get("password", "")
    return jsonify({"ok": ok}), 200 if ok else 401
