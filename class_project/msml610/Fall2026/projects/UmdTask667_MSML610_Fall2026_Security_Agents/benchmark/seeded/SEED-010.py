"""
id: SEED-010
type: seeded
vulnerable: true
cwe: 328
owasp: A04:2025 Cryptographic Failures
severity: high
bug_lines: [31]
attack_scenario: After a database leak, unsalted MD5 hashes are cracked in seconds with rainbow tables or a GPU
remediation: Hash passwords with a slow salted KDF, e.g., werkzeug.security.generate_password_hash or bcrypt/argon2
source: Written by us, based on the CWE-328 page
cve_id:
fix_commit:
pair_id:
labelled_by: draft
reviewed_by:
first_pass_agree:
"""

import hashlib
import sqlite3

from flask import Flask, jsonify, request

app = Flask(__name__)
conn = sqlite3.connect(":memory:", check_same_thread=False)
conn.execute("CREATE TABLE accounts (username TEXT PRIMARY KEY, pw_hash TEXT)")


def hash_password(password):
    return hashlib.md5(password.encode("utf-8")).hexdigest()  # {!LINE}


@app.route("/signup", methods=["POST"])
def signup():
    data = request.get_json(force=True)
    username, password = data["username"], data["password"]
    if len(password) < 10:
        return jsonify({"error": "password too short"}), 400
    conn.execute(
        "INSERT INTO accounts VALUES (?, ?)", (username, hash_password(password))
    )
    conn.commit()
    return jsonify({"created": username}), 201


@app.route("/login", methods=["POST"])
def login():
    data = request.get_json(force=True)
    row = conn.execute(
        "SELECT pw_hash FROM accounts WHERE username = ?", (data["username"],)
    ).fetchone()
    ok = row is not None and row[0] == hash_password(data["password"])
    return jsonify({"ok": ok}), 200 if ok else 401
