"""
id: SAFE-015
type: secure
vulnerable: false
cwe: 89
owasp: A05:2025 Injection
severity: none
bug_lines: []
attack_scenario: Looks like SQL built with an f-string, but the column comes from a fixed allowlist and the filter value is a bound parameter
remediation: None needed; user input never reaches the SQL text
source: Written by us, look-alike from the benchmark plan
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
conn.executescript(
    """
    CREATE TABLE books (title TEXT, author TEXT, year INTEGER);
    INSERT INTO books VALUES ('Dune', 'Herbert', 1965);
    INSERT INTO books VALUES ('Emma', 'Austen', 1815);
    """
)

SORT_COLUMNS = {"title": "title", "author": "author", "year": "year"}


@app.route("/books")
def list_books():
    sort = SORT_COLUMNS.get(request.args.get("sort", "title"), "title")
    author = request.args.get("author", "%")
    sql = f"SELECT title, author, year FROM books WHERE author LIKE ? ORDER BY {sort}"
    rows = conn.execute(sql, (author,)).fetchall()
    return jsonify([{"title": r[0], "author": r[1], "year": r[2]} for r in rows])
