"""
Checkpoint store: content-addressed blobs, an SQLite manifest, append-only tapes
and the API spending ledger.

Layout under the store root:
  blobs/context/<sha256>.json   agent half (canonical JSON, sorted keys)
  blobs/env/<sha256>.json       environment half (canonical JSON of the game)
  manifest.sqlite               runs, branches, checkpoints
  tapes/<branch_id>.jsonl       one envelope per boundary, hash-chained

Write order for a checkpoint: context blob, then environment blob, then the
manifest row in one transaction. Blobs are written to a temporary name, fsynced
and renamed, so a crash can leave an orphaned blob but never a manifest row that
points at missing or partial data.

Import as:

import caaf_store as cstore
"""

import fcntl
import hashlib
import json
import os
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

_SCHEMA = """
create table if not exists runs (
  run_id text primary key, created real not null, config_json text not null);
create table if not exists branches (
  branch_id text primary key, run_id text not null references runs(run_id),
  parent_snapshot_id text references snapshots(id), fork_step integer,
  intervention_json text, created real not null,
  status text not null default 'running', outcome_json text);
create table if not exists snapshots (
  id text primary key, run_id text not null,
  branch_id text not null references branches(branch_id),
  seq integer not null, step integer not null, parent_id text,
  context_hash text not null, env_hash text not null, created real not null,
  unique (branch_id, seq));
"""

_SCHEMA_LOCK = threading.Lock()


def canonical(obj: Any) -> bytes:
    """
    Serialize to canonical JSON: sorted keys, no spaces, UTF-8.
    """
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    os.chmod(path, 0o444)


# #############################################################################
# Store
# #############################################################################


class Store:
    """
    One store per experiment; each thread opens its own `Store` (one SQLite
    connection per thread).
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(
            self.root / "manifest.sqlite",
            timeout=120,
            isolation_level=None,
            check_same_thread=False,
        )
        self.db.execute("pragma busy_timeout=120000")
        self.db.execute("pragma foreign_keys=on")
        with _SCHEMA_LOCK:
            mode = self.db.execute("pragma journal_mode").fetchall()[0][0]
            if mode != "wal":
                self.db.execute("pragma journal_mode=wal")
            self.db.executescript(_SCHEMA)

    # ---- blobs

    def put_blob(self, kind: str, data: bytes) -> str:
        h = sha256(data)
        _atomic_write(self.root / "blobs" / kind / f"{h}.json", data)
        return h

    def get_blob(self, kind: str, h: str) -> bytes:
        data = (self.root / "blobs" / kind / f"{h}.json").read_bytes()
        if sha256(data) != h:
            raise ValueError(f"blob {kind}/{h} is corrupt")
        return data

    # ---- runs and branches

    def new_run(self, config: Dict) -> str:
        run_id = "run-" + uuid.uuid4().hex[:12]
        self.db.execute(
            "insert into runs values (?,?,?)",
            (run_id, time.time(), canonical(config).decode()),
        )
        return run_id

    def run_config(self, run_id: str) -> Dict:
        rows = self.query(
            "select config_json from runs where run_id=?", (run_id,)
        )
        return json.loads(rows[0]["config_json"])

    def new_branch(
        self,
        run_id: str,
        *,
        parent_snapshot_id: Optional[str] = None,
        fork_step: Optional[int] = None,
        intervention: Optional[Dict] = None,
    ) -> str:
        branch_id = "br-" + uuid.uuid4().hex[:12]
        self.db.execute(
            "insert into branches (branch_id, run_id, parent_snapshot_id,"
            " fork_step, intervention_json, created) values (?,?,?,?,?,?)",
            (
                branch_id,
                run_id,
                parent_snapshot_id,
                fork_step,
                canonical(intervention).decode(),
                time.time(),
            ),
        )
        return branch_id

    def finish_branch(self, branch_id: str, status: str, outcome: Dict) -> None:
        self.db.execute(
            "update branches set status=?, outcome_json=? where branch_id=?",
            (status, canonical(outcome).decode(), branch_id),
        )

    def branch(self, branch_id: str) -> Optional[Dict]:
        rows = self.query(
            "select * from branches where branch_id=?", (branch_id,)
        )
        return rows[0] if rows else None

    # ---- checkpoints

    def put_snapshot(
        self,
        *,
        run_id: str,
        branch_id: str,
        seq: int,
        step: int,
        parent_id: Optional[str],
        context: Dict,
        env_state: bytes,
    ) -> Dict:
        """
        Save a checkpoint.

        `seq` numbers the branch's checkpoints; `step` is the number of model
        calls made so far.
        """
        # 1. Context half.
        context_hash = self.put_blob("context", canonical(context))
        # 2. Environment half.
        env_hash = self.put_blob("env", env_state)
        sid = sha256(
            f"{context_hash}:{env_hash}:{parent_id or ''}:{branch_id}".encode()
        )
        # 3. Manifest row, only after both halves are durable.
        self.db.execute("begin immediate")
        try:
            self.db.execute(
                "insert into snapshots values (?,?,?,?,?,?,?,?,?)",
                (
                    sid,
                    run_id,
                    branch_id,
                    seq,
                    step,
                    parent_id,
                    context_hash,
                    env_hash,
                    time.time(),
                ),
            )
            self.db.execute("commit")
        except BaseException:
            self.db.execute("rollback")
            raise
        return {"id": sid, "context_hash": context_hash, "env_hash": env_hash}

    def snapshot(self, sid: str) -> Dict:
        rows = self.query("select * from snapshots where id=?", (sid,))
        if not rows:
            raise KeyError(sid)
        return rows[0]

    def snapshots(self, branch_id: str) -> List[Dict]:
        return self.query(
            "select * from snapshots where branch_id=? order by seq",
            (branch_id,),
        )

    def query(self, q: str, args: tuple = ()) -> List[Dict]:
        """
        Read every row and close the cursor, so no read transaction stays open.
        """
        cur = self.db.execute(q, args)
        try:
            cols = [c[0] for c in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]
        finally:
            cur.close()

    # ---- tapes

    def tape_path(self, branch_id: str) -> Path:
        return self.root / "tapes" / f"{branch_id}.jsonl"

    def read_tape(self, branch_id: str) -> List[Dict]:
        p = self.tape_path(branch_id)
        if not p.exists():
            return []
        return [json.loads(line) for line in p.read_text().splitlines()]


# #############################################################################
# Tape
# #############################################################################


class Tape:
    """
    Append-only, hash-chained envelopes for one branch, fsynced as they happen.
    """

    def __init__(self, store: Store, run_id: str, branch_id: str) -> None:
        self.path = store.tape_path(branch_id)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id, self.branch_id = run_id, branch_id
        self.seq, self.prev = 0, None
        self.f = open(self.path, "a")

    def record(
        self,
        kind: str,
        step: int,
        request: Dict,
        response: Dict,
        meta: Optional[Dict] = None,
    ) -> Dict:
        env = {
            "v": 1,
            "run_id": self.run_id,
            "branch_id": self.branch_id,
            "seq": self.seq,
            "step": step,
            "kind": kind,
            "request": request,
            "response": response,
            "meta": meta or {},
            "prev": self.prev,
        }
        env["hash"] = sha256(canonical(env))
        self.f.write(canonical(env).decode() + "\n")
        self.f.flush()
        os.fsync(self.f.fileno())
        self.seq, self.prev = self.seq + 1, env["hash"]
        return env

    def close(self) -> None:
        if not self.f.closed:
            self.f.close()
            os.chmod(self.path, 0o444)


def verify_tape(envelopes: List[Dict]) -> bool:
    prev = None
    for e in envelopes:
        body = {k: v for k, v in e.items() if k != "hash"}
        if e["prev"] != prev or sha256(canonical(body)) != e["hash"]:
            return False
        prev = e["hash"]
    return True


# #############################################################################
# Ledger
# #############################################################################


class BudgetExceeded(Exception):
    pass


class Ledger:
    """
    Spending guard for the hosted model.

    Every call's cost is appended to a file-locked ledger shared by all threads
    and processes, and no call starts once the total reaches the cap. Cost is
    computed from the usage the API reports: uncached input, cache writes (x1.25
    input price), cache reads (x0.1 input price) and output.
    """

    def __init__(self, path: Path, cap_usd: float, prices: Dict) -> None:
        self.path, self.cap, self.prices = Path(path), float(cap_usd), prices
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def spent(self) -> float:
        with open(self.path) as f:
            fcntl.flock(f, fcntl.LOCK_SH)
            return sum(
                json.loads(line)["cost_usd"] for line in f if line.strip()
            )

    def check(self) -> None:
        spent = self.spent()
        if spent >= self.cap - self.prices.get("margin_usd", 0.0):
            raise BudgetExceeded(
                f"spent ${spent:.4f} of the ${self.cap:.2f} cap"
            )

    def cost(self, usage: Optional[Dict]) -> Dict:
        usage = usage or {}
        prompt = usage.get("prompt_tokens") or 0
        details = usage.get("prompt_tokens_details") or {}
        read = (
            usage.get("cache_read_input_tokens")
            or details.get("cached_tokens")
            or 0
        )
        write = usage.get("cache_creation_input_tokens") or 0
        # litellm reports prompt_tokens including cache reads and writes for
        # Anthropic; guard the other convention.
        uncached = prompt - read - write if prompt >= read + write else prompt
        prompt = max(prompt, uncached + read + write)
        out = usage.get("completion_tokens") or 0
        long = prompt > self.prices.get("long_context_threshold", 10**12)
        pin = self.prices["input_long" if long else "input"] / 1e6
        pout = self.prices["output_long" if long else "output"] / 1e6
        usd = uncached * pin + write * pin * 1.25 + read * pin * 0.1 + out * pout
        return {
            "prompt_tokens": prompt,
            "uncached": uncached,
            "cache_write": write,
            "cache_read": read,
            "completion_tokens": out,
            "cost_usd": round(usd, 6),
        }

    def add(self, usage: Optional[Dict], model: str) -> float:
        row = {"t": time.time(), "model": model, **self.cost(usage)}
        with open(self.path, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(json.dumps(row) + "\n")
        return row["cost_usd"]
