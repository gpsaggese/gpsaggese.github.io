"""Lossless compressed blobs with the vendor's uncompressed identity hashes."""
import gzip
import json
import os
import sqlite3
from pathlib import Path

import caaf_store as base


class CompactStore(base.Store):
    def put_blob(self, kind, data):
        key = base.sha256(data)
        plain = self.root / "blobs" / kind / f"{key}.json"
        if plain.exists():
            assert base.sha256(plain.read_bytes()) == key
        else:
            zipped = self.root / "blobs" / kind / f"{key}.json.gz"
            base._atomic_write(zipped, gzip.compress(data, compresslevel=6, mtime=0))
        return key

    def get_blob(self, kind, key):
        plain = self.root / "blobs" / kind / f"{key}.json"
        if plain.exists():
            data = plain.read_bytes()
        else:
            data = gzip.decompress((self.root / "blobs" / kind / f"{key}.json.gz").read_bytes())
        if base.sha256(data) != key:
            raise ValueError("Compressed checkpoint identity mismatch")
        return data


def seed_pilot_store(source: Path, target: Path, branches: list[str]):
    """Import original rows and blobs only for selected pilot root branches."""
    if (target / "manifest.sqlite").exists():
        raise FileExistsError("Refusing to overwrite a working store")
    original = sqlite3.connect(f"file:{source/'manifest.sqlite'}?mode=ro&immutable=1", uri=True)
    original.row_factory = sqlite3.Row
    new = CompactStore(target)
    counts = {"runs": 0, "branches": 0, "snapshots": 0}
    for bid in sorted(branches):
        branch = dict(original.execute("select * from branches where branch_id=?", (bid,)).fetchone())
        assert branch["parent_snapshot_id"] is None
        run = dict(original.execute("select * from runs where run_id=?", (branch["run_id"],)).fetchone())
        for table, record in [("runs", run), ("branches", branch)]:
            keys = list(record)
            new.db.execute(f"insert into {table} ({','.join(keys)}) values ({','.join('?' for _ in keys)})", [record[k] for k in keys])
            counts[table] += 1
        snapshots = original.execute("select * from snapshots where branch_id=? order by seq", (bid,)).fetchall()
        for snap in snapshots:
            record = dict(snap); keys = list(record)
            for kind, key in [("context", snap["context_hash"]), ("env", snap["env_hash"])]:
                data = (source / "blobs" / kind / f"{key}.json").read_bytes()
                assert new.put_blob(kind, data) == key
            new.db.execute(f"insert into snapshots ({','.join(keys)}) values ({','.join('?' for _ in keys)})", [record[k] for k in keys])
            counts["snapshots"] += 1
        destination = target / "tapes" / f"{bid}.jsonl"
        destination.parent.mkdir(exist_ok=True)
        data = (source / "tapes" / f"{bid}.jsonl").read_bytes()
        assert base.verify_tape([json.loads(l) for l in data.decode().splitlines()])
        destination.write_bytes(data);destination.chmod(0o444)
    assert new.db.execute("pragma integrity_check").fetchone()[0] == "ok"
    new.db.close(); original.close()
    return counts
