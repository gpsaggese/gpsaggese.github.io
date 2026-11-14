"""
Unit tests for the checkpoint machinery; no API calls (every model reply is
planted with an override).

Run from the project folder:

> python -m pytest -q test/test_caaf.py
"""

import json
import sys
from pathlib import Path

import chess
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import caaf_agent as cagent  # noqa: E402
import caaf_chess as cchess  # noqa: E402
import caaf_store as cstore  # noqa: E402
import caaf_utils as cutils  # noqa: E402


def _config() -> dict:
    config = cagent.load_config("chess_haiku.yaml")
    # No ledger: the tests never call the model.
    config["model"].pop("budget")
    return config


def _reply(move: str) -> str:
    return f"THOUGHT: test.\n\n```mswea_bash_command\nplay move {move}\n```"


def _best_line(fen: str) -> list:
    """
    White's fastest mate against the defender, as SAN moves.
    """
    game, line = cchess.new_game(fen), []
    while game["status"] == "ongoing":
        b = cchess.board_of(game)
        best = None
        for m in b.legal_moves:
            b.push(m)
            v = 0 if b.is_checkmate() else None
            if v is None and not (
                b.is_stalemate() or b.is_insufficient_material()
            ):
                reply = cchess.defender_choice(b)
                b.push(reply)
                val = cchess.white_value(b)
                if val["wdl"] == 2:
                    v = val["dtz"] + 2
                b.pop()
            b.pop()
            if v is not None and (best is None or v < best[0]):
                best = (v, b.san(m))
        line.append(best[1])
        game, _, _ = cchess.play(game, ["play", "move", best[1]])
    return line


# #############################################################################
# Board and referee
# #############################################################################


def test_board_refuses_other_commands() -> None:
    fen = cchess.start_positions(1, seed=1, dtz=3, slack_plies=8)[0]["fen"]
    game = cchess.new_game(fen)
    for cmd in ("ls /", "cat /etc/passwd", "play", "play move", "python x.py"):
        out_game, out, code = cchess.play(game, cmd.split())
        assert code == 127 and out_game == game and "Unknown command" in out


def test_illegal_move_leaves_board_unchanged() -> None:
    fen = cchess.start_positions(1, seed=1, dtz=3, slack_plies=8)[0]["fen"]
    game = cchess.new_game(fen)
    out_game, out, code = cchess.play(game, ["play", "move", "Qh5"])
    assert code == 1 and out_game == game and "Illegal" in out


def test_best_line_mates_and_referee_agrees() -> None:
    pos = cchess.start_positions(1, seed=2, dtz=5, slack_plies=8)[0]
    line = _best_line(pos["fen"])
    assert len(line) == 3
    game = cchess.new_game(pos["fen"])
    for mv in line:
        game, _, _ = cchess.play(game, ["play", "move", mv])
    assert game["status"] == "checkmate"
    ref = cchess.referee(pos["fen"], game["moves_uci"])
    assert (
        ref["won"] and ref["started_winning"] and ref["pivot_white_move"] is None
    )


def test_referee_finds_pivot_on_stalemate_blunder() -> None:
    # White Kc7, Rb1; Black Ka8: Ra1 mates, but Rb7 stalemates.
    fen = "k7/2K5/8/8/8/8/8/1R6 w - - 0 1"
    b = chess.Board(fen)
    assert cchess.white_value(b)["win"]
    ref = cchess.referee(fen, ["b1b7"])
    assert ref["pivot_white_move"] == 1 and ref["end"] == "stalemate"


# #############################################################################
# Store and tape
# #############################################################################


def test_tape_chain_detects_tampering(tmp_path: Path) -> None:
    st = cstore.Store(tmp_path)
    tape = cstore.Tape(st, "run-x", "br-x")
    for i in range(3):
        tape.record("command", i, {"command": "play show"}, {"output": str(i)})
    tape.close()
    envs = st.read_tape("br-x")
    assert cstore.verify_tape(envs)
    envs[1]["response"]["output"] = "changed"
    assert not cstore.verify_tape(envs)


def test_corrupt_blob_is_rejected(tmp_path: Path) -> None:
    st = cstore.Store(tmp_path)
    h = st.put_blob("env", b'{"a":1}')
    p = tmp_path / "blobs" / "env" / f"{h}.json"
    p.chmod(0o644)
    p.write_bytes(b'{"a":2}')
    with pytest.raises(ValueError):
        st.get_blob("env", h)


def test_crash_between_halves_leaves_no_manifest_row(tmp_path: Path) -> None:
    st = cstore.Store(tmp_path)
    run_id = st.new_run({})
    branch_id = st.new_branch(run_id)

    class Boom(Exception):
        pass

    orig = st.put_blob

    def put_blob(kind, data):
        if kind == "env":
            raise Boom()
        return orig(kind, data)

    st.put_blob = put_blob
    with pytest.raises(Boom):
        st.put_snapshot(
            run_id=run_id,
            branch_id=branch_id,
            seq=0,
            step=0,
            parent_id=None,
            context={"x": 1},
            env_state=b"{}",
        )
    assert st.snapshots(branch_id) == []
    assert len(list((tmp_path / "blobs" / "context").iterdir())) == 1


def test_ledger_cost_and_cap(tmp_path: Path) -> None:
    prices = {"input": 0.10, "output": 0.50}
    led = cstore.Ledger(tmp_path / "l.jsonl", 0.001, prices)
    usage = {"prompt_tokens": 1000, "completion_tokens": 100}
    assert led.add(usage, "m") == pytest.approx(1000 * 0.1e-6 + 100 * 0.5e-6)
    led.check()
    for _ in range(10):
        led.add(usage, "m")
    with pytest.raises(cstore.BudgetExceeded):
        led.check()


# #############################################################################
# Checkpoint, fork and replay
# #############################################################################


def _record_planted_game(tmp_path: Path):
    """
    A full game whose replies are all planted: a format error, an illegal move,
    an unknown command, then the fastest mate.
    """
    config = _config()
    pos = cchess.start_positions(1, seed=3, dtz=5, slack_plies=8)[0]
    replies = [
        "no code block here",
        _reply("Qa1"),
        "THOUGHT: x\n\n```mswea_bash_command\nls\n```",
    ]
    replies += [_reply(m) for m in _best_line(pos["fen"])]
    overrides = {
        k + 1: {"content": c, "source": "test"} for k, c in enumerate(replies)
    }
    st = cstore.Store(tmp_path)
    br = cagent.record_run(st, pos["fen"], config=config, overrides=overrides)
    outcome = cagent.run_to_end(br)
    return config, st, br, outcome, len(replies)


def test_checkpoint_after_every_step(tmp_path: Path) -> None:
    _, st, br, outcome, n = _record_planted_game(tmp_path)
    assert outcome["won"] and outcome["n_calls"] == n
    snaps = st.snapshots(br.branch_id)
    assert [s["step"] for s in snaps] == list(range(n + 1))
    assert cstore.verify_tape(st.read_tape(br.branch_id))


def test_replay_from_every_checkpoint_matches(tmp_path: Path) -> None:
    config, st, br, _, n = _record_planted_game(tmp_path)
    snaps = st.snapshots(br.branch_id)
    calls = cutils.model_calls(st, br.branch_id)
    for s in snaps[:-1]:
        k = s["step"] + 1
        iv = {"type": "replay", "source_branch": br.branch_id, "steps": [k]}
        r = cutils.play_fork(tmp_path, s["id"], iv, config, stop_after_calls=k)
        new_calls = cutils.model_calls(st, r["branch_id"])
        assert (
            new_calls[k]["request"]["input_hash"]
            == calls[k]["request"]["input_hash"]
        )
        after = st.snapshots(r["branch_id"])[0]
        orig_after = snaps[k]
        assert after["env_hash"] == orig_after["env_hash"]
        assert cutils.api_view_hash(st, after) == cutils.api_view_hash(
            st, orig_after
        )


def test_fork_with_full_replay_reproduces_the_ending(tmp_path: Path) -> None:
    config, st, br, outcome, n = _record_planted_game(tmp_path)
    snaps = st.snapshots(br.branch_id)
    mid = snaps[2]
    iv = {
        "type": "replay",
        "source_branch": br.branch_id,
        "steps": list(range(mid["step"] + 1, n + 1)),
    }
    r = cutils.play_fork(tmp_path, mid["id"], iv, config)
    assert r["moves_san"] == outcome["moves_san"] and r["won"]
    last = st.snapshot(r["last_snapshot"])
    assert last["env_hash"] == snaps[-1]["env_hash"]
    assert json.loads(st.branch(r["branch_id"])["intervention_json"]) == iv
