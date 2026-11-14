"""
The chess task: King and Rook vs. King endgames, judged by the Syzygy tablebase.

- `ChessEnv`: the agent's environment. The agent's only commands are
  `play show` and `play move <move>`; anything else is refused without running
  anything, so the environment is sealed by construction and needs no
  container. Its whole state is the game record, a small JSON document.
- Black is a fixed function of the position (best tablebase defence), so the
  environment is deterministic.
- `referee`: the true win/draw value of every position in a game and the pivot
  (the first White move that throws the win away).
- `start_positions`: seeded starting positions with a known distance to mate.

Value under the fifty-move rule: White (to move) wins iff WDL == +2 and
clock + DTZ <= 100. In KRvK the only zeroing move for White is mate, so DTZ is
the exact distance to mate in plies.

Import as:

import caaf_chess as cchess
"""

import json
import random
import shlex
import time
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chess
import chess.syzygy

import caaf_store as cstore

_TB_DIR = Path(__file__).resolve().parent / "data" / "syzygy"

# Printed by the board when the game ends; the agent stops on it.
GAME_OVER = "GAME OVER:"

_USAGE = (
    "Unknown command. The only commands are `play show` and"
    " `play move <move>` (SAN such as Rh7 or Kd6, or UCI such as h1h7)."
)


@lru_cache(maxsize=1)
def tablebase() -> chess.syzygy.Tablebase:
    if not (_TB_DIR / "KRvK.rtbz").exists():
        raise FileNotFoundError(
            f"KRvK tables missing in {_TB_DIR}: run data/fetch_tablebases.sh"
        )
    return chess.syzygy.open_tablebase(str(_TB_DIR))


# #############################################################################
# Tablebase values
# #############################################################################


def white_value(b: chess.Board) -> Dict:
    """
    Value of a position with White to move (or a finished position) under the
    fifty-move rule.
    """
    if b.is_checkmate():
        return {
            "wdl": 2 if b.turn == chess.BLACK else -2,
            "dtz": 0,
            "clock": b.halfmove_clock,
            "win": b.turn == chess.BLACK,
            "terminal": "checkmate",
        }
    for name, done in (
        ("stalemate", b.is_stalemate()),
        ("insufficient_material", b.is_insufficient_material()),
        ("fifty_move_rule", b.halfmove_clock >= 100),
    ):
        if done:
            return {
                "wdl": 0,
                "dtz": 0,
                "clock": b.halfmove_clock,
                "win": False,
                "terminal": name,
            }
    assert b.turn == chess.WHITE
    wdl, dtz = tablebase().probe_wdl(b), tablebase().probe_dtz(b)
    return {
        "wdl": wdl,
        "dtz": dtz,
        "clock": b.halfmove_clock,
        "win": wdl == 2 and b.halfmove_clock + dtz <= 100,
        "terminal": None,
    }


def defender_choice(b: chess.Board) -> chess.Move:
    """
    Black's reply: minimise White's WDL, then maximise DTZ (longest resistance,
    also the best defence against the fifty-move rule), ties broken by the
    alphabetically first UCI string.
    """
    scored = []
    for m in b.legal_moves:
        b.push(m)
        if b.is_insufficient_material() or b.is_stalemate():
            key: Tuple[int, int] = (-3, 0)
        else:
            key = (tablebase().probe_wdl(b), -tablebase().probe_dtz(b))
        b.pop()
        scored.append((key, m.uci(), m))
    scored.sort(key=lambda x: (x[0], x[1]))
    return scored[0][2]


def referee(start_fen: str, moves_uci: List[str]) -> Dict:
    """
    Replay a game and return the value before each White move and the pivot.

    The pivot is the first White move after which a won position is no longer
    won against Black's best defence (which our defender plays).
    """
    b = chess.Board(start_fen)
    rows = [{"ply": 0, "fen": b.fen(), **white_value(b)}]
    i = 0
    while i < len(moves_uci) and not rows[-1]["terminal"]:
        # White.
        b.push_uci(moves_uci[i])
        i += 1
        if i < len(moves_uci):
            # Black.
            b.push_uci(moves_uci[i])
            i += 1
        elif not (b.is_game_over() or b.halfmove_clock >= 100):
            b.push(defender_choice(b))
        rows.append({"ply": len(b.move_stack), "fen": b.fen(), **white_value(b)})
    pivot = None
    for k in range(1, len(rows)):
        if rows[k - 1]["win"] and not rows[k]["win"]:
            pivot = k
            break
    final = rows[-1]
    return {
        "positions": rows,
        "pivot_white_move": pivot,
        "started_winning": rows[0]["win"],
        "won": final["terminal"] == "checkmate" and final["wdl"] == 2,
        "end": final["terminal"] or "unfinished",
    }


def start_positions(n: int, seed: int, dtz: int, slack_plies: int) -> List[Dict]:
    """
    Seeded White-to-move KRvK starts that are `dtz` plies from mate.

    The clock is set so White has exactly `slack_plies` spare half-moves beyond
    perfect play before the fifty-move rule draws the game.
    """
    rng = random.Random(seed)
    out: List[Dict] = []
    seen = set()
    while len(out) < n:
        sq = rng.sample(range(64), 3)
        b = chess.Board(None)
        b.set_piece_at(sq[0], chess.Piece(chess.KING, chess.WHITE))
        b.set_piece_at(sq[1], chess.Piece(chess.KING, chess.BLACK))
        b.set_piece_at(sq[2], chess.Piece(chess.ROOK, chess.WHITE))
        b.turn = chess.WHITE
        if not b.is_valid() or b.is_game_over() or b.board_fen() in seen:
            continue
        if tablebase().probe_wdl(b) != 2 or tablebase().probe_dtz(b) != dtz:
            continue
        b.halfmove_clock = 100 - dtz - slack_plies
        seen.add(b.board_fen())
        out.append({"fen": b.fen(), "dtz": dtz, "slack_plies": slack_plies})
    return out


# #############################################################################
# Board
# #############################################################################


def new_game(start_fen: str) -> Dict:
    return {
        "start_fen": start_fen,
        "moves_uci": [],
        "moves_san": [],
        "status": "ongoing",
        "result": None,
    }


def board_of(game: Dict) -> chess.Board:
    b = chess.Board(game["start_fen"])
    for u in game["moves_uci"]:
        b.push_uci(u)
    return b


def _status_of(b: chess.Board) -> Tuple[str, Optional[str]]:
    if b.is_checkmate():
        return "checkmate", "1-0" if b.turn == chess.BLACK else "0-1"
    if b.is_stalemate():
        return "stalemate", "1/2-1/2"
    if b.is_insufficient_material():
        return "insufficient_material", "1/2-1/2"
    if b.halfmove_clock >= 100:
        return "fifty_move_rule", "1/2-1/2"
    return "ongoing", None


def render(game: Dict) -> str:
    b = board_of(game)
    lines = ["Board (White = uppercase, Black = lowercase):"]
    for rank in range(7, -1, -1):
        row = [str(b.piece_at(chess.square(f, rank)) or ".") for f in range(8)]
        lines.append(f"{rank + 1}  " + " ".join(row))
    lines.append("   a b c d e f g h")
    lines.append(f"FEN: {b.fen()}")
    if game["status"] == "ongoing":
        left = 100 - b.halfmove_clock
        lines.append(
            f"White to move. Checkmate must happen within {left} half-moves"
            f" ({(left + 1) // 2} of your moves), otherwise the game is drawn"
            " by the fifty-move rule."
        )
        legal = sorted(b.san(m) for m in b.legal_moves)
        lines.append("Legal moves: " + " ".join(legal))
    else:
        lines.append(f"{GAME_OVER} {game['status']} ({game['result']})")
    return "\n".join(lines)


def play(game: Dict, argv: List[str]) -> Tuple[Dict, str, int]:
    """
    Run one board command; return the new game, the output and the exit code.

    The game passed in is never modified.
    """
    game = json.loads(json.dumps(game))
    if not argv or argv[0] != "play" or len(argv) < 2:
        return game, _USAGE, 127
    if argv[1] == "show" and len(argv) == 2:
        return game, render(game), 0
    if argv[1] != "move" or len(argv) != 3:
        return game, _USAGE, 127
    if game["status"] != "ongoing":
        return game, render(game), 1
    b = board_of(game)
    text = argv[2].strip()
    try:
        mv = b.parse_san(text)
    except ValueError:
        try:
            mv = chess.Move.from_uci(text)
            if mv not in b.legal_moves:
                raise ValueError
        except ValueError:
            out = (
                f"Illegal or unreadable move: {text!r}. The board is unchanged."
            )
            return game, out + "\n" + render(game), 1
    san = b.san(mv)
    b.push(mv)
    game["moves_uci"].append(mv.uci())
    game["moves_san"].append(san)
    msg = [f"You played {san}."]
    st, res = _status_of(b)
    if st == "ongoing":
        reply = defender_choice(b)
        rsan = b.san(reply)
        b.push(reply)
        game["moves_uci"].append(reply.uci())
        game["moves_san"].append(rsan)
        msg.append(f"Black replied {rsan}.")
        st, res = _status_of(b)
    game["status"], game["result"] = st, res
    return game, " ".join(msg) + "\n" + render(game), 0


# #############################################################################
# ChessEnv
# #############################################################################


class ChessEnv:
    """
    mini-swe-agent environment whose only program is the board.

    Implements the parts of mini-swe-agent's `Environment` protocol that the
    agent uses: `execute`, `get_template_vars` and `serialize`.
    """

    def __init__(self, start_fen: str) -> None:
        self.game = new_game(start_fen)
        self.last_duration = 0.0

    def execute(
        self, action: Dict, cwd: str = "", *, timeout: Optional[int] = None
    ) -> Dict:
        t0 = time.monotonic()
        command = action.get("command", "")
        try:
            argv = shlex.split(command)
        except ValueError:
            argv = []
        self.game, out, code = play(self.game, argv)
        self.last_duration = time.monotonic() - t0
        return {"output": out + "\n", "returncode": code, "exception_info": ""}

    def get_template_vars(self, **kwargs) -> Dict:
        return dict(kwargs)

    def serialize(self) -> Dict:
        return {"info": {"config": {"environment_type": "caaf_chess.ChessEnv"}}}

    # ---- environment half of a checkpoint

    def state(self) -> bytes:
        return cstore.canonical(self.game)

    def restore(self, data: bytes, expect_hash: str) -> None:
        """
        Load a saved game and verify it by re-serializing.
        """
        self.game = json.loads(data)
        got = cstore.sha256(self.state())
        if got != expect_hash:
            raise RuntimeError(
                f"restored environment {got} != saved {expect_hash}"
            )
