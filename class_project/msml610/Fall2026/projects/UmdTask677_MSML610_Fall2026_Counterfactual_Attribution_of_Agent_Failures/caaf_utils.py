"""
Experiment helpers: parallel forks, checkpoint lookups and statistics.

Import as:

import caaf_utils as cutils
"""

import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import caaf_agent as cagent
import caaf_store as cstore


def parallel(fn: Callable, items: List, workers: int = 8) -> List:
    """
    Run `fn` over `items` in threads, in submission order, keeping that order.
    """
    out: Dict[int, Any] = {}
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(fn, it): i for i, it in enumerate(items)}
        for f in as_completed(futs):
            out[futs[f]] = f.result()
    return [out[i] for i in range(len(items))]


def play_new_game(store_root: Path, fen: str, config: Dict, label: str) -> Dict:
    st = cstore.Store(store_root)
    br = cagent.record_run(st, fen, config=config, label=label)
    out = cagent.run_to_end(br)
    return {
        "run_id": br.run_id,
        "branch_id": br.branch_id,
        "start_fen": fen,
        **out,
    }


def play_fork(
    store_root: Path,
    snapshot_id: str,
    intervention: Dict,
    config: Dict,
    *,
    stop_after_calls: Optional[int] = None,
    attempts: int = 3,
) -> Dict:
    """
    Fork and play to the end.

    An infrastructure failure (API error) marks that branch `failed` and retries
    with a new branch, so failures are never counted as game outcomes.
    """
    st = cstore.Store(store_root)
    for attempt in range(attempts):
        br = None
        try:
            br = cagent.fork(
                st, snapshot_id, config=config, intervention=intervention
            )
            out = cagent.run_to_end(br, stop_after_calls=stop_after_calls)
            return {
                "branch_id": br.branch_id,
                "parent": snapshot_id,
                "attempt": attempt + 1,
                **br.info,
                **out,
            }
        except Exception as e:
            if br is not None:
                err = f"{type(e).__name__}: {e}"[:500]
                st.finish_branch(br.branch_id, "failed", {"error": err})
            if attempt == attempts - 1:
                raise
            time.sleep(20 * (attempt + 1))
    raise AssertionError("unreachable")


def root_branches(st: cstore.Store) -> List[Dict]:
    return st.query(
        "select b.branch_id, b.run_id, b.outcome_json, r.config_json"
        " from branches b join runs r using(run_id)"
        " where b.parent_snapshot_id is null and b.status='finished'"
        " order by b.created"
    )


def model_calls(st: cstore.Store, branch_id: str) -> Dict[int, Dict]:
    tape = st.read_tape(branch_id)
    return {e["step"]: e for e in tape if e["kind"] == "model_call"}


def api_view_hash(st: cstore.Store, snapshot: Dict) -> str:
    """
    Hash of the conversation exactly as the model would receive it.
    """
    ctx = json.loads(st.get_blob("context", snapshot["context_hash"]))
    return cstore.sha256(cstore.canonical(cagent.api_messages(ctx["messages"])))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n")


# #############################################################################
# Statistics (no SciPy needed)
# #############################################################################


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """
    95% Wilson score interval for a proportion.
    """
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def newcombe_diff(k1: int, n1: int, k2: int, n2: int) -> Tuple[float, float]:
    """
    95% CI for p1 - p2 (Newcombe hybrid score interval).
    """
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    d = p1 - p2
    lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return (lo, hi)


def fisher_two_sided(k1: int, n1: int, k2: int, n2: int) -> float:
    total_k, total_n = k1 + k2, n1 + n2

    def pmf(x: int) -> float:
        return (
            math.comb(n1, x)
            * math.comb(n2, total_k - x)
            / math.comb(total_n, total_k)
        )

    p0 = pmf(k1)
    xs = range(max(0, total_k - n2), min(total_k, n1) + 1)
    return min(1.0, sum(pmf(x) for x in xs if pmf(x) <= p0 * (1 + 1e-9)))
