"""Process-safe subagent worker — runs in a child process, communicates via queues."""
from __future__ import annotations

import logging
import multiprocessing as mp
from dataclasses import asdict
from typing import Any

_LOG = logging.getLogger(__name__)


def subagent_process_worker(
    *,
    connection_file: str,
    hypothesis_id: str,
    hypothesis_title: str,
    hypothesis_description: str,
    relevant_cols: list[str],
    all_columns: list[str],
    time_col: str | None,
    session_id: str,
    event_queue: mp.Queue,
    cell_counter_start: int,
    max_cells: int,
    notebook_id: str,
    kg_context: str,
    result_queue: mp.Queue,
    run_guidance: str = "",
    deadline: float = 0,
) -> None:
    """Top-level function that runs in a child process.

    Connects to the kernel via connection_file, runs the subagent,
    sends events via event_queue, puts the InvestigationResult dict on result_queue.
    """
    from backend.services.kernel_manager import PersistentKernelConnection
    from src.agent.subagent import run_subagent

    cell_counter = [cell_counter_start]

    def push_event_via_queue(sid: str, event: dict) -> None:
        event_queue.put(event)

    # Create a persistent connection once — avoids ZMQ slow-joiner problem
    # where iopub messages are lost when creating a new client per cell.
    conn = PersistentKernelConnection(connection_file)

    def execute_code_via_connection(
        sid: str, code: str, timeout: int = 60, cell_id: str | None = None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        return conn.execute(code, timeout=timeout, cell_id=cell_id)

    try:
        result = run_subagent(
            hypothesis_id=hypothesis_id,
            hypothesis_title=hypothesis_title,
            hypothesis_description=hypothesis_description,
            relevant_cols=relevant_cols,
            all_columns=all_columns,
            time_col=time_col,
            session_id=session_id,
            push_event=push_event_via_queue,
            execute_code=execute_code_via_connection,
            cell_counter=cell_counter,
            max_cells=max_cells,
            notebook_id=notebook_id,
            kg_context=kg_context,
            run_guidance=run_guidance,
            deadline=deadline,
        )
        # Convert to dict for pickling across process boundary.
        # IMPORTANT: images (base64 strings, potentially MBs) are written to a
        # temp file instead of the Queue pipe to avoid macOS pipe buffer deadlock.
        result_dict = result.to_dict()
        result_dict["relevant_cols"] = result.relevant_cols
        result_dict["cell_sources"] = result.cell_sources
        result_dict["cell_outputs"] = result.cell_outputs

        if result.images:
            import json as _json, tempfile, os
            images_file = os.path.join(
                tempfile.gettempdir(),
                f"agenticeda_images_{hypothesis_id}_{os.getpid()}.json",
            )
            with open(images_file, "w") as f:
                _json.dump({k: list(v) for k, v in result.images.items()}, f)
            result_dict["_images_file"] = images_file
        else:
            result_dict["_images_file"] = None

        result_queue.put(("ok", result_dict))
    except Exception as exc:
        _LOG.warning("Subagent process for %s failed: %s", hypothesis_id, exc)
        result_queue.put(("error", str(exc)))
    finally:
        # Signal no more events from this worker
        event_queue.put(None)
        try:
            conn.close()
        except Exception:
            pass
