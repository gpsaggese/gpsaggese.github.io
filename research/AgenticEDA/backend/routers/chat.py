"""Chat agent endpoints (REST + WebSocket)."""
from __future__ import annotations

import datetime
import json
import logging
import threading
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from src.chat.chat_agent import build_chat_agent

_LOG = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])

# In-memory state cache per session (populated by pipeline run)
_session_states: dict[str, dict] = {}
_session_kgs: dict = {}

def set_session_kg(session_id: str, kg) -> None:
    """Called after pipeline completes to make KG available to chat."""
    _session_kgs[session_id] = kg


def set_session_state(session_id: str, state: dict) -> None:
    """Called after pipeline completes to make state available to chat."""
    _session_states[session_id] = state


@router.websocket("/chat/{session_id}")
async def chat_ws(websocket: WebSocket, session_id: str) -> None:
    """Accept a WebSocket connection and relay messages through the chat agent."""
    await websocket.accept()
    state = _session_states.get(session_id, {})
    # Load KG for chat context
    kg = _session_kgs.get(session_id)
    if kg is None:
        try:
            from backend.services.session_manager import get_session_dir
            from src.agent.knowledge_graph import KnowledgeGraph
            story_path = get_session_dir(session_id) / "story.json"
            if story_path.exists():
                story_data = json.loads(story_path.read_text())
                if "knowledge_graph" in story_data:
                    kg = KnowledgeGraph.from_dict(story_data["knowledge_graph"])
                    _session_kgs[session_id] = kg
        except Exception:
            pass
    agent = build_chat_agent(session_id, state, kg=kg)

    try:
        while True:
            data = await websocket.receive_json()
            user_msg = data.get("content", data.get("message", ""))
            _LOG.info("Chat [%s]: %s", session_id, user_msg[:100])

            response = agent(user_msg)
            await websocket.send_json(response)
    except WebSocketDisconnect:
        _LOG.info("Chat client disconnected for session %s", session_id)


class ChatRequest(BaseModel):
    content: str
    cells: list[dict] | None = None


@router.post("/chat/{session_id}/message")
async def chat_message(session_id: str, req: ChatRequest):
    """REST endpoint for chat — handles questions, actions, and hypothesis investigations."""
    state = {**_session_states.get(session_id, {})}
    # Merge current cells from frontend so the LLM knows what exists
    if req.cells is not None:
        state["cells"] = req.cells

    # Check if this is a hypothesis request (user wants investigation, not just answer)
    try:
        from src.agent.hypothesis import hypothesis_from_user_question
        hyp = hypothesis_from_user_question(
            req.content,
            columns=state.get("columns", []),
            numeric_cols=state.get("numeric_cols", []),
            time_col=state.get("time_col"),
        )
        if hyp is not None:
            return _run_hypothesis_investigation(session_id, state, hyp, req.content)
    except Exception as exc:
        _LOG.warning("Hypothesis detection failed: %s", exc)

    # Load KG for chat context
    kg = _session_kgs.get(session_id)
    if kg is None:
        try:
            from backend.services.session_manager import get_session_dir
            from src.agent.knowledge_graph import KnowledgeGraph
            story_path = get_session_dir(session_id) / "story.json"
            if story_path.exists():
                story_data = json.loads(story_path.read_text())
                if "knowledge_graph" in story_data:
                    kg = KnowledgeGraph.from_dict(story_data["knowledge_graph"])
                    _session_kgs[session_id] = kg
        except Exception:
            pass

    # Normal chat flow
    agent = build_chat_agent(session_id, state, kg=kg)
    response = agent(req.content)
    _LOG.info("Chat REST [%s]: %s -> %s", session_id, req.content[:60], response.get("content", "")[:60])

    # Handle structured actions
    action_type = response.get("action_type")
    if action_type:
        try:
            from backend.services.kernel_manager import execute_code, is_kernel_alive
            from backend.services.session_manager import get_session_dir
            from backend.routers.stream import push_event

            _ensure_kernel(session_id)

            if action_type == "new_cell":
                response = _handle_new_cell(session_id, response, req.content, push_event, execute_code)

            elif action_type == "edit_cell":
                response = _handle_edit_cell(session_id, response, req.content, push_event, execute_code)

            elif action_type == "delete_cells":
                response = _handle_delete_cells(session_id, response, req.content, push_event)

            elif action_type == "reorder_cell":
                response = _handle_reorder_cell(session_id, response, req.content, push_event)

        except Exception as exc:
            _LOG.warning("Chat action failed: %s", exc)
            response["content"] += f"\n\nCouldn't execute action: {exc}"

    return response


def _ensure_kernel(session_id: str):
    """Make sure the kernel is alive and has data loaded."""
    from backend.services.kernel_manager import execute_code, is_kernel_alive
    from backend.services.session_manager import get_session_dir

    if not is_kernel_alive(session_id):
        session_dir = get_session_dir(session_id)
        uploads = list((session_dir / "uploads").iterdir())
        if uploads:
            dataset_path = str(uploads[0])
            load_code = f'import pandas as pd\nimport numpy as np\nimport matplotlib\nmatplotlib.use("Agg")\nimport matplotlib.pyplot as plt\ndf = pd.read_csv("{dataset_path}")\nprint(f"Reloaded {{len(df)}} rows")'
            execute_code(session_id, load_code, timeout=15)


def _execute_and_push(session_id: str, cell_id: str, code: str, push_event, execute_code) -> tuple[list, str | None]:
    """Execute code in kernel and push output/error events. Returns (outputs, error)."""
    push_event(session_id, {"type": "cell_executing", "cell_id": cell_id})
    outputs, error = execute_code(session_id, code, timeout=30)

    if error:
        push_event(session_id, {"type": "cell_error", "cell_id": cell_id, "error": error})
    else:
        push_event(session_id, {"type": "cell_output", "cell_id": cell_id, "outputs": outputs})

    return outputs, error


def _handle_new_cell(session_id, response, user_msg, push_event, execute_code):
    """Create a new cell, execute it, push events."""
    code = response["action_code"]
    cell_id = f"chat_cell_{int(time.time())}"

    push_event(session_id, {
        "type": "cell_write",
        "cell_id": cell_id,
        "cell_type": "code",
        "source": code,
    })

    # Log user action in activity
    push_event(session_id, {
        "type": "chat_action",
        "action": "new_cell",
        "detail": f"New cell: {user_msg[:60]}",
        "cell_id": cell_id,
    })

    outputs, error = _execute_and_push(session_id, cell_id, code, push_event, execute_code)
    response = _append_output_summary(response, outputs, error)
    _snapshot(session_id, f"Chat: {user_msg[:40]}")
    return response


def _handle_edit_cell(session_id, response, user_msg, push_event, execute_code):
    """Edit an existing cell's source and re-execute."""
    cell_id = response["target_cell_id"]
    code = response["action_code"]

    # Push source update
    push_event(session_id, {
        "type": "cell_update",
        "cell_id": cell_id,
        "source": code,
    })

    # Log user action in activity
    push_event(session_id, {
        "type": "chat_action",
        "action": "edit_cell",
        "detail": f"Edited: {user_msg[:60]}",
        "cell_id": cell_id,
    })

    outputs, error = _execute_and_push(session_id, cell_id, code, push_event, execute_code)
    response = _append_output_summary(response, outputs, error)
    _snapshot(session_id, f"Edit: {user_msg[:40]}")
    return response


def _handle_delete_cells(session_id, response, user_msg, push_event):
    """Delete one or more cells."""
    cell_ids = response.get("target_cell_ids", [])
    for cid in cell_ids:
        push_event(session_id, {"type": "cell_delete", "cell_id": cid})

    push_event(session_id, {
        "type": "chat_action",
        "action": "delete_cells",
        "detail": f"Deleted {len(cell_ids)} cell(s): {user_msg[:60]}",
        "cell_ids": cell_ids,
    })

    _snapshot(session_id, f"Delete: {user_msg[:40]}")
    return response


def _handle_reorder_cell(session_id, response, user_msg, push_event):
    """Move a cell up or down."""
    cell_id = response["target_cell_id"]
    direction = response["direction"]

    push_event(session_id, {
        "type": "cell_reorder",
        "cell_id": cell_id,
        "direction": direction,
    })

    push_event(session_id, {
        "type": "chat_action",
        "action": "reorder_cell",
        "detail": f"Moved {direction}: {user_msg[:60]}",
        "cell_id": cell_id,
    })

    _snapshot(session_id, f"Reorder: {user_msg[:40]}")
    return response


def _append_output_summary(response: dict, outputs: list, error: str | None) -> dict:
    """Append execution result summary to the response content."""
    if error:
        response["content"] += f"\n\nError executing code: {error}"
    else:
        output_text = "\n".join(o.get("text", "") for o in outputs if o.get("text"))
        if output_text:
            response["content"] += f"\n\nResult:\n{output_text[:500]}"
        else:
            response["content"] += "\n\nCode executed — check the notebook for the output."
    return response


def _snapshot(session_id: str, label: str):
    """Create a version snapshot, swallowing errors."""
    try:
        from src.reporting.versioning import create_snapshot
        create_snapshot(session_id, label)
    except Exception as exc:
        _LOG.warning("Snapshot failed: %s", exc)


def _run_hypothesis_investigation(session_id: str, state: dict, hyp, user_question: str) -> dict:
    """Run a full subagent investigation for a user hypothesis.

    Returns immediately with an acknowledgement — the actual investigation
    runs in a background process and streams results via WebSocket events.
    """
    from backend.services.kernel_manager import execute_code, is_kernel_alive, get_kernel_connection_file
    from backend.services.session_manager import get_session_dir
    from backend.routers.stream import push_event

    # Check KG for existing answer
    kg = _session_kgs.get(session_id)
    if kg is None:
        try:
            from src.agent.knowledge_graph import KnowledgeGraph
            story_path = get_session_dir(session_id) / "story.json"
            if story_path.exists():
                story_data = json.loads(story_path.read_text())
                if "knowledge_graph" in story_data:
                    kg = KnowledgeGraph.from_dict(story_data["knowledge_graph"])
                    _session_kgs[session_id] = kg
                    _LOG.info("Loaded KG from story.json for session %s", session_id)
        except Exception as exc:
            _LOG.warning("Failed to load KG from story.json: %s", exc)

    if kg is not None:
        existing = kg.find_similar_hypothesis(hyp, threshold=0.8)
        if existing and existing.confidence > 0.85:
            return {
                "role": "agent",
                "type": "text",
                "content": (
                    f"**Previously investigated:** {existing.phase.replace('Investigation: ', '')}\n\n"
                    f"{existing.text}\n\n"
                    f"*Confidence: {existing.confidence:.0%}* (from earlier investigation)"
                ),
                "action_code": None,
            }

    _LOG.info("Running hypothesis investigation for session %s: %s", session_id, hyp.title)

    chat_notebook_id = f"chat_{int(time.time())}"

    # Ensure kernel has data before spawning process
    if not is_kernel_alive(session_id):
        session_dir = get_session_dir(session_id)
        uploads = list((session_dir / "uploads").iterdir())
        if uploads:
            load_code = f'import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\n%matplotlib inline\ndf = pd.read_csv("{uploads[0]}")\nprint(f"Loaded {{len(df)}} rows")'
            execute_code(session_id, load_code, timeout=15)

    conn_file = get_kernel_connection_file(session_id)
    if conn_file is None:
        return {
            "role": "agent",
            "type": "text",
            "content": f"Cannot investigate '{hyp.title}': kernel is not available.",
            "action_code": None,
        }

    # Push start events synchronously so frontend sees the tab immediately
    push_event(session_id, {
        "type": "cell_write",
        "cell_id": f"chat_main_{int(time.time())}",
        "cell_type": "markdown",
        "source": f"---\n\n### Chat Investigation: {hyp.title}\n\n> {hyp.description}\n\n*Spawning investigation subagent...*",
        "notebook_id": "main",
    })
    push_event(session_id, {
        "type": "subagent_start",
        "hypothesis_id": hyp.id,
        "notebook_id": chat_notebook_id,
        "title": hyp.title,
    })
    push_event(session_id, {"type": "phase_transition", "phase": f"Chat Investigation: {hyp.title}", "notebook_id": chat_notebook_id})
    push_event(session_id, {"type": "cell_write", "cell_id": "chat_hyp_header", "cell_type": "markdown", "source": f"### Chat Investigation: {hyp.title}\n\n{hyp.description}", "notebook_id": chat_notebook_id})

    # Run investigation in a background process
    import multiprocessing as mp
    from src.agent.subagent_worker import subagent_process_worker

    event_queue: mp.Queue = mp.Queue()
    result_queue: mp.Queue = mp.Queue()

    p = mp.Process(
        target=subagent_process_worker,
        kwargs=dict(
            connection_file=conn_file,
            hypothesis_id=hyp.id,
            hypothesis_title=hyp.title,
            hypothesis_description=hyp.description,
            relevant_cols=hyp.relevant_cols,
            all_columns=state.get("columns", []),
            time_col=state.get("time_col"),
            session_id=session_id,
            event_queue=event_queue,
            cell_counter_start=int(time.time()) % 10000,
            max_cells=5,
            notebook_id=chat_notebook_id,
            kg_context=kg.get_context_for_hypothesis_generation() if kg else "",
            result_queue=result_queue,
            deadline=time.time() + 240,  # 4 minute deadline for chat investigations
        ),
        daemon=True,
    )
    p.start()

    # Background thread to drain events and collect result
    def _background_finish():
        drain_deadline = time.time() + 300  # 5 min hard cap
        # Drain events from child process → push_event
        while time.time() < drain_deadline:
            try:
                evt = event_queue.get(timeout=2)
                if evt is None:
                    break
                push_event(session_id, evt)
            except Exception:
                if not p.is_alive():
                    break

        # Ensure process is finished; terminate if hung
        p.join(timeout=10)
        if p.is_alive():
            _LOG.warning("Chat investigation process still alive, terminating")
            p.terminate()
            p.join(timeout=5)

        # Collect result
        try:
            status, data = result_queue.get(timeout=5)
        except Exception:
            status, data = "error", "No result from investigation process"

        if status == "ok":
            from src.agent.subagent import InvestigationResult
            # Load images from temp file
            images = {}
            images_file = data.get("_images_file")
            if images_file:
                try:
                    import os as _os
                    with open(images_file) as f:
                        images = json.loads(f.read())
                    _os.remove(images_file)
                except Exception:
                    pass

            result = InvestigationResult(
                hypothesis_id=data["hypothesis_id"],
                hypothesis_title=data["hypothesis_title"],
                finding=data["finding"],
                cell_ids=data.get("cell_ids", []),
                plot_cell_ids=data.get("plot_cell_ids", []),
                confidence=data.get("confidence", 0.5),
                status=data.get("status", "complete"),
                sub_findings=data.get("sub_findings", []),
                relevant_cols=data.get("relevant_cols", []),
                images=images,
            )
            if getattr(result, "status", "complete") == "complete":
                _finalize_chat_investigation(session_id, hyp, chat_notebook_id, result)
            elif getattr(result, "status", "complete") == "timeout":
                push_event(session_id, {
                    "type": "subagent_timeout",
                    "hypothesis_id": hyp.id,
                    "notebook_id": chat_notebook_id,
                })
            else:
                push_event(session_id, {
                    "type": "subagent_complete",
                    "hypothesis_id": hyp.id,
                    "notebook_id": chat_notebook_id,
                    "finding": result.finding,
                    "confidence": result.confidence,
                    "status": "failed",
                })
        else:
            _LOG.warning("Chat investigation failed: %s", data)
            push_event(session_id, {
                "type": "subagent_complete",
                "hypothesis_id": hyp.id,
                "notebook_id": chat_notebook_id,
                "finding": f"Investigation failed: {data}",
                "confidence": 0.0,
                "status": "failed",
            })

    bg = threading.Thread(target=_background_finish, daemon=True)
    bg.start()

    # Return immediately — results will stream via WebSocket
    return {
        "role": "agent",
        "type": "text",
        "content": f"**Investigating: {hyp.title}**\n\nI've started a background investigation. Watch the **{chat_notebook_id}** tab in the notebook for live progress. Results will appear when complete.",
        "action_code": None,
    }


def _finalize_chat_investigation(session_id: str, hyp, chat_notebook_id: str, result) -> None:
    """Called after a background chat investigation completes — writes conclusion, updates KG and story."""
    from backend.services.session_manager import get_session_dir
    from backend.routers.stream import push_event

    conf_pct = int(result.confidence * 100)
    conf_label = "High" if conf_pct >= 70 else "Medium" if conf_pct >= 40 else "Low"

    push_event(session_id, {
        "type": "cell_write",
        "cell_id": f"{chat_notebook_id}_conclusion",
        "cell_type": "markdown",
        "source": f"### Conclusion\n\n{result.finding}\n\n**Confidence:** {conf_label} ({conf_pct}%)",
        "notebook_id": chat_notebook_id,
    })

    # Add to KG
    kg = _session_kgs.get(session_id)
    if kg is not None:
        nid = kg.add_investigation(
            hypothesis_id=hyp.id,
            hypothesis_title=hyp.title,
            finding=result.finding,
            evidence_cells=result.cell_ids,
            plot_cells=result.plot_cell_ids,
            confidence=result.confidence,
            sub_findings=result.sub_findings,
            columns=getattr(result, 'relevant_cols', []) or [],
            analysis_type="chat_investigation",
        )
        all_images = []
        for cid, imgs in getattr(result, 'images', {}).items():
            for img in imgs:
                all_images.append({"cell_id": cid, "image_png": img})
        if all_images:
            node = kg.nodes.get(nid)
            if node:
                node.metadata["plot_images"] = all_images
        _LOG.info("Chat investigation added to KG: %s (confidence=%.2f)", hyp.title, result.confidence)

    # Write result to main notebook
    push_event(session_id, {
        "type": "cell_write",
        "cell_id": f"chat_result_{int(time.time())}",
        "cell_type": "markdown",
        "source": f"**Chat Investigation Result: {hyp.title}**\n\n{result.finding}\n\n*Confidence: {conf_label} ({conf_pct}%)*",
        "notebook_id": "main",
    })

    # Append to story.json (not replace)
    try:
        session_dir = get_session_dir(session_id)
        story_path = session_dir / "story.json"
        if story_path.exists():
            story = json.loads(story_path.read_text())
            # Check for contradictions — if new finding contradicts existing, mark old as superseded
            existing_sections = story.get("sections", [])
            for sec in existing_sections:
                if sec.get("type") == "investigation" and sec.get("title") == hyp.title:
                    sec["superseded"] = True
                    sec["superseded_by"] = result.finding
            story.setdefault("sections", []).append({
                "phase": f"Investigation: {hyp.title}",
                "title": hyp.title,
                "content": result.finding,
                "cell_ids": result.cell_ids,
                "plot_cell_ids": result.plot_cell_ids,
                "type": "investigation",
                "confidence": result.confidence,
            })
            story["generated_at"] = datetime.datetime.now().isoformat()
            # Persist KG too
            if kg is not None:
                story["knowledge_graph"] = kg.to_dict()
            from backend.routers.story import atomic_write_json
            atomic_write_json(story_path, story)

        from src.reporting.versioning import create_snapshot
        create_snapshot(session_id, f"Investigation: {hyp.title[:40]}")
    except Exception as exc:
        _LOG.warning("Story update failed for investigation %s: %s", hyp.title, exc)

    push_event(session_id, {
        "type": "subagent_complete",
        "hypothesis_id": hyp.id,
        "notebook_id": chat_notebook_id,
        "finding": result.finding,
        "confidence": result.confidence,
        "status": "complete",
    })
