"""Run router: trigger the real-time EDA agent for a session."""
from __future__ import annotations

import json
import logging
import os
import pathlib
import threading

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import nbformat

from backend.services.session_manager import get_session_dir
from backend.services.steering_service import (
    clear_steering,
    enqueue_steering,
    get_steering_items,
)
from backend.routers.stream import push_event
from src.reporting.plot_contract import (
    plot_artifacts_from_outputs,
    plot_specs_by_cell,
)

from pydantic import BaseModel

class RunConfig(BaseModel):
    max_subagents: int = 3
    max_loops: int = 2
    loop_timeout: int = 300
    seed: int | None = None
    research_direction: str | None = None


class SteeringRequest(BaseModel):
    content: str
    message_id: str | None = None

router = APIRouter(tags=["run"])
_LOG = logging.getLogger(__name__)

# Track running agents per session
_running: dict[str, threading.Thread] = {}


def _run_agent_in_thread(session_id: str, dataset_path: str, session_dir: pathlib.Path,
                         max_subagents: int = 3, max_loops: int = 2, loop_timeout: int = 300,
                         seed: int | None = None,
                         research_direction: str | None = None):
    """Run the EDA agent loop in a background thread, streaming all events."""
    try:
        (session_dir / "status.json").write_text(
            json.dumps({"status": "running", "phase": "starting"})
        )

        from src.agent.eda_agent import run_agent
        state = run_agent(
            session_id=session_id,
            dataset_path=dataset_path,
            push_event=push_event,
            max_subagents=max_subagents,
            max_loops=max_loops,
            loop_timeout=loop_timeout,
            seed=seed,
            research_direction=research_direction,
        )

        # Save state summary
        (session_dir / "agent_state.json").write_text(
            json.dumps({
                "findings": state.findings,
                "errors": state.errors_encountered,
                "phases": state.phases_completed,
                "row_count": state.row_count,
                "col_count": state.col_count,
                "time_col": state.time_col,
                "numeric_cols": state.numeric_cols,
                "summary": state.summarize(),
            }, default=str, indent=2)
        )

        # Save notebook.ipynb from accumulated cell data
        try:
            nb = nbformat.v4.new_notebook()
            for entry in state.cell_registry:
                if entry["cell_type"] == "markdown":
                    cell = nbformat.v4.new_markdown_cell(entry["source"])
                else:
                    cell = nbformat.v4.new_code_cell(entry["source"])
                    cell.outputs = [nbformat.v4.new_output(**o) for o in (entry.get("outputs") or [])]
                cell.id = entry["id"]
                nb.cells.append(cell)
            nbformat.write(nb, str(session_dir / "notebook.ipynb"))
            _LOG.info("Notebook saved: %d cells for session %s", len(nb.cells), session_id)
        except Exception as exc:
            _LOG.warning("Notebook save failed for session %s: %s", session_id, exc)

        # Generate story from knowledge graph (if available) or findings
        def _report_plan(details: list[dict]):
            """Emit plan_update with report generation progress."""
            completed_phases = [{"phase": p, "status": "complete"} for p in state.phases_completed]
            push_event(session_id, {"type": "plan_update", "steps": completed_phases + [
                {"phase": "Deep Investigation", "status": "complete"},
                {"phase": "Report Generation", "status": "current", "details": details},
            ]})

        _report_plan([{"label": "Building sections", "status": "current"}])
        try:
            import datetime
            plot_specs_map = plot_specs_by_cell(session_dir)

            # Use knowledge graph if the investigation phase ran
            kg = getattr(state, "knowledge_graph", None)
            if kg is not None:
                sections = kg.get_story_sections()
                conclusions = kg.get_top_conclusions(5)
            else:
                # Fallback: group findings by phase
                phase_findings: dict[str, list[str]] = {}
                for f in state.findings:
                    phase_findings.setdefault(f.get("phase", ""), []).append(f.get("finding", ""))
                sections = []
                for phase, flist in phase_findings.items():
                    phase_cells = [cid for cid, p in state.cell_phases.items() if p == phase]
                    sections.append({
                        "phase": phase,
                        "title": phase,
                        "content": "\n".join(f"- {f}" for f in flist),
                        "cell_ids": phase_cells,
                        "plot_cell_ids": phase_cells,
                        "plots": [],
                    })
                conclusions = [f.get("finding", "") for f in state.findings[:5]]

            # Ensure sections have cell_ids from the agent's cell_phases map
            for section in sections:
                phase = section.get("phase", "")
                if not section.get("cell_ids"):
                    phase_cells = [cid for cid, p in state.cell_phases.items() if p == phase]
                    if phase_cells:
                        section["cell_ids"] = phase_cells
                if not section.get("plot_cell_ids"):
                    section["plot_cell_ids"] = section.get("cell_ids", [])

            # Attach plot artifacts from the generated notebook whenever possible.
            nb_path = session_dir / "notebook.ipynb"
            if nb_path.exists():
                try:
                    nb = nbformat.read(str(nb_path), as_version=4)
                    cell_map = {}
                    for cell in nb.cells:
                        cell_id = cell.get("id")
                        if cell_id:
                            cell_map[str(cell_id)] = cell
                    for section in sections:
                        plot_cell_ids = section.get("plot_cell_ids") or section.get("cell_ids") or []
                        plots: list[dict] = []
                        for cell_id in plot_cell_ids:
                            cell = cell_map.get(str(cell_id))
                            if not cell:
                                continue
                            plots.extend(
                                plot_artifacts_from_outputs(
                                    cell.get("outputs", []) or [],
                                    title=section.get("title", ""),
                                    caption=section.get("visual_caption", ""),
                                    source_cell_id=str(cell_id),
                                    plot_specs=plot_specs_map.get(str(cell_id), []),
                                )
                            )
                        if plots:
                            section["plots"] = plots

                        # Fallback for investigation plots: extract from KG metadata
                        if not plots and section.get("type") == "investigation":
                            try:
                                kg_data = kg.to_dict() if kg else {}
                                for nid_key, node_data in kg_data.get("nodes", {}).items():
                                    if (node_data.get("type") == "conclusion" and
                                        node_data.get("phase", "").replace("Investigation: ", "") == section.get("title", "")):
                                        plot_images = node_data.get("metadata", {}).get("plot_images", [])
                                        for pi in plot_images:
                                            plots.append({
                                                "kind": "image",
                                                "mime_type": "image/png",
                                                "source": pi["image_png"],
                                                "title": section.get("title", ""),
                                                "caption": f"Investigation plot for {section.get('title', '')}",
                                                "source_cell_id": pi.get("cell_id", ""),
                                            })
                                        break
                            except Exception as plot_exc:
                                _LOG.warning("KG plot extraction failed: %s", plot_exc)
                            if plots:
                                section["plots"] = plots
                except Exception as plot_exc:
                    _LOG.warning("Plot artifact bridge failed for session %s: %s", session_id, plot_exc)

            _report_plan([
                {"label": "Building sections", "status": "complete"},
                {"label": "Extracting plots", "status": "complete"},
                {"label": "Writing executive summary", "status": "current"},
            ])
            # LLM narrative for executive summary
            narrative = ""
            try:
                from src.config.config import get_chat_model
                from langchain_core.messages import SystemMessage, HumanMessage
                llm = get_chat_model()

                conclusions_text = "\n".join(f"- {c}" for c in conclusions)
                findings_text = "\n".join(
                    f"- [{f.get('phase', '')}] {f.get('finding', '')}" for f in state.findings
                )

                kg_context = ""
                if kg is not None:
                    kg_context = kg.get_context_for_hypothesis_generation()

                resp = llm.invoke([
                    SystemMessage(content="Write 2-3 paragraphs of flowing prose for an EDA report executive summary. Describe: what the data contains, key patterns, notable anomalies, investigated hypotheses and their conclusions, and recommended next steps. Be specific with numbers. Do NOT use bullet points. Never use emojis or special unicode symbols."),
                    HumanMessage(content=f"Dataset: {os.path.basename(dataset_path)}\n{state.row_count} rows x {state.col_count} cols\nColumns: {', '.join(state.columns[:15])}\nTime column: {state.time_col}\n\nKnowledge graph context:\n{kg_context[:2000]}\n\nTop conclusions:\n{conclusions_text}\n\nAll findings:\n{findings_text[:3000]}"),
                ])
                narrative = resp.content.strip()
            except Exception as llm_exc:
                _LOG.warning("LLM narrative failed: %s", llm_exc)
                narrative = "\n".join(f"- {c}" for c in conclusions)

            _report_plan([
                {"label": "Building sections", "status": "complete"},
                {"label": "Extracting plots", "status": "complete"},
                {"label": "Writing executive summary", "status": "complete"},
                {"label": "Curating plots and captions", "status": "current"},
            ])
            from src.reporting.story_builder import build_curated_story

            story_data = build_curated_story(
                sections=sections,
                executive_summary=narrative,
                dataset_name=os.path.basename(dataset_path),
                max_plots_per_section=3,
            )
            if kg is not None:
                story_data["knowledge_graph"] = kg.to_dict()

            _report_plan([
                {"label": "Building sections", "status": "complete"},
                {"label": "Extracting plots", "status": "complete"},
                {"label": "Writing executive summary", "status": "complete"},
                {"label": "Curating plots and captions", "status": "complete"},
                {"label": "Saving report", "status": "current"},
            ])
            from backend.routers.story import atomic_write_json
            atomic_write_json(session_dir / "story.json", story_data)
            _LOG.info("Story written: %d sections, narrative %d chars, kg=%s",
                      len(sections), len(narrative), kg is not None)
            # Mark report generation complete
            completed_phases = [{"phase": p, "status": "complete"} for p in state.phases_completed]
            push_event(session_id, {"type": "plan_update", "steps": completed_phases + [
                {"phase": "Deep Investigation", "status": "complete"},
                {"phase": "Report Generation", "status": "complete"},
            ]})
        except Exception as exc:
            _LOG.exception("Story generation failed: %s", exc)

        # Create version snapshot
        try:
            from src.reporting.versioning import create_snapshot
            create_snapshot(session_id, "Initial EDA run")
            _LOG.info("Version snapshot created for session %s", session_id)
        except Exception as exc:
            _LOG.warning("Version snapshot failed: %s", exc)

        # Update chat agent with findings
        try:
            from backend.routers.chat import set_session_state
            chat_state = {
                "findings": state.findings,
                "insights": state.findings,
                "time_col": state.time_col,
                "numeric_cols": state.numeric_cols,
                "categorical_cols": state.categorical_cols,
                "columns": state.columns,
                "dtypes": state.dtypes,
                "row_count": state.row_count,
                "col_count": state.col_count,
                "phases_completed": state.phases_completed,
                "decision_summary": {"summary": state.summarize()},
                "cell_phases": state.cell_phases,
            }
            set_session_state(session_id, chat_state)
            if kg is not None:
                from backend.routers.chat import set_session_kg
                set_session_kg(session_id, kg)
            _LOG.info("Chat state set for session %s: %d findings, %d numeric cols",
                      session_id, len(state.findings), len(state.numeric_cols))
        except Exception as exc:
            _LOG.warning("Chat state update failed: %s", exc)

        (session_dir / "status.json").write_text(
            json.dumps({"status": "completed", "phases_run": state.phases_completed})
        )

    except Exception as exc:
        _LOG.exception("Agent failed for session %s", session_id)
        (session_dir / "status.json").write_text(
            json.dumps({"status": "failed", "error": str(exc)})
        )
        push_event(session_id, {"type": "complete", "summary": f"Error: {exc}"})
    finally:
        _running.pop(session_id, None)
        # Kernel intentionally kept alive for chat-driven code execution


@router.post("/run/{session_id}", status_code=202)
async def run_pipeline(session_id: str, config: RunConfig = RunConfig()):
    """Kick off the real-time EDA agent in a background thread."""
    if session_id in _running and _running[session_id].is_alive():
        return JSONResponse(
            status_code=409,
            content={"status": "conflict", "message": "Agent already running"},
        )

    try:
        session_dir = get_session_dir(session_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")

    # Find the uploaded dataset
    uploads_dir = session_dir / "uploads"
    dataset_files = list(uploads_dir.iterdir()) if uploads_dir.is_dir() else []
    if not dataset_files:
        raise HTTPException(status_code=400, detail="No dataset uploaded")

    dataset_path = str(dataset_files[0])
    clear_steering(session_id)
    thread = threading.Thread(
        target=_run_agent_in_thread,
        args=(session_id, dataset_path, session_dir),
        kwargs={
            "max_subagents": config.max_subagents,
            "max_loops": config.max_loops,
            "loop_timeout": config.loop_timeout,
            "seed": config.seed,
            "research_direction": config.research_direction,
        },
        daemon=True,
    )
    _running[session_id] = thread
    thread.start()

    return JSONResponse(
        status_code=202,
        content={"status": "accepted", "session_id": session_id},
    )


@router.post("/run/{session_id}/steering")
async def enqueue_run_steering(session_id: str, req: SteeringRequest):
    """Queue user steering for a currently running agent."""
    try:
        get_session_dir(session_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")

    content = req.content.strip()
    if not content:
        raise HTTPException(status_code=400, detail="Steering content cannot be blank")

    thread = _running.get(session_id)
    if thread is None or not thread.is_alive():
        raise HTTPException(status_code=409, detail="Agent is not running")

    item = enqueue_steering(session_id, content, message_id=req.message_id)
    push_event(session_id, {
        "type": "steering_queued",
        "message_id": item["id"],
        "content": item["content"],
        "status": item["status"],
    })
    return item


@router.get("/run/{session_id}/steering")
async def get_run_steering(session_id: str):
    """Return steering items for a session."""
    try:
        get_session_dir(session_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")

    return {"items": get_steering_items(session_id)}


@router.get("/run/{session_id}/status")
async def pipeline_status(session_id: str):
    """Check the agent run status for a session."""
    try:
        session_dir = get_session_dir(session_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")

    status_file = session_dir / "status.json"
    if not status_file.exists():
        return {"status": "idle"}

    return json.loads(status_file.read_text())


@router.get("/run/{session_id}/plan")
async def get_plan(session_id: str):
    """Return the current execution plan for a session."""
    try:
        session_dir = get_session_dir(session_id)
    except FileNotFoundError:
        return {"steps": []}

    plan_file = session_dir / "plan.json"
    if not plan_file.exists():
        return {"steps": []}

    try:
        return json.loads(plan_file.read_text())
    except (json.JSONDecodeError, OSError):
        return {"steps": []}
