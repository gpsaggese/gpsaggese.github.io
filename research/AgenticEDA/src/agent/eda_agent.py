"""Main EDA agent loop — writes and executes cells in real time."""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Callable

from src.agent.state import AgentState
from src.agent.goals import build_goal_checklist, EDAGoal
from src.agent import code_templates as ct
from src.agent.reasoning import decide_next_step, interpret_output

_LOG = logging.getLogger(__name__)


def run_agent(
    session_id: str,
    dataset_path: str,
    push_event: Callable[[str, dict], None],
    max_subagents: int = 3,
    max_loops: int = 2,
    loop_timeout: int = 180,
    seed: int | None = None,
    research_direction: str | None = None,
):
    """
    Run the EDA agent loop.

    The agent writes notebook cells one at a time, executes them via the kernel,
    reads outputs, and decides what to do next. All actions are streamed as events.

    :param session_id: session identifier
    :param dataset_path: path to the uploaded dataset file
    :param push_event: callback to push streaming events to the frontend
    :param seed: optional random seed for reproducibility
    """
    if seed is not None:
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        from src.config.config import set_llm_seed
        set_llm_seed(seed)

    from backend.services.kernel_manager import execute_code, shutdown_kernel

    state = AgentState(
        dataset_path=dataset_path,
        session_id=session_id,
        research_direction=research_direction.strip() if research_direction else None,
    )
    goals = build_goal_checklist()
    filename = os.path.basename(dataset_path)

    # Detect format from extension
    ext = os.path.splitext(filename)[1].lower().lstrip(".")
    fmt_map = {"csv": "csv", "tsv": "tsv", "xlsx": "xlsx", "xls": "xls",
               "json": "json", "jsonl": "json", "parquet": "parquet", "pq": "parquet"}
    file_format = fmt_map.get(ext, "csv")

    current_phase = ""
    last_code_cell_id: str | None = None

    # ---- Execution Plan tracking ----
    _current_goal_name: str | None = None

    def _build_plan_phases(current_goal_name: str | None = None) -> list[dict]:
        """Collapse goals into unique phases with status and sub-item details.

        Two modes:
        - current_goal_name provided: everything before it = complete,
          it = current, everything after = upcoming.
        - current_goal_name is None: rely on state.phases_completed only.
          Completed phases = complete, others = upcoming.
        """
        from collections import OrderedDict
        phases: OrderedDict[str, dict] = OrderedDict()  # phase -> {status, details}
        found_current = False
        for g in goals:
            skipped = g.should_skip(state)
            if skipped and g.phase not in phases:
                continue

            if current_goal_name is not None:
                # Explicit current goal — position-based status
                if g.name == current_goal_name:
                    goal_status = "current"
                    found_current = True
                elif found_current:
                    goal_status = "upcoming"
                else:
                    goal_status = "complete"
            else:
                # No current goal — use phases_completed as source of truth
                if g.phase in state.phases_completed:
                    goal_status = "complete"
                else:
                    goal_status = "upcoming"

            detail = {"label": g.description, "status": goal_status}
            if g.phase not in phases:
                phases[g.phase] = {"status": goal_status, "details": [detail]}
            else:
                phases[g.phase]["details"].append(detail)
                if goal_status == "current":
                    phases[g.phase]["status"] = "current"
        return [{"phase": p, "status": d["status"], "details": d["details"]}
                for p, d in phases.items()]

    def _push_plan(current_goal_name: str | None = None, extra_phases: list[dict] | None = None):
        """Emit a plan_update event with collapsed phase-level steps."""
        steps = _build_plan_phases(current_goal_name)
        if extra_phases:
            steps.extend(extra_phases)
        push_event(session_id, {"type": "plan_update", "steps": steps})
        state.plan_steps = steps
        # Persist for REST endpoint on page reload
        try:
            from backend.services.session_manager import get_session_dir
            plan_path = get_session_dir(session_id) / "plan.json"
            plan_path.write_text(json.dumps({"steps": steps}))
        except Exception:
            pass

    def _think(content: str):
        push_event(session_id, {"type": "thinking", "content": content, "notebook_id": "main"})
        time.sleep(0.02)

    def _drain_steering(checkpoint: str, *, acknowledge: bool = True) -> list[dict]:
        """Read queued user steering at safe gaps between agent actions."""
        try:
            from backend.services.steering_service import drain_steering
            items = drain_steering(session_id)
        except Exception as exc:
            _LOG.warning("Steering drain failed at %s: %s", checkpoint, exc)
            return []

        for item in items:
            state.steering_notes.append(item)
            state.consumed_steering_ids.append(item["id"])
            push_event(session_id, {
                "type": "steering_read",
                "message_id": item["id"],
                "content": item["content"],
                "read_at": item.get("read_at"),
                "checkpoint": checkpoint,
            })

        if items and acknowledge:
            joined = "; ".join(item["content"] for item in items)
            _think(f"Steering incorporated: {joined[:180]}")
            _write_and_run(
                f"> Steering applied: {joined[:240]}",
                "markdown",
            )
        return items

    def _write_and_run(
        code: str,
        cell_type: str = "code",
        *,
        cell_id: str | None = None,
        overwrite: bool = False,
    ) -> tuple[str, list[dict], str | None]:
        """Write or overwrite a cell, execute it, return (cell_id, outputs, error)."""
        nonlocal last_code_cell_id
        target_cell_id = cell_id or state.next_cell_id()
        state.cell_phases[target_cell_id] = current_phase
        replacement = overwrite and cell_id is not None
        if replacement:
            push_event(session_id, {"type": "cell_delete", "cell_id": target_cell_id})
            time.sleep(0.05)
        push_event(session_id, {
            "type": "cell_write",
            "cell_id": target_cell_id,
            "cell_type": cell_type,
            "source": code,
            "overwrite": replacement,
            "notebook_id": "main",
        })
        time.sleep(0.02)

        if cell_type == "markdown":
            state.register_cell(target_cell_id, "markdown", code, notebook_id="main")
            _drain_steering(f"after_cell:{target_cell_id}", acknowledge=False)
            return target_cell_id, [], None

        last_code_cell_id = target_cell_id
        push_event(session_id, {"type": "cell_executing", "cell_id": target_cell_id})
        outputs, error = execute_code(session_id, code, cell_id=target_cell_id)

        if error:
            push_event(session_id, {
                "type": "cell_error",
                "cell_id": target_cell_id,
                "error": error,
                "traceback": [o.get("traceback", []) for o in outputs if o.get("output_type") == "error"],
                "notebook_id": "main",
            })
        else:
            push_event(session_id, {
                "type": "cell_output",
                "cell_id": target_cell_id,
                "outputs": outputs,
                "notebook_id": "main",
            })

        state.register_cell(target_cell_id, "code", code, outputs, notebook_id="main")
        _drain_steering(f"after_cell:{target_cell_id}", acknowledge=False)
        time.sleep(0.05)
        return target_cell_id, outputs, error

    def _transition(phase: str, message: str = "", *, render_cell: bool = True):
        nonlocal current_phase
        if phase != current_phase:
            current_phase = phase
            push_event(session_id, {
                "type": "phase_transition",
                "phase": phase,
                "message": message,
            })
            if render_cell:
                # Write phase header as markdown cell with divider
                header = f"---\n\n## {phase}"
                if message:
                    header += f"\n\n{message}"
                _write_and_run(header, "markdown")

    def _backtrack_and_fix(error: str, fix_code: str, reason: str, failed_cell_id: str):
        push_event(session_id, {"type": "backtrack", "reason": reason, "cell_id": failed_cell_id})
        time.sleep(0.02)
        return _write_and_run(fix_code, cell_id=failed_cell_id, overwrite=True)

    def _extract_output_text(outputs: list[dict]) -> str:
        """Extract readable text from cell outputs."""
        parts = []
        for o in outputs:
            text = o.get("text", "")
            if text:
                parts.append(text)
            data = o.get("data", {})
            if data.get("text/plain"):
                parts.append(str(data["text/plain"]))
            if data.get("image/png"):
                parts.append("[matplotlib plot generated]")
        return "\n".join(parts)

    def _extract_images(outputs: list[dict]) -> list[str]:
        """Extract base64 PNG images from cell outputs for vision analysis."""
        images = []
        for o in outputs:
            img = o.get("data", {}).get("image/png")
            if img:
                images.append(img)
        return images

    def _interpret_and_follow_up(
        outputs: list[dict],
        error: str | None,
        goal: EDAGoal,
        state: AgentState,
        goals: list[EDAGoal],
    ):
        """Interpret cell output via LLM and optionally run ONE follow-up."""
        if error:
            # Try to fix the error with LLM help
            _try_fix_error(error, goal, state, goals)
            return

        output_text = _extract_output_text(outputs)
        images = _extract_images(outputs)
        if not output_text.strip() and not images:
            return

        # Interpret what we learned — ONE finding per goal (overwrite, don't pile up)
        # Pass images so vision-capable models can analyze plots
        finding = interpret_output(output_text, goal.phase, images=images)
        if finding:
            # Use goal.name as key to prevent duplicates
            state.findings = [f for f in state.findings if f.get("goal") != goal.name]
            state.findings.append({
                "phase": goal.phase,
                "finding": finding,
                "goal": goal.name,
                "raw_output": output_text[:2000],  # preserve raw numbers for KG/story
            })

        _drain_steering(f"before_followup:{goal.name}")

        # Check for ONE follow-up only (not 3) — must be genuinely surprising
        step = decide_next_step(
            state_summary=state.summarize(),
            last_output=output_text[:1500],
            current_phase=goal.phase,
            goals_remaining=[
                g.name for g in goals
                if g.name not in state.phases_completed and not g.should_skip(state)
            ],
            columns=state.columns,
            run_guidance=state.run_guidance(),
        )

        if step["follow_up"] and step["code"]:
            _think(step["thinking"])
            _, fu_outputs, fu_error = _write_and_run(step["code"], step["cell_type"])
            if fu_error:
                _try_fix_error(fu_error, goal, state, goals)

    def _try_fix_error(error: str, goal: EDAGoal, state: AgentState, goals: list[EDAGoal]):
        """Ask LLM to fix a cell error, using actual column names."""
        step = decide_next_step(
            state_summary=state.summarize(),
            last_output="",
            current_phase=goal.phase,
            goals_remaining=[g.name for g in goals if g.name not in state.phases_completed],
            columns=state.columns,
            error_context=error,
            run_guidance=state.run_guidance(),
        )
        failed_cell_id = last_code_cell_id

        def _skip_cell(reason: str):
            """Remove the failed cell and replace with a markdown skip note."""
            if not failed_cell_id:
                return
            push_event(session_id, {"type": "cell_delete", "cell_id": failed_cell_id})
            time.sleep(0.05)
            _write_and_run(
                f"> **Skipped step** — `{reason[:150]}`",
                "markdown",
                cell_id=failed_cell_id,
                overwrite=True,
            )

        if not step["code"] or not failed_cell_id:
            # LLM couldn't produce a fix — skip the cell
            if failed_cell_id:
                state.add_error(current_phase, error, "No fix generated")
                _skip_cell(error)
            return

        # Attempt 1: first fix
        _, _, fix_error1 = _backtrack_and_fix(
            error,
            step["code"],
            step["thinking"] or f"Retrying with corrected code: {error[:100]}",
            failed_cell_id,
        )
        if not fix_error1:
            return  # Fixed!

        # Attempt 2: second fix with both errors as context
        _think("Second attempt also failed, trying alternative approach")
        second_fix = decide_next_step(
            state_summary=state.summarize(),
            last_output="",
            current_phase=goal.phase,
            goals_remaining=[g.name for g in goals if g.name not in state.phases_completed],
            columns=state.columns,
            error_context=f"Original error: {error}\nFirst fix error: {fix_error1}",
            run_guidance=state.run_guidance(),
        )
        if not second_fix or not second_fix.get("code"):
            state.add_error(current_phase, error, "No alternative fix found")
            _skip_cell(error)
            return

        _, _, fix_error2 = _backtrack_and_fix(
            fix_error1,
            second_fix["code"],
            f"Alternative fix attempt: {fix_error1[:80]}",
            failed_cell_id,
        )
        if not fix_error2:
            return  # Fixed on second attempt!

        # All attempts exhausted — skip the cell
        state.add_error(current_phase, error, "Failed after 3 attempts")
        _skip_cell(error)

    # ---- Title cell ----
    title_cell = (
        f"# Exploratory Data Analysis: `{filename}`\n\n"
        f"Automated analysis powered by **AgenticEDA**."
    )
    if state.research_direction:
        title_cell += f"\n\n**Research direction:** {state.research_direction}"
    _write_and_run(
        title_cell,
        "markdown",
    )

    # ---- Execute goals ----

    # Emit initial plan (all upcoming)
    _push_plan(None, extra_phases=[{"phase": "Deep Investigation", "status": "upcoming"}, {"phase": "Report Generation", "status": "upcoming"}])

    for goal in goals:
        _drain_steering(f"before_goal:{goal.name}")
        if goal.should_skip(state):
            _LOG.info("Skipping goal: %s (condition met)", goal.name)
            continue

        _transition(goal.phase, goal.description)
        _push_plan(goal.name, extra_phases=[{"phase": "Deep Investigation", "status": "upcoming"}, {"phase": "Report Generation", "status": "upcoming"}])

        try:
            if goal.name == "load_dataset":
                _think(f"Loading {filename} ({file_format} format)...")
                failed_cell_id, outputs, error = _write_and_run(ct.load_dataset_code(filename, file_format))
                if error:
                    _think("Load failed. Trying with different encoding...")
                    _, outputs, error = _backtrack_and_fix(
                        error,
                        f'df = pd.read_csv("{filename}", encoding="latin-1", on_bad_lines="skip")\nprint(f"Loaded {{len(df)}} rows x {{len(df.columns)}} columns")\ndf.head()',
                        "CSV parse error — retrying with latin-1 encoding and skipping bad lines",
                        failed_cell_id,
                    )
                if not error:
                    state.dataset_loaded = True
                    # Parse row/col count from output
                    for o in outputs:
                        text = o.get("text", "")
                        if "rows" in text and "columns" in text:
                            try:
                                parts = text.split()
                                for i, p in enumerate(parts):
                                    if p == "rows":
                                        state.row_count = int(parts[i - 1])
                                    if p == "columns":
                                        state.col_count = int(parts[i - 1])
                            except (ValueError, IndexError):
                                pass
                    # Get column names into state
                    _, col_outputs, _ = _write_and_run("print(list(df.columns))")
                    col_text = _extract_output_text(col_outputs)
                    if col_text.strip().startswith("["):
                        try:
                            import ast
                            state.columns = ast.literal_eval(col_text.strip())
                        except Exception as exc:
                            _LOG.warning("Column name parsing failed: %s", exc)
                    _interpret_and_follow_up(outputs, error, goal, state, goals)

            elif goal.name == "inspect_dtypes":
                _think("Checking column types to identify datetime, numeric, and categorical columns...")
                _, outputs, error = _write_and_run(ct.inspect_dtypes_code())
                if not error:
                    # Parse dtypes from the machine-readable JSON emitted by inspect_dtypes_code().
                    # Format: AGENTICEDA_DTYPES:{...json dict of col -> dtype string...}
                    datetime_cols: list[str] = []
                    dtypes_raw: dict[str, str] = {}
                    for o in outputs:
                        text = o.get("text", "")
                        for line in text.split("\n"):
                            if line.startswith("AGENTICEDA_DTYPES:"):
                                try:
                                    dtypes_raw = json.loads(line[len("AGENTICEDA_DTYPES:"):])
                                except json.JSONDecodeError:
                                    pass
                                break
                        if dtypes_raw:
                            break

                    for col_name, dtype in dtypes_raw.items():
                        if not col_name or not dtype:
                            continue
                        dtype_l = dtype.lower()
                        state.dtypes[col_name] = dtype
                        if dtype_l.startswith("datetime"):
                            datetime_cols.append(col_name)
                        elif dtype_l.startswith("float") or dtype_l.startswith("int"):
                            if col_name not in state.numeric_cols:
                                state.numeric_cols.append(col_name)
                        elif dtype_l in ("object", "str", "string", "bool", "boolean", "category"):
                            if col_name not in state.categorical_cols:
                                state.categorical_cols.append(col_name)

                    if state.dtypes:
                        state.columns = list(state.dtypes.keys())

                    # If a datetime column already exists, prefer it as the time axis.
                    if datetime_cols and state.time_col is None:
                        def _time_name_score(name: str) -> int:
                            n = name.lower()
                            score = 0
                            if "timestamp" in n or "datetime" in n:
                                score += 6
                            if n == "time":
                                score += 5
                            if "time" in n:
                                score += 4
                            if n == "date":
                                score += 4
                            if "date" in n:
                                score += 3
                            return score

                        state.time_col = sorted(datetime_cols, key=_time_name_score, reverse=True)[0]
                    _interpret_and_follow_up(outputs, error, goal, state, goals)

            elif goal.name == "inspect_describe":
                _think("Computing statistical summary for all columns...")
                _, outputs, error = _write_and_run(ct.inspect_describe_code())
                _interpret_and_follow_up(outputs, error, goal, state, goals)

            elif goal.name == "parse_datetime":
                # Parse into a canonical time axis (__agenticeda_time) with quality gating.
                def _score(name: str) -> int:
                    n = name.lower()
                    score = 0
                    if "timestamp" in n or "datetime" in n:
                        score += 6
                    if n == "time":
                        score += 5
                    if "time" in n:
                        score += 4
                    if n == "date":
                        score += 4
                    if "date" in n:
                        score += 3
                    if n == "dt":
                        score += 2
                    if "year" in n:
                        score += 1
                    dtype_l = str(state.dtypes.get(name, "")).lower()
                    if dtype_l.startswith("datetime"):
                        score += 8
                    elif score > 0 and (dtype_l.startswith("float") or dtype_l.startswith("int")):
                        # Numeric + time-like name may indicate epoch seconds/ms.
                        score += 2
                    return score

                # Candidate split date+time columns (e.g., Date + Time).
                date_candidates = [c for c in state.columns if "date" in c.lower()]
                time_candidates = [
                    c for c in state.columns
                    if "time" in c.lower()
                    and "datetime" not in c.lower()
                    and "timestamp" not in c.lower()
                ]

                date_col = sorted(date_candidates, key=_score, reverse=True)[0] if date_candidates else None
                time_col = sorted(time_candidates, key=_score, reverse=True)[0] if time_candidates else None

                # Otherwise choose a single best candidate by name.
                ranked = sorted(state.columns, key=_score, reverse=True)
                single_col = ranked[0] if ranked and _score(ranked[0]) > 0 else None

                chosen_time_col = None
                chosen_date_col = None
                if date_col and time_col and date_col != time_col:
                    chosen_date_col = date_col
                    chosen_time_col = time_col
                elif single_col:
                    chosen_time_col = single_col

                if chosen_time_col:
                    if chosen_date_col:
                        _think(f"Combining '{chosen_date_col}' + '{chosen_time_col}' into a canonical datetime column...")
                    else:
                        _think(f"'{chosen_time_col}' looks like a datetime column. Parsing it into a canonical time axis...")

                    _, outputs, error = _write_and_run(
                        ct.parse_datetime_code(chosen_time_col, chosen_date_col)
                    )

                    if not error:
                        out_text = _extract_output_text(outputs)
                        parse_line = next(
                            (ln for ln in out_text.splitlines() if ln.startswith("AGENTICEDA_TIME_PARSE ")),
                            "",
                        )
                        ok = "ok=true" in parse_line
                        if ok:
                            state.time_col = "__agenticeda_time"
                            state.add_finding("Data Cleaning", "Parsed canonical time axis '__agenticeda_time'")
                        else:
                            # Leave time_col unset so time-series steps are skipped (prevents blank charts).
                            state.time_col = None
                            state.add_finding(
                                "Data Cleaning",
                                "Time parsing attempted but quality was low; skipping time-series steps",
                            )

                    _interpret_and_follow_up(outputs, error, goal, state, goals)
                else:
                    _think("No obvious datetime column found. Proceeding without time index.")

            elif goal.name == "check_missing":
                _think("Auditing missing values across all columns...")
                _, outputs, error = _write_and_run(ct.inspect_missing_code())
                _interpret_and_follow_up(outputs, error, goal, state, goals)

            elif goal.name == "handle_missing":
                # Check if there are missing values by running a quick check
                _, outputs, _ = _write_and_run("missing_count = df.isnull().sum().sum()\nprint(f'Total missing: {missing_count}')")
                has_missing = False
                for o in outputs:
                    if "Total missing:" in o.get("text", "") and "Total missing: 0" not in o.get("text", ""):
                        has_missing = True

                if has_missing:
                    _think("There are missing values. I'll handle each column appropriately...")
                    for col in state.numeric_cols:
                        if state.time_col:
                            _write_and_run(ct.handle_missing_interpolate_code(col))
                        else:
                            _write_and_run(ct.handle_missing_ffill_code(col))
                    state.add_finding("Data Cleaning", "Handled missing values via interpolation/forward-fill")
                else:
                    _think("No missing values — data is clean!")

            elif goal.name == "distributions":
                _think(f"Plotting distributions for {len(state.numeric_cols)} numeric columns...")
                _, outputs, error = _write_and_run(ct.plot_distributions_code(state.numeric_cols))
                state.add_finding("Univariate Analysis", f"Plotted distributions for {len(state.numeric_cols)} columns")
                _interpret_and_follow_up(outputs, error, goal, state, goals)

            elif goal.name == "time_series_plot":
                plot_cols = state.numeric_cols[:4] if state.numeric_cols else state.target_cols
                if plot_cols and state.time_col:
                    _think(f"Plotting time series for: {', '.join(plot_cols[:4])}...")
                    _, outputs, error = _write_and_run(ct.plot_time_series_code(state.time_col, plot_cols[:4]))
                    state.add_finding("Time Series", f"Plotted time series for {len(plot_cols[:4])} variables")
                    _interpret_and_follow_up(outputs, error, goal, state, goals)

            elif goal.name == "seasonality":
                if state.numeric_cols and state.time_col:
                    col = state.numeric_cols[0]
                    _think(f"Checking for seasonal patterns in '{col}'...")
                    _, outputs, error = _write_and_run(ct.seasonality_code(state.time_col, col))
                    state.add_finding("Time Series", f"Checked seasonality for '{col}'")
                    _interpret_and_follow_up(outputs, error, goal, state, goals)

            elif goal.name == "rolling_stats":
                if state.numeric_cols and state.time_col:
                    col = state.numeric_cols[0]
                    _think(f"Computing rolling statistics for '{col}' to detect trends and volatility...")
                    _, outputs, error = _write_and_run(ct.rolling_stats_code(state.time_col, col))
                    state.add_finding("Dynamics", f"Computed rolling stats for '{col}'")
                    _interpret_and_follow_up(outputs, error, goal, state, goals)

            elif goal.name == "outlier_detection":
                if state.numeric_cols:
                    col = state.numeric_cols[0]
                    _think(f"Detecting outliers in '{col}' using IQR method...")
                    _, outputs, error = _write_and_run(ct.outlier_detection_code(col))
                    state.add_finding("Dynamics", f"Outlier detection on '{col}'")
                    _interpret_and_follow_up(outputs, error, goal, state, goals)

            elif goal.name == "correlations":
                if len(state.numeric_cols) >= 2:
                    _think(f"Computing correlation matrix for {len(state.numeric_cols)} numeric columns...")
                    _, outputs, error = _write_and_run(ct.correlation_code(state.numeric_cols))
                    state.add_finding("Correlations", "Computed pairwise correlations")
                    _interpret_and_follow_up(outputs, error, goal, state, goals)

            elif goal.name == "train_test_split":
                if state.time_col:
                    _think("Splitting data chronologically (80/20) for model readiness...")
                    _, outputs, error = _write_and_run(ct.train_test_split_code(state.time_col))
                    state.add_finding("Train/Test Split", "Chronological 80/20 split")
                    _interpret_and_follow_up(outputs, error, goal, state, goals)

            elif goal.name == "summary":
                _think("Summarizing all findings...")
                _write_and_run(ct.summary_markdown(state.findings), "markdown")

            state.mark_phase_done(goal.phase)

        except Exception as exc:
            _LOG.exception("Goal '%s' failed: %s", goal.name, exc)
            _think(f"Error in {goal.name}: {exc}. Moving to next step...")
            state.add_error(goal.phase, str(exc), "skipped")

    # ---- Investigation Phase: Multi-Loop Hypothesis-Driven Deep Dives ----
    from src.agent.hypothesis import generate_hypotheses
    from src.agent.subagent import run_subagent, InvestigationResult
    from src.agent.knowledge_graph import KnowledgeGraph, KGEdge
    from backend.services.kernel_pool import KernelPoolManager

    kg = KnowledgeGraph()
    pool = KernelPoolManager()

    # Populate knowledge graph with pass 1 findings
    for f in state.findings:
        kg.add_fact(f.get("finding", ""), f.get("phase", ""), f.get("cell_id"))

    # Save dataset checkpoint for subagent kernels
    try:
        _write_and_run(
            "import os; os.makedirs('.cache', exist_ok=True)\n"
            "df.to_parquet('.cache/df_clean.parquet', index=True)\n"
            "print('Dataset checkpoint saved')"
        )
    except Exception as exc:
        _LOG.warning("Dataset checkpoint failed: %s", exc)

    # Get session dir for subagent preamble
    from backend.services.session_manager import get_session_dir
    session_dir = str(get_session_dir(session_id) / "uploads")

    # Verify checkpoint exists
    import pathlib
    cache_path = pathlib.Path(session_dir) / ".cache" / "df_clean.parquet"
    parquet_available = cache_path.exists()
    if not parquet_available:
        _LOG.warning("Dataset checkpoint not found at %s — subagents will use main kernel", cache_path)

    _transition("Investigation Phase", "Generating hypotheses from initial findings...", render_cell=False)
    _drain_steering("before_hypothesis_generation")
    # Track investigation sub-items (hypothesis titles + statuses)
    _investigation_details: list[dict] = [{"label": "Generating hypotheses...", "status": "current"}]
    _push_plan(None, extra_phases=[{"phase": "Deep Investigation", "status": "current", "details": _investigation_details}, {"phase": "Report Generation", "status": "upcoming"}])
    findings_summary = "\n".join(f"- {f.get('finding', '')}" for f in state.findings if f.get('finding'))
    _write_and_run(
        "---\n\n"
        "# Deep-Dive Investigations\n\n"
        "The initial EDA is complete. The agent is now formulating and testing hypotheses "
        "based on what it discovered above.\n\n"
        "**Key findings so far:**\n\n"
        f"{findings_summary}",
        "markdown",
    )

    cell_counters = []  # Track for final cell_count update

    _LOG.info("Starting investigation loops: max_loops=%d, max_subagents=%d, columns=%s",
              max_loops, max_subagents, state.columns[:5])

    for loop_num in range(1, max_loops + 1):
        state.loop_count = loop_num
        _LOG.info("=== LOOP %d/%d START === columns=%d, findings=%d",
                  loop_num, max_loops, len(state.columns), len(state.findings))

        push_event(session_id, {
            "type": "loop_start",
            "loop_number": loop_num,
            "total_loops": max_loops,
        })

        _transition(
            f"Investigation Loop {loop_num}/{max_loops}",
            f"Generating hypotheses for loop {loop_num}...",
            render_cell=False,
        )

        # Generate hypotheses using KG context
        try:
            kg_context = kg.get_context_for_hypothesis_generation()
            hypotheses = generate_hypotheses(
                columns=state.columns,
                numeric_cols=state.numeric_cols,
                time_col=state.time_col,
                findings=state.findings,
                row_count=state.row_count,
                col_count=state.col_count,
                kg_context=kg_context,
                run_guidance=state.run_guidance(),
            )

            # Deduplicate against KG
            novel = []
            for hyp in hypotheses:
                existing = kg.find_similar_hypothesis(hyp, threshold=0.75)
                if existing and existing.confidence > 0.8:
                    _think(f"Skipping '{hyp.title}' - already investigated (confidence: {existing.confidence:.0%})")
                    continue
                novel.append(hyp)
                if len(novel) >= max_subagents:
                    break

            hypotheses = novel
        except Exception as exc:
            _LOG.exception("Hypothesis generation failed in loop %d: %s", loop_num, exc)
            _think(f"Hypothesis generation failed: {exc}")
            hypotheses = []

        if not hypotheses:
            _think("No novel hypotheses remain. Moving to report generation.")
            push_event(session_id, {"type": "loop_complete", "loop_number": loop_num})
            break

        _think(f"Loop {loop_num}: investigating {len(hypotheses)} hypotheses in parallel.")

        _write_and_run(
            f"---\n\n## Investigation Loop {loop_num}\n\n"
            f"Testing {len(hypotheses)} hypothesis(es).",
            "markdown",
        )

        # Write waiting cell in main notebook
        waiting_cell_id = state.next_cell_id()
        hyp_list = "\n".join(f"- **{h.title}**" for h in hypotheses)
        _write_and_run(
            f"---\n\n### Dispatched {len(hypotheses)} Subagents — Loop {loop_num}\n\n"
            f"Investigating in parallel:\n{hyp_list}\n\n"
            f"*Waiting for results...*",
            "markdown",
        )

        push_event(session_id, {
            "type": "subagents_dispatched",
            "notebook_id": "main",
            "loop_number": loop_num,
            "count": len(hypotheses),
            "hypothesis_ids": [h.id for h in hypotheses],
            "titles": [h.title for h in hypotheses],
        })

        # Update investigation plan — accumulate hypotheses across loops
        # Remove the placeholder "Generating hypotheses..." if still present
        _investigation_details[:] = [d for d in _investigation_details if d.get("label") != "Generating hypotheses..."]
        for h in hypotheses:
            _investigation_details.append({"label": h.title, "status": "current"})
        _push_plan(None, extra_phases=[{"phase": "Deep Investigation", "status": "current", "details": _investigation_details}, {"phase": "Report Generation", "status": "upcoming"}])

        # Allocate subagent kernels for parallel execution
        sub_kernel_ids = [None] * len(hypotheses)
        if parquet_available:
            try:
                sub_kernel_ids = pool.allocate_subagent_kernels(session_id, len(hypotheses))
                for kid in sub_kernel_ids:
                    pool.inject_dataset_preamble(kid, session_dir)
            except Exception as exc:
                _LOG.warning("Kernel allocation/preamble failed: %s — falling back to main kernel", exc)
                try:
                    pool.shutdown_subagent_kernels(session_id)
                except Exception as exc2:
                    _LOG.warning("Kernel pool cleanup failed: %s", exc2)
                sub_kernel_ids = [None] * len(hypotheses)
        else:
            _think("Using main kernel for investigations (no parquet checkpoint).")

        # Run subagents via multiprocessing — true parallelism, no GIL
        import multiprocessing as mp
        import threading
        from backend.services.kernel_manager import get_kernel_connection_file
        from src.agent.subagent_worker import subagent_process_worker

        loop_cell_counters = [[1000 + loop_num * 1000 + i * 100] for i in range(len(hypotheses))]
        cell_counters.extend(loop_cell_counters)
        results: list[InvestigationResult] = []

        hypothesis_order = []
        processes: list[tuple] = []  # (process, hyp, notebook_id, event_queue, result_queue)

        # Shared event queue — one drainer thread forwards events to push_event
        event_queue: mp.Queue = mp.Queue()
        active_workers = 0

        for i, (hyp, kid) in enumerate(zip(hypotheses, sub_kernel_ids)):
            notebook_id = f"agent_{i + 1}"
            hypothesis_order.append((hyp, notebook_id))

            push_event(session_id, {
                "type": "subagent_start",
                "hypothesis_id": hyp.id,
                "notebook_id": notebook_id,
                "title": hyp.title,
            })
            push_event(session_id, {
                "type": "cell_write",
                "cell_id": f"{notebook_id}_loop{loop_num}_header",
                "cell_type": "markdown",
                "source": f"## Loop {loop_num}: {hyp.title}\n\n> {hyp.description}",
                "notebook_id": notebook_id,
            })

            # Get connection file for the sub-kernel (or main kernel)
            conn_file = get_kernel_connection_file(kid) if kid else get_kernel_connection_file(session_id)
            if conn_file is None:
                _LOG.warning("No connection file for kernel %s, skipping", kid or session_id)
                continue

            result_queue: mp.Queue = mp.Queue()
            subagent_deadline = time.time() + max(loop_timeout - 30, 60)  # leave 30s margin
            p = mp.Process(
                target=subagent_process_worker,
                kwargs=dict(
                    connection_file=conn_file,
                    hypothesis_id=hyp.id,
                    hypothesis_title=hyp.title,
                    hypothesis_description=hyp.description,
                    relevant_cols=hyp.relevant_cols,
                    all_columns=state.columns,
                    time_col=state.time_col,
                    session_id=session_id,
                    event_queue=event_queue,
                    cell_counter_start=loop_cell_counters[i][0],
                    max_cells=5,
                    notebook_id=notebook_id,
                    kg_context=kg.get_context_for_hypothesis_generation() if kg else "",
                    run_guidance=state.run_guidance(),
                    result_queue=result_queue,
                    deadline=subagent_deadline,
                ),
                daemon=True,
            )
            processes.append((p, hyp, notebook_id, result_queue))
            active_workers += 1

        # Drainer thread: forward events from child processes to push_event
        drain_done = threading.Event()
        def _drain_events():
            finished = 0
            drain_deadline = time.time() + loop_timeout + 30  # generous deadline
            while finished < active_workers and time.time() < drain_deadline:
                try:
                    evt = event_queue.get(timeout=2)
                    if evt is None:
                        finished += 1
                        continue
                    push_event(session_id, evt)
                except Exception:
                    # Check if all processes are dead (sentinel lost due to terminate())
                    all_dead = all(not p.is_alive() for p, _, _, _ in processes)
                    if all_dead:
                        break
            drain_done.set()

        drainer = threading.Thread(target=_drain_events, daemon=True)
        drainer.start()

        # Start all processes
        _think(f"Starting {len(processes)} investigation processes (timeout: {loop_timeout}s)...")
        for p, _, _, _ in processes:
            p.start()

        # Collect results.
        # IMPORTANT: Read from result_queue BEFORE p.join() to prevent deadlock.
        # On macOS, multiprocessing.Queue uses a pipe with limited buffer. If the
        # child puts data on the queue and the pipe fills up, put() blocks and the
        # child can't exit, so p.join() hangs forever. Reading first drains the pipe.
        results_by_hyp: dict[str, InvestigationResult] = {}
        join_deadline = time.time() + loop_timeout

        for idx, (p, hyp, notebook_id, rq) in enumerate(processes):
            remaining = max(join_deadline - time.time(), 1)
            _LOG.info("Waiting for subagent %s (%s) — %.0fs remaining", hyp.id, hyp.title[:40], remaining)

            # First: try to read the result (with timeout). This unblocks the pipe.
            try:
                status, data = rq.get(timeout=remaining)
            except Exception:
                status, data = None, None

            # Now join — process should exit quickly since queue is drained
            p.join(timeout=10)
            if p.is_alive():
                _LOG.warning("Subagent process for %s still alive after result read, terminating", hyp.id)
                p.terminate()
                p.join(timeout=5)

            def _mark_hyp_done(title: str, detail_status: str = "complete"):
                for d in _investigation_details:
                    if d["label"] == title:
                        d["status"] = detail_status
                        break
                _push_plan(None, extra_phases=[{"phase": "Deep Investigation", "status": "current", "details": _investigation_details}, {"phase": "Report Generation", "status": "upcoming"}])

            if status == "ok":
                # Load images from temp file (avoids pipe deadlock)
                images = {}
                images_file = data.get("_images_file")
                if images_file:
                    try:
                        with open(images_file) as f:
                            images = json.loads(f.read())
                        os.remove(images_file)
                    except Exception as img_exc:
                        _LOG.warning("Failed to load images from %s: %s", images_file, img_exc)

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
                    cell_sources=data.get("cell_sources", {}),
                    cell_outputs=data.get("cell_outputs", {}),
                )

                if getattr(result, "status", "complete") == "complete":
                    results.append(result)
                    results_by_hyp[hyp.id] = result
                    state.subagent_run_count += 1
                    push_event(session_id, {
                        "type": "subagent_complete",
                        "hypothesis_id": hyp.id,
                        "notebook_id": notebook_id,
                        "finding": result.finding,
                        "confidence": result.confidence,
                        "status": "complete",
                    })
                    _mark_hyp_done(hyp.title, "complete")
                elif getattr(result, "status", "complete") == "timeout":
                    _LOG.warning("Subagent for %s hit internal deadline (timeout)", hyp.id)
                    push_event(session_id, {"type": "subagent_timeout", "hypothesis_id": hyp.id, "notebook_id": notebook_id})
                    _think(f"Investigation of '{hyp.title}' timed out. Moving on.")
                    _mark_hyp_done(hyp.title, "complete")
                else:
                    _LOG.warning("Subagent for %s returned failed result", hyp.id)
                    push_event(session_id, {
                        "type": "subagent_complete",
                        "hypothesis_id": hyp.id,
                        "notebook_id": notebook_id,
                        "finding": result.finding,
                        "confidence": result.confidence,
                        "status": "failed",
                    })
                    _think(f"Investigation of '{hyp.title}' failed. Moving on.")
                    _mark_hyp_done(hyp.title, "complete")
            elif status == "error":
                _LOG.warning("Subagent for %s returned error: %s", hyp.id, data)
                push_event(session_id, {"type": "subagent_timeout", "hypothesis_id": hyp.id, "notebook_id": notebook_id})
                _think(f"Investigation of '{hyp.title}' failed. Moving on.")
                _mark_hyp_done(hyp.title, "complete")
            else:
                _LOG.warning("Subagent for %s: no result received (timeout)", hyp.id)
                push_event(session_id, {"type": "subagent_timeout", "hypothesis_id": hyp.id, "notebook_id": notebook_id})
                _think(f"Investigation of '{hyp.title}' timed out. Moving on.")
                _mark_hyp_done(hyp.title, "complete")

        # Wait for event drainer to finish
        drain_done.wait(timeout=10)
        drainer.join(timeout=5)

        # Shutdown subagent kernels after this loop
        try:
            pool.shutdown_subagent_kernels(session_id)
        except Exception as exc:
            _LOG.warning("Subagent kernel shutdown failed: %s", exc)

        if not results:
            _think(f"Loop {loop_num}: all investigations failed. Stopping.")
            push_event(session_id, {"type": "loop_complete", "loop_number": loop_num})
            break

        # --- Accumulation phase: write results + ingest into KG ---
        total_steps = len(hypothesis_order) + 2  # hypotheses + compilation + KG reinforcement
        step_num = 0

        def _progress(label: str):
            nonlocal step_num
            step_num += 1
            push_event(session_id, {
                "type": "accumulation_progress",
                "step": step_num,
                "total": total_steps,
                "label": label,
                "notebook_id": "main",
            })
            _think(f"[{step_num}/{total_steps}] {label}")

        _progress(f"Compiling {len(results)} investigation results...")

        compilation = f"---\n\n### Loop {loop_num} Results — {len(results)} Investigation(s)\n\n"
        for result in results:
            conf_pct = int(result.confidence * 100)
            conf_label = "High" if conf_pct >= 70 else "Medium" if conf_pct >= 40 else "Low"
            compilation += f"**{result.hypothesis_title}** ({conf_label} {conf_pct}%)\n"
            compilation += f"> {result.finding}\n\n"
        _write_and_run(compilation, "markdown")

        push_event(session_id, {
            "type": "subagents_returned",
            "notebook_id": "main",
            "loop_number": loop_num,
            "results_count": len(results),
        })

        # Write hypothesis→result pairs in original order, ingest into KG
        for hyp, notebook_id in hypothesis_order:
            result = results_by_hyp.get(hyp.id)
            cols_str = ", ".join(f"`{c}`" for c in hyp.relevant_cols) if hyp.relevant_cols else "all columns"

            _progress(f"Recording: {hyp.title[:50]}...")

            _write_and_run(
                f"---\n\n### Hypothesis: {hyp.title}\n\n"
                f"> {hyp.description}\n\n"
                f"**Relevant columns:** {cols_str}",
                "markdown",
            )

            if not result:
                _write_and_run(
                    f"**Result:** Investigation failed or timed out.",
                    "markdown",
                )
                continue

            conf_pct = int(result.confidence * 100)
            conf_label = "High" if conf_pct >= 70 else "Medium" if conf_pct >= 40 else "Low"
            _write_and_run(
                f"**Finding:** {result.finding}\n\n"
                f"**Confidence:** {conf_label} ({conf_pct}%)",
                "markdown",
            )

            # Ingest into KG (fast — no LLM calls)
            nid = kg.add_investigation(
                hypothesis_id=result.hypothesis_id,
                hypothesis_title=result.hypothesis_title,
                finding=result.finding,
                evidence_cells=result.cell_ids,
                plot_cells=result.plot_cell_ids,
                confidence=result.confidence,
                sub_findings=result.sub_findings,
                columns=getattr(result, 'relevant_cols', []) or [],
                analysis_type="hypothesis_investigation",
                loop_number=loop_num,
            )
            state.add_finding(f"Investigation: {result.hypothesis_title}", result.finding)

            # Store plot images in KG metadata for story generation (no vision LLM call)
            all_images = []
            for cell_id, imgs in getattr(result, 'images', {}).items():
                for img in imgs:
                    all_images.append({"cell_id": cell_id, "image_png": img})
            if all_images:
                node = kg.nodes.get(nid)
                if node:
                    node.metadata["plot_images"] = all_images

            node = kg.nodes.get(nid)
            if node:
                node.metadata["cell_sources"] = getattr(result, 'cell_sources', {})
                node.metadata["cell_outputs"] = getattr(result, 'cell_outputs', {})

        _progress("Cross-referencing findings...")

        push_event(session_id, {"type": "loop_complete", "loop_number": loop_num})

        # Convergence check: stop if this loop produced no new high-confidence findings
        new_high_confidence = sum(
            1 for r in results
            if r.confidence > 0.5
        )
        if loop_num > 1 and new_high_confidence == 0:
            _think("No new significant findings this loop. Analysis converged.")
            break

        # Cross-investigation reinforcement: if findings overlap on columns, reinforce both
        conclusion_nodes = kg.query_by_type("conclusion")
        for i, n1 in enumerate(conclusion_nodes):
            n1_cols = set(c.lower() for c in n1.metadata.get("columns", []))
            if not n1_cols:
                continue
            for n2 in conclusion_nodes[i+1:]:
                n2_cols = set(c.lower() for c in n2.metadata.get("columns", []))
                if n1_cols & n2_cols:  # Shared columns
                    kg.reinforce(n1.id)
                    kg.reinforce(n2.id)

    # Update cell count from all subagent counters
    if cell_counters:
        state.cell_count = max(state.cell_count, *(cc[0] for cc in cell_counters))

    # Synthesize conclusions via LLM — connect threads, identify contradictions, rank
    _transition("Conclusions", "Synthesizing all findings...", render_cell=False)
    _push_plan(None, extra_phases=[{"phase": "Deep Investigation", "status": "complete"}, {"phase": "Report Generation", "status": "current", "details": [{"label": "Synthesizing conclusions", "status": "current"}]}])
    raw_conclusions = kg.get_top_conclusions(8)
    all_findings_text = "\n".join(
        f"- [{f.get('phase', '')}] {f.get('finding', '')} (raw: {f.get('raw_output', '')[:200]})"
        for f in state.findings
    )

    if raw_conclusions or all_findings_text.strip():
        try:
            from src.config.config import get_chat_model
            from langchain_core.messages import SystemMessage, HumanMessage

            llm = get_chat_model()
            kg_context = kg.get_context_for_hypothesis_generation() if kg else ""

            synth_response = llm.invoke([
                SystemMessage(content=(
                    "You are writing the final conclusions for an EDA report. Given all findings "
                    "from multiple investigation loops, write 3-5 numbered conclusions. For each:\n"
                    "- State the finding with specific numbers\n"
                    "- Note if multiple investigations confirmed it (cross-reference)\n"
                    "- Flag any contradictions between investigations\n"
                    "- Suggest what to investigate next\n"
                    "Be specific and quantitative. Do NOT use bullet points inside numbered items. "
                    "Format with markdown. For math notation, use proper LaTeX delimiters: $x$ for inline "
                    "(e.g., $r = 0.95$, $p < 0.05$, $\\chi^2$). Never write raw LaTeX without $ delimiters. "
                    "Never use emojis or special unicode symbols."
                )),
                HumanMessage(content=(
                    f"Knowledge graph summary:\n{kg_context[:2000]}\n\n"
                    f"Top findings by confidence:\n" + "\n".join(f"- {c}" for c in raw_conclusions) + "\n\n"
                    f"All investigation findings:\n{all_findings_text[:3000]}"
                )),
            ])
            conclusion_text = synth_response.content.strip()
        except Exception as exc:
            _LOG.warning("Conclusion synthesis LLM failed: %s", exc)
            conclusion_text = "\n".join(f"{i+1}. {c}" for i, c in enumerate(raw_conclusions)) if raw_conclusions else "No conclusions drawn."
    else:
        conclusion_text = "No conclusions drawn. This analysis did not produce significant findings."

    _write_and_run(
        "---\n\n"
        "# Conclusions\n\n"
        f"{conclusion_text}\n\n"
        "---\n\n"
        "*This analysis was generated automatically by the AgenticEDA agent.*",
        "markdown",
    )

    # Store knowledge graph in state for story generation
    state.knowledge_graph = kg

    # ---- Done ----
    # Mark all plan steps complete
    _push_plan(None, extra_phases=[{"phase": "Deep Investigation", "status": "complete"}, {"phase": "Report Generation", "status": "complete"}])
    push_event(session_id, {
        "type": "complete",
        "summary": state.summarize(),
    })

    _LOG.info("Agent complete for session %s: %d phases, %d findings, %d hypotheses investigated",
              session_id, len(state.phases_completed), len(state.findings), len(hypotheses))

    return state
