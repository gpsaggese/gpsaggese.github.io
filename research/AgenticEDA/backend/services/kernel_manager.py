"""Real IPython kernel manager for executing code cells."""
from __future__ import annotations

import json
import logging
import os
import pathlib
import queue
import subprocess
import sys
from typing import Any

import jupyter_client

_LOG = logging.getLogger(__name__)

# Active kernels per session
_kernels: dict[str, jupyter_client.KernelManager] = {}
_clients: dict[str, jupyter_client.KernelClient] = {}

SESSIONS_DIR = pathlib.Path(__file__).resolve().parents[2] / "sessions"
KERNEL_NAME = "agenticeda"


def _ensure_kernel_spec() -> str:
    """Ensure the active Python environment has a usable kernelspec."""
    ksm = jupyter_client.kernelspec.KernelSpecManager()
    available = ksm.find_kernel_specs()
    if KERNEL_NAME in available:
        return KERNEL_NAME

    try:
        import ipykernel  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "ipykernel is not installed in the active Python environment. "
            "Install requirements.txt in the same environment used to run the backend."
        ) from exc

    cmd = [
        sys.executable,
        "-m",
        "ipykernel",
        "install",
        "--prefix",
        sys.prefix,
        "--name",
        KERNEL_NAME,
        "--display-name",
        "AgenticEDA",
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Failed to install the Jupyter kernelspec for the active Python environment. "
            f"stderr: {exc.stderr.strip()}"
        ) from exc

    return KERNEL_NAME


def get_or_create_kernel(session_id: str) -> jupyter_client.KernelClient:
    """Get or create a kernel for a session."""
    if session_id in _clients:
        client = _clients[session_id]
        if client.is_alive():
            return client
        # Dead client, clean up
        shutdown_kernel(session_id)

    km = jupyter_client.KernelManager(kernel_name=_ensure_kernel_spec())

    # Set working directory to session uploads
    session_dir = SESSIONS_DIR / session_id
    uploads_dir = session_dir / "uploads"
    cwd = str(uploads_dir if uploads_dir.exists() else session_dir)
    env = dict(os.environ)
    env["AGENTICEDA_SESSION_ID"] = session_id
    env["AGENTICEDA_SESSION_DIR"] = str(session_dir)
    km.start_kernel(cwd=cwd, env=env)

    client = km.client()
    client.start_channels()
    client.wait_for_ready(timeout=30)

    _kernels[session_id] = km
    _clients[session_id] = client

    # Pre-inject common imports + inline plotting
    _execute_silent(client, """%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import json
import os
import datetime as _agenticeda_datetime
from pathlib import Path
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 20)
pd.set_option('display.width', 120)
__agenticeda_plot_spec_emitted = 0
__agenticeda_plot_spec_has_source = False

def _agenticeda_json_default(value):
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    if isinstance(value, (set, tuple)):
        return list(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)

def emit_plot_spec(plot_spec=None, **kwargs):
    global __agenticeda_plot_spec_emitted, __agenticeda_plot_spec_has_source
    payload = {}
    if isinstance(plot_spec, dict):
        payload.update(plot_spec)
    elif plot_spec is not None:
        payload["source"] = plot_spec
    if kwargs:
        payload.update(kwargs)

    session_dir = os.environ.get("AGENTICEDA_SESSION_DIR")
    if not session_dir:
        return None

    payload.setdefault("session_id", os.environ.get("AGENTICEDA_SESSION_ID", ""))
    payload.setdefault("cell_id", globals().get("__agenticeda_cell_id", ""))
    payload.setdefault("plot_spec_version", 1)
    payload.setdefault("emitted_at", _agenticeda_datetime.datetime.now(_agenticeda_datetime.timezone.utc).isoformat())

    path = Path(session_dir) / "plot_specs.jsonl"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, default=_agenticeda_json_default))
            fh.write("\\n")
        __agenticeda_plot_spec_emitted += 1
        src = payload.get("source")
        if isinstance(src, dict) and "data" in src:
            __agenticeda_plot_spec_has_source = True
    except Exception:
        return None

    return None

def _agenticeda_axis_title(ax, axis_name):
    try:
        method_name = "get_xlabel" if axis_name == "x" else "get_ylabel"
        return str(getattr(ax, method_name)() or "").strip()
    except Exception:
        return ""

def _agenticeda_axis_series_label(ax):
    handles, labels = ax.get_legend_handles_labels()
    cleaned = [str(label).strip() for label in labels if str(label).strip() and not str(label).startswith("_")]
    return cleaned if cleaned else []

def _agenticeda_emit_fallback_plot_specs():
    global __agenticeda_plot_spec_emitted, __agenticeda_plot_spec_has_source
    if __agenticeda_plot_spec_has_source:
        return None

    try:
        fig_nums = list(plt.get_fignums())
    except Exception:
        return None

    for fig_num in fig_nums:
        try:
            fig = plt.figure(fig_num)
        except Exception:
            continue

        for axis_index, ax in enumerate(getattr(fig, "axes", []) or []):
            title = str(ax.get_title() or f"Figure {fig_num}.{axis_index + 1}").strip()
            x_label = _agenticeda_axis_title(ax, "x")
            y_label = _agenticeda_axis_title(ax, "y")
            legend_labels = _agenticeda_axis_series_label(ax)

            # Heatmap / image-like artists.
            images = list(getattr(ax, "images", []) or [])
            if images:
                image = images[0]
                try:
                    matrix = image.get_array()
                    z_values = matrix.tolist() if hasattr(matrix, "tolist") else list(matrix)
                    emit_plot_spec(
                        kind="plotly",
                        mime_type="application/vnd.plotly.v1+json",
                        title=title,
                        caption=title,
                        chart_family="heatmap",
                        semantic_intent="matrix",
                        x_axis_role="category",
                        y_axis_role="category",
                        source={
                            "data": [{
                                "type": "heatmap",
                                "name": title,
                                "z": z_values,
                            }],
                            "layout": {
                                "title": title,
                                "xaxis": {"title": x_label},
                                "yaxis": {"title": y_label},
                            },
                        },
                    )
                    continue
                except Exception:
                    pass

            # Boxplot detection — matplotlib boxplots produce patches (IQR boxes)
            # and lines (whiskers, caps, medians). Detect by checking if patches
            # are vertically-oriented rectangles at distinct x positions.
            patches = [p for p in list(getattr(ax, "patches", []) or []) if hasattr(p, "get_height") and hasattr(p, "get_width")]
            lines_raw = [line for line in list(ax.get_lines() or []) if len(line.get_xdata()) and len(line.get_ydata())]
            if patches and lines_raw:
                try:
                    # Group patches by x-center to find box groups
                    box_centers = {}
                    for p in patches:
                        cx = round(p.get_x() + p.get_width() / 2, 6)
                        h = abs(p.get_height())
                        w = abs(p.get_width())
                        if h > 0 and w > 0 and h > w * 0.3:
                            box_centers.setdefault(cx, []).append(p)
                    # If we have distinct box groups with narrow rectangles, it's a boxplot
                    if len(box_centers) >= 2 or (len(box_centers) == 1 and len(patches) <= 4):
                        box_traces = []
                        sorted_centers = sorted(box_centers.keys())
                        group_labels = legend_labels if len(legend_labels) == len(sorted_centers) else [f"Group {i+1}" for i in range(len(sorted_centers))]
                        for gi, cx in enumerate(sorted_centers):
                            box_patches = box_centers[cx]
                            # Collect all y-values from lines at this x-center for whisker/median detection
                            y_values_at_cx = []
                            for line in lines_raw:
                                xd = line.get_xdata().tolist()
                                yd = line.get_ydata().tolist()
                                if len(xd) == 2 and abs(xd[0] - cx) < 0.5 and abs(xd[1] - cx) < 0.5:
                                    y_values_at_cx.extend(yd)
                                elif len(xd) == 2 and abs(sum(xd)/2 - cx) < 0.5:
                                    y_values_at_cx.extend(yd)
                            # Get box bounds from patch
                            p = box_patches[0]
                            box_bottom = p.get_y()
                            box_top = box_bottom + p.get_height()
                            q1 = min(box_bottom, box_top)
                            q3 = max(box_bottom, box_top)
                            import math as _math
                            all_y = sorted(v for v in set(y_values_at_cx + [q1, q3])
                                           if isinstance(v, (int, float)) and _math.isfinite(v))
                            if not all_y:
                                continue
                            name = group_labels[gi] if gi < len(group_labels) else f"Group {gi+1}"
                            box_traces.append({
                                "type": "box",
                                "name": name,
                                "y": [_agenticeda_json_default(v) for v in all_y],
                                "boxpoints": False,
                            })
                        if box_traces:
                            emit_plot_spec(
                                kind="plotly",
                                mime_type="application/vnd.plotly.v1+json",
                                title=title,
                                caption=title,
                                chart_family="box",
                                semantic_intent="distribution",
                                x_axis_role="category",
                                y_axis_role="measure",
                                source={
                                    "data": box_traces,
                                    "layout": {
                                        "title": title,
                                        "xaxis": {"title": x_label},
                                        "yaxis": {"title": y_label},
                                    },
                                },
                            )
                            continue
                except Exception:
                    pass

            # Line charts.
            lines = [line for line in list(ax.get_lines() or []) if len(line.get_xdata()) and len(line.get_ydata())]
            if lines:
                # Detect boxplot artifacts: many short unlabeled lines = matplotlib boxplot components
                # (whiskers, caps, medians each produce a 2-point line with a "_"-prefixed label).
                # Skip emission here — the PNG image fallback will handle these correctly.
                all_short = all(len(line.get_xdata()) <= 5 for line in lines)
                all_unlabeled = all(
                    not str(line.get_label() or "").strip() or str(line.get_label() or "").startswith("_")
                    for line in lines
                )
                if len(lines) >= 6 and all_short and all_unlabeled:
                    continue  # Skip — boxplot artifacts, let PNG handle it

                traces = []
                for idx, line in enumerate(lines):
                    name = str(line.get_label() or "").strip()
                    if not name or name.startswith("_"):
                        name = legend_labels[idx] if idx < len(legend_labels) else f"Series {idx + 1}"
                    traces.append({
                        "type": "scatter",
                        "mode": "lines",
                        "name": name,
                        "x": [_agenticeda_json_default(v) for v in line.get_xdata().tolist()],
                        "y": [_agenticeda_json_default(v) for v in line.get_ydata().tolist()],
                    })
                emit_plot_spec(
                    kind="plotly",
                    mime_type="application/vnd.plotly.v1+json",
                    title=title,
                    caption=title,
                    chart_family="line",
                    semantic_intent="trend",
                    x_axis_role="numeric",
                    y_axis_role="measure",
                    source={
                        "data": traces,
                        "layout": {
                            "title": title,
                            "xaxis": {"title": x_label},
                            "yaxis": {"title": y_label},
                        },
                    },
                )
                continue

            # Scatter charts.
            collections = list(getattr(ax, "collections", []) or [])
            scatter_traces = []
            for idx, collection in enumerate(collections):
                get_offsets = getattr(collection, "get_offsets", None)
                if not callable(get_offsets):
                    continue
                try:
                    offsets = get_offsets()
                except Exception:
                    continue
                if offsets is None or len(offsets) == 0:
                    continue
                x_vals = [_agenticeda_json_default(row[0]) for row in offsets]
                y_vals = [_agenticeda_json_default(row[1]) for row in offsets]
                name = legend_labels[idx] if idx < len(legend_labels) else f"Series {idx + 1}"
                scatter_traces.append({
                    "type": "scatter",
                    "mode": "markers",
                    "name": name,
                    "x": x_vals,
                    "y": y_vals,
                })
            if scatter_traces:
                emit_plot_spec(
                    kind="plotly",
                    mime_type="application/vnd.plotly.v1+json",
                    title=title,
                    caption=title,
                    chart_family="scatter",
                    semantic_intent="relationship",
                    x_axis_role="numeric",
                    y_axis_role="numeric",
                    source={
                        "data": scatter_traces,
                        "layout": {
                            "title": title,
                            "xaxis": {"title": x_label},
                            "yaxis": {"title": y_label},
                        },
                    },
                )
                continue

            # Bar / histogram charts.
            patches = [patch for patch in list(getattr(ax, "patches", []) or []) if hasattr(patch, "get_height")]
            if patches:
                centers = []
                heights = []
                for patch in patches:
                    try:
                        centers.append(_agenticeda_json_default(patch.get_x() + patch.get_width() / 2))
                        heights.append(_agenticeda_json_default(patch.get_height()))
                    except Exception:
                        continue
                if centers and heights:
                    family = "histogram" if "count" in y_label.lower() or "distribution" in title.lower() else "bar"
                    intent = "distribution" if family == "histogram" else "comparison"
                    emit_plot_spec(
                        kind="plotly",
                        mime_type="application/vnd.plotly.v1+json",
                        title=title,
                        caption=title,
                        chart_family=family,
                        semantic_intent=intent,
                        x_axis_role="numeric",
                        y_axis_role="count" if family == "histogram" else "measure",
                        source={
                            "data": [{
                                "type": "bar",
                                "name": title,
                                "x": centers,
                                "y": heights,
                            }],
                            "layout": {
                                "title": title,
                                "xaxis": {"title": x_label},
                                "yaxis": {"title": y_label},
                            },
                        },
                    )
    return None

_agenticeda_original_show = plt.show
def _agenticeda_show(*args, **kwargs):
    try:
        _agenticeda_emit_fallback_plot_specs()
    except Exception:
        pass
    return _agenticeda_original_show(*args, **kwargs)
plt.show = _agenticeda_show
""")

    _LOG.info("Kernel started for session %s (cwd=%s)", session_id, cwd)
    return client


def _execute_silent(client: jupyter_client.KernelClient, code: str, timeout: int = 30):
    """Execute code without collecting output (for setup)."""
    msg_id = client.execute(code, silent=True)
    # Wait for completion
    while True:
        try:
            msg = client.get_iopub_msg(timeout=timeout)
            if msg["parent_header"].get("msg_id") == msg_id and msg["msg_type"] == "status":
                if msg["content"]["execution_state"] == "idle":
                    break
        except queue.Empty:
            break


def execute_code(
    session_id: str,
    code: str,
    timeout: int = 60,
    cell_id: str | None = None,
) -> tuple[list[dict[str, Any]], str | None]:
    """
    Execute code in the session's kernel and return structured outputs.

    Returns: (outputs, error)
        outputs: list of output dicts with output_type, text, data, etc.
        error: error string if execution failed, None otherwise
    """
    client = get_or_create_kernel(session_id)
    return _execute_on_client(client, code, timeout=timeout, cell_id=cell_id)


def shutdown_kernel(session_id: str):
    """Shut down the kernel for a session."""
    client = _clients.pop(session_id, None)
    km = _kernels.pop(session_id, None)

    if client:
        try:
            client.stop_channels()
        except Exception:
            pass

    if km:
        try:
            km.shutdown_kernel(now=True)
        except Exception:
            pass

    _LOG.info("Kernel shut down for session %s", session_id)


def is_kernel_alive(session_id: str) -> bool:
    """Check if a session has an active kernel."""
    km = _kernels.get(session_id)
    return km is not None and km.is_alive()


def get_kernel_connection_file(session_id: str) -> str | None:
    """Return the connection file path for an existing kernel (for child-process use)."""
    km = _kernels.get(session_id)
    if km is None or not km.is_alive():
        return None
    return km.connection_file


def execute_code_on_connection(
    connection_file: str,
    code: str,
    timeout: int = 60,
    cell_id: str | None = None,
) -> tuple[list[dict[str, Any]], str | None]:
    """Execute code on a kernel given its connection file (safe for child processes).

    NOTE: Creates a new BlockingKernelClient per call. For repeated calls in
    subagent loops, prefer PersistentKernelConnection to avoid the ZMQ
    slow-joiner problem (iopub messages lost before subscription establishes).
    """
    try:
        client = jupyter_client.BlockingKernelClient()
        client.load_connection_file(connection_file)
        client.start_channels()
    except Exception as exc:
        _LOG.warning("Kernel connection failed: %s", exc)
        return [], f"Kernel connection failed: {exc}"
    try:
        try:
            client.wait_for_ready(timeout=10)
        except RuntimeError as exc:
            client.stop_channels()
            _LOG.warning("Kernel not ready: %s", exc)
            return [], f"Kernel not ready: {exc}"
        return _execute_on_client(client, code, timeout=timeout, cell_id=cell_id)
    finally:
        client.stop_channels()


def _execute_on_client(
    client: jupyter_client.BlockingKernelClient,
    code: str,
    timeout: int = 60,
    cell_id: str | None = None,
) -> tuple[list[dict[str, Any]], str | None]:
    """Core execution logic on an already-connected client."""
    if cell_id:
        code = (
            f'__agenticeda_cell_id = {json.dumps(cell_id)}\n'
            "__agenticeda_plot_spec_emitted = 0\n"
            "__agenticeda_plot_spec_has_source = False\n"
            f"{code}\n"
            "_agenticeda_emit_fallback_plot_specs()\n"
        )
    # Flush stale iopub messages before executing
    while True:
        try:
            client.get_iopub_msg(timeout=0.05)
        except queue.Empty:
            break
    msg_id = client.execute(code)
    outputs: list[dict[str, Any]] = []
    error: str | None = None
    while True:
        try:
            msg = client.get_iopub_msg(timeout=timeout)
        except queue.Empty:
            error = f"Execution timed out after {timeout}s"
            break
        if msg["parent_header"].get("msg_id") != msg_id:
            continue
        msg_type = msg["msg_type"]
        content = msg["content"]
        if msg_type == "stream":
            outputs.append({"output_type": "stream", "name": content.get("name", "stdout"), "text": content.get("text", "")})
        elif msg_type in ("execute_result", "display_data"):
            data = content.get("data", {})
            clean_data = {}
            for k, v in data.items():
                clean_data[k] = v if isinstance(v, str) else "".join(v) if isinstance(v, list) else str(v)
            outputs.append({"output_type": msg_type, "data": clean_data, "metadata": content.get("metadata", {})})
        elif msg_type == "error":
            error = f"{content.get('ename', 'Error')}: {content.get('evalue', '')}"
            outputs.append({"output_type": "error", "ename": content.get("ename", ""), "evalue": content.get("evalue", ""), "traceback": content.get("traceback", [])})
        elif msg_type == "status":
            if content["execution_state"] == "idle":
                break
    return outputs, error


class PersistentKernelConnection:
    """Reusable kernel connection for subagent processes.

    Avoids the ZMQ slow-joiner problem by creating the BlockingKernelClient
    once and reusing it across multiple execute() calls, keeping the iopub
    subscription alive throughout the subagent's lifetime.
    """

    def __init__(self, connection_file: str):
        self._client = jupyter_client.BlockingKernelClient()
        self._client.load_connection_file(connection_file)
        self._client.start_channels()
        self._client.wait_for_ready(timeout=15)
        # Flush any stale iopub messages from kernel startup
        while True:
            try:
                self._client.get_iopub_msg(timeout=0.2)
            except queue.Empty:
                break

    def execute(
        self, code: str, timeout: int = 60, cell_id: str | None = None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        return _execute_on_client(self._client, code, timeout=timeout, cell_id=cell_id)

    def close(self):
        try:
            self._client.stop_channels()
        except Exception:
            pass
