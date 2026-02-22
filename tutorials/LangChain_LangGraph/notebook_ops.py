# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Notebook ops: nbformat + nbclient + artifacts + papermill
#
# Notebooks are just JSON documents.
# That means you can:
# - generate them (`nbformat`)
# - execute them programmatically (`nbclient`)
# - collect outputs and errors
# - parameterize runs (`papermill`)
#
# Why include this in a LangChain/LangGraph tutorial?
# Because “agents that write and run notebooks” is a surprisingly practical workflow for data work.
# We’ll keep the demos safe: everything writes under `tmp_runs/`.
#

# %%
# This cell will:
# - Demonstrate notebook operations (write/execute/parameterize notebooks).
from pathlib import Path

import nbformat
from nbformat import validate
from nbclient import NotebookClient

run_dir = Path("tmp_runs").resolve()
run_dir.mkdir(parents=True, exist_ok=True)

nb = nbformat.v4.new_notebook()
nb.cells = [
    nbformat.v4.new_markdown_cell("# nbclient smoke test"),
    nbformat.v4.new_code_cell("x = 2 + 3\nprint(x)"),
    nbformat.v4.new_code_cell("import math\nprint(math.sqrt(81))"),
]
validate(nb)

in_path = run_dir / "smoke_in.ipynb"
out_path = run_dir / "smoke_out.ipynb"
nbformat.write(nb, str(in_path))

nb2 = nbformat.read(str(in_path), as_version=4)
client = NotebookClient(
    nb2, resources={"metadata": {"path": str(run_dir)}}, timeout=60
)
client.execute()
nbformat.write(nb2, str(out_path))

str(out_path)


# %% [markdown]
# ### Write a notebook via a tool (from a spec)
#
# We’ll build a tiny notebook in memory (a title + a code cell), then write it to disk.
#
# This is the first building block for “notebook automation” — generating a notebook artifact from a structured spec.
#

# %%
# This cell will:
# - Demonstrate notebook operations (write/execute/parameterize notebooks).
# write_notebook is defined in langchain.API_utils.
spec = {
    "cells": [
        {"type": "markdown", "source": "# Tool-written notebook"},
        {"type": "code", "source": "print('ok')"},
    ]
}
ut.write_notebook.invoke({"spec": spec, "out_rel": "demo/tool_hello.ipynb"})


# %% [markdown]
# ### Notebook ops as tools + secure injected workspace (ToolNode)
#
# Here we treat notebook operations as **tools** inside a LangGraph workflow.
#
# The important idea:
# - tools can be powerful (file access, execution)
# - so we often want a *controlled* workspace root
#
# You’ll see us use an injected workspace directory so the graph can safely read/write only where we intend.
#

# %%
# This cell will:
# - Build and compile a `StateGraph` (a small LangGraph workflow).
# - Use `ToolNode` to execute tool calls inside a graph.
# - Demonstrate notebook operations (write/execute/parameterize notebooks).
# nb_write, nb_run, nb_extract_errors, nb_extract_artifacts, nb_list_files, ToolGraphState
# are defined in langchain.API_utils.
from pathlib import Path

from langchain_core.messages import AIMessage
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode

workspace = Path("tmp_runs/ipynb_tools_workspace").resolve()
workspace.mkdir(parents=True, exist_ok=True)

tool_node = ToolNode(
    [
        ut.nb_write,
        ut.nb_run,
        ut.nb_extract_errors,
        ut.nb_extract_artifacts,
        ut.nb_list_files,
    ]
)
g = StateGraph(ut.ToolGraphState)
g.add_node("tools", tool_node)
g.add_edge(START, "tools")
g.add_edge("tools", END)
graph = g.compile()

spec = {
    "cells": [
        {"type": "markdown", "source": "# Tool-made notebook"},
        {"type": "code", "source": "print('hello')"},
    ]
}

# IMPORTANT: Tool calls in a single ToolNode are not a dependency graph.
# Execute dependent operations in separate invocations for deterministic behavior.

out1 = graph.invoke(
    {
        "workspace_dir": str(workspace),
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "nb_write",
                        "args": {"spec": spec, "out_rel": "demo/in.ipynb"},
                        "id": "t1",
                        "type": "tool_call",
                    },
                ],
            )
        ],
    }
)

out2 = graph.invoke(
    {
        "workspace_dir": str(workspace),
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "nb_run",
                        "args": {
                            "in_rel": "demo/in.ipynb",
                            "out_rel": "demo/out.executed.ipynb",
                            "timeout_s": 60,
                        },
                        "id": "t2",
                        "type": "tool_call",
                    },
                ],
            )
        ],
    }
)

out3 = graph.invoke(
    {
        "workspace_dir": str(workspace),
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "nb_list_files",
                        "args": {},
                        "id": "t3",
                        "type": "tool_call",
                    },
                ],
            )
        ],
    }
)

(
    out1["messages"][-1].content,
    out2["messages"][-1].content,
    out3["messages"][-1].content[:200],
)


# %% [markdown]
# ### Execute notebooks + collect errors
#
# We’ll execute a notebook programmatically and capture:
# - stdout
# - execution errors (if any)
#
# This is a friendly way to build “run this notebook and report back” pipelines.
#

# %%
# This cell will:
# - Demonstrate notebook operations (write/execute/parameterize notebooks).
# extract_errors is defined in langchain.API_utils.
from pathlib import Path

import nbformat
from nbformat import validate
from nbclient import NotebookClient

run_dir = Path("tmp_runs/execute").resolve()
run_dir.mkdir(parents=True, exist_ok=True)

# Notebook that errors.
nb_err = nbformat.v4.new_notebook()
nb_err.cells = [
    nbformat.v4.new_markdown_cell("# Intentional error"),
    nbformat.v4.new_code_cell("print('before')"),
    nbformat.v4.new_code_cell("1/0"),
    nbformat.v4.new_code_cell("print('after')"),
]
validate(nb_err)
in_path = run_dir / "error_in.ipynb"
out_path = run_dir / "error_out.executed.ipynb"
nbformat.write(nb_err, str(in_path))

nb = nbformat.read(str(in_path), as_version=4)
client = NotebookClient(
    nb,
    timeout=60,
    allow_errors=True,
    resources={"metadata": {"path": str(run_dir)}},
)
client.execute()
nbformat.write(nb, str(out_path))

ut.extract_errors(nb)


# %% [markdown]
# ### Extract artifacts from executed notebooks (stdout + inline images)
#
# Executed notebooks can contain rich outputs (plots, tables, HTML).
#
# We’ll show a simple approach to pull a couple useful artifacts out of the executed notebook:
# - printed output
# - embedded images
#

# %%
# This cell will:
# - Demonstrate notebook operations (write/execute/parameterize notebooks).
import base64
import json

run_dir = Path("tmp_runs/artifacts").resolve()
run_dir.mkdir(parents=True, exist_ok=True)

nb = nbformat.v4.new_notebook()
nb.cells = [
    nbformat.v4.new_markdown_cell("# Artifact notebook"),
    nbformat.v4.new_code_cell("print('hello from stdout')"),
    nbformat.v4.new_code_cell(
        "import matplotlib.pyplot as plt\n"
        "plt.plot([0,1,2],[0,1,4])\n"
        "plt.title('inline')\n"
        "plt.show()\n"
    ),
]
in_nb = run_dir / "artifacts_in.ipynb"
executed_nb = run_dir / "artifacts.executed.ipynb"
nbformat.write(nb, str(in_nb))

nb2 = nbformat.read(str(in_nb), as_version=4)
NotebookClient(
    nb2, timeout=120, resources={"metadata": {"path": str(run_dir)}}
).execute()
nbformat.write(nb2, str(executed_nb))

out_dir = run_dir / "out"
out_dir.mkdir(parents=True, exist_ok=True)
manifest = []

for i, cell in enumerate(nb2.cells):
    if cell.get("cell_type") != "code":
        continue
    for j, out in enumerate(cell.get("outputs", [])):
        if out.get("output_type") == "stream":
            txt = out.get("text", "")
            p = out_dir / f"cell_{i}_stream_{j}.txt"
            p.write_text(txt if isinstance(txt, str) else "".join(txt))
            manifest.append({"cell": i, "kind": "stream", "path": str(p)})
        if out.get("output_type") in ("display_data", "execute_result"):
            data = out.get("data", {})
            if "text/plain" in data:
                t = data["text/plain"]
                p = out_dir / f"cell_{i}_text_{j}.txt"
                p.write_text(t if isinstance(t, str) else "".join(t))
                manifest.append(
                    {"cell": i, "kind": "text/plain", "path": str(p)}
                )
            if "image/png" in data:
                b64 = data["image/png"]
                b = base64.b64decode(
                    b64 if isinstance(b64, str) else "".join(b64)
                )
                p = out_dir / f"cell_{i}_img_{j}.png"
                p.write_bytes(b)
                manifest.append({"cell": i, "kind": "image/png", "path": str(p)})

(out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
{
    "executed_nb": str(executed_nb),
    "n_artifacts": len(manifest),
    "manifest": str(out_dir / "manifest.json"),
}


# %% [markdown]
# ### Filesystem artifacts (notebooks that write files)
#
# Sometimes notebooks produce *real files* (CSVs, images, model outputs).
#
# In the next cell we execute a notebook that writes files into a run directory, then list what it produced.
# Everything stays under `tmp_runs/`.
#

# %%
# This cell will:
# - Demonstrate notebook operations (write/execute/parameterize notebooks).

run_dir = Path("tmp_runs/writes_files").resolve()
run_dir.mkdir(parents=True, exist_ok=True)

nb = nbformat.v4.new_notebook()
nb.cells = [
    nbformat.v4.new_markdown_cell("# Writes files"),
    nbformat.v4.new_code_cell(
        "import csv\n"
        "import matplotlib.pyplot as plt\n"
        "\n"
        "rows = [(i, i*i) for i in range(5)]\n"
        "with open('table.csv', 'w', newline='') as f:\n"
        "    w = csv.writer(f)\n"
        "    w.writerow(['x','y'])\n"
        "    w.writerows(rows)\n"
        "\n"
        "xs = [r[0] for r in rows]\n"
        "ys = [r[1] for r in rows]\n"
        "plt.plot(xs, ys)\n"
        "plt.title('y=x^2')\n"
        "plt.savefig('plot.png', dpi=120)\n"
        "print('wrote table.csv and plot.png')\n"
    ),
]

in_nb = run_dir / "writes_files.ipynb"
out_nb = run_dir / "writes_files.executed.ipynb"
nbformat.write(nb, str(in_nb))

nb2 = nbformat.read(str(in_nb), as_version=4)
NotebookClient(
    nb2, timeout=120, resources={"metadata": {"path": str(run_dir)}}
).execute()
nbformat.write(nb2, str(out_nb))

sorted([p.name for p in run_dir.iterdir() if p.is_file()])


# %% [markdown]
# ### Parameterized runs (Papermill)
#
# Papermill is a simple way to run the *same* notebook with different parameters.
#
# This is useful for:
# - experiments
# - scheduled reports
# - batch runs over multiple inputs
#
# We’ll do a tiny demo so you can see the mechanics.
#

# %%
# This cell will:
# - Demonstrate notebook operations (write/execute/parameterize notebooks).
import papermill as pm

run_dir = Path("tmp_runs/papermill").resolve()
run_dir.mkdir(parents=True, exist_ok=True)

nb = nbformat.v4.new_notebook()
nb.cells = [
    nbformat.v4.new_markdown_cell("# Papermill demo"),
    nbformat.v4.new_code_cell(
        "# Parameters\nx = 1\ny = 2", metadata={"tags": ["parameters"]}
    ),
    nbformat.v4.new_code_cell("print({'x': x, 'y': y, 'x_plus_y': x + y})"),
]
nb.metadata["kernelspec"] = {
    "name": "python3",
    "display_name": "Python 3",
    "language": "python",
}

in_nb = run_dir / "pm_in.ipynb"
out_nb = run_dir / "pm_out.ipynb"
nbformat.write(nb, str(in_nb))

pm.execute_notebook(
    str(in_nb),
    str(out_nb),
    parameters={"x": 10, "y": 32},
    cwd=str(run_dir),
    kernel_name="python3",
)
str(out_nb)

