"""
Utility library for deep_agents.API.ipynb.

All reusable functions extracted from the notebook live here, organised in the
same order as the notebook sections they belong to.

Import as:
    import deep_agents_API_utils as ut
"""

import logging
from pathlib import Path

from langchain_API_utils import get_chat_model  # noqa: F401

_LOG = logging.getLogger(__name__)


# ##############################################################################
# Deep Agents: check imports
# ##############################################################################


def check_deepagents() -> str:
    """
    Check that deepagents is importable and return its version string.
    """
    try:
        import deepagents  # type: ignore

        return getattr(deepagents, "__version__", "(unknown)")
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "This section requires `deepagents`.\n"
            f"Import error: {type(e).__name__}: {str(e)[:200]}"
        )


# ##############################################################################
# Deep Agents: Filesystem backend demo
# ##############################################################################


def create_filesystem_agent(model, root_dir: str | None = None):
    """
    Create a Deep Agents agent backed by a virtual FilesystemBackend.
    """
    from deepagents import create_deep_agent  # type: ignore
    from deepagents.backends import FilesystemBackend  # type: ignore

    if root_dir is None:
        root_dir = str(Path(".").resolve())
    Path("workspace").mkdir(parents=True, exist_ok=True)
    backend = FilesystemBackend(root_dir=root_dir, virtual_mode=True)
    return create_deep_agent(model=model, backend=backend)


def run_filesystem_demo(model) -> tuple[list[str], str]:
    """
    Run the filesystem write/read demo.

    Returns (paths_on_disk, final_message_preview).
    """
    agent = create_filesystem_agent(model)
    out = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Call write_file with file_path='/workspace/hello.txt' and content='hello'. "
                        "Then call read_file on '/workspace/hello.txt' and return the content."
                    ),
                }
            ]
        }
    )
    paths = sorted([str(p) for p in Path("workspace").rglob("hello.txt")])
    preview = getattr(out["messages"][-1], "content", "")[:200]
    return paths, preview


# ##############################################################################
# Deep Agents: Human-in-the-loop (HITL) demo
# ##############################################################################


def run_hitl_demo(model, thread_id: str = "API_HITL_GUARDRAIL"):
    """
    Run the HITL approve-flow demo.

    Returns (agent, final_out, was_interrupted).
    """
    from deepagents import create_deep_agent  # type: ignore
    from deepagents.backends import FilesystemBackend  # type: ignore
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import Command

    try:
        from langchain.agents.middleware.human_in_the_loop import InterruptOnConfig
    except ModuleNotFoundError as e:  # pragma: no cover
        raise RuntimeError(
            "This Deep Agents HITL demo needs the tutorial dependencies.\n\n"
            "Run it from `tutorials/LangChain_LangGraph` with `requirements.txt` installed (or via Docker)."
        ) from e

    root_dir = str(Path(".").resolve())
    backend = FilesystemBackend(root_dir=root_dir, virtual_mode=True)
    agent = create_deep_agent(
        model=model,
        checkpointer=MemorySaver(),
        backend=backend,
        interrupt_on={
            "edit_file": InterruptOnConfig(allowed_decisions=["approve", "reject"])
        },
    )
    agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "write_file /workspace/hitl_api_demo.txt with 'line1\nline2\n'",
                }
            ]
        },
        config={"configurable": {"thread_id": thread_id}},
    )
    out = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "edit_file /workspace/hitl_api_demo.txt replace 'line2' with "
                        "'LINE2_APPROVED' then read_file /workspace/hitl_api_demo.txt"
                    ),
                }
            ]
        },
        config={"configurable": {"thread_id": thread_id}},
    )
    interrupted = "__interrupt__" in out
    if interrupted:
        out = agent.invoke(
            Command(resume={"decisions": [{"type": "approve"}]}),
            config={"configurable": {"thread_id": thread_id}},
        )
    return agent, out, interrupted
