"""
Agents that answer benchmark tasks.

Each agent gets one `benchmarks.Task` and a private working directory that holds
copies of the task's data files, and returns a `TaskResult` with its final
answer plus cost and usage. The runner (`run_benchmarks.py`) scores the answer.

| Name        | Agent                                  | Model                        |
| :---------- | :------------------------------------- | :--------------------------- |
| claude_code | Claude Code, through the Agent SDK     | `BAKEOFF_MODEL` or default   |
| smolagents  | Hugging Face smolagents `CodeAgent`    | `BAKEOFF_MODEL` (LiteLLM id) |
| oracle      | Returns the gold answer (testing only) | none                         |

All LLM agents receive the same prompt (`build_prompt`) and must end with
`FINAL ANSWER: ...`, so differences come from the agent, not the instructions.
"""

import abc
import asyncio
import logging
import os
import re
import time
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type

import research.agentic_data_science.bakeoff.benchmarks as radsbben

_LOG = logging.getLogger(__name__)

# Model used by every LLM agent, so the comparison is between agents.
MODEL_ENV = "BAKEOFF_MODEL"

_PROMPT = """You are a data scientist. Answer the question below using the data files.

Data files (in your working directory):
{files}

Question:
{question}

You can write and run Python code (pandas, numpy, scipy, statsmodels, sklearn).
Work only inside your working directory. {finish}
"""

# How an agent hands back its answer.
FINISH_MARKER = (
    "When you are done, end your reply with one line in exactly this form:\n"
    "FINAL ANSWER: <your answer, in the format the question asks for>"
)
FINISH_TOOL = (
    "When you are done, call final_answer with your answer, in the format the "
    "question asks for."
)


@dataclass
class TaskResult:
    """
    What an agent returns for one task.

    :param answer: final answer text; empty if the agent gave none
    :param raw: full final message of the agent, for failure analysis
    :param tokens_in: input tokens used, if known
    :param tokens_out: output tokens used, if known
    :param cost_usd: cost in US dollars, if known
    :param runtime_s: wall-clock seconds, filled by `TaskAgent.timed_answer`
    :param error: None on success, otherwise the error text
    :param model: model id the agent used
    """

    answer: str = ""
    raw: str = ""
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None
    cost_usd: Optional[float] = None
    runtime_s: Optional[float] = None
    error: Optional[str] = None
    model: Optional[str] = None
    extra: Dict = field(default_factory=dict)


def build_prompt(
    task: radsbben.Task, workdir: str, finish: str = FINISH_MARKER
) -> str:
    """
    Build the prompt every LLM agent receives.

    :param finish: how to hand back the answer; only this part differs by agent
    """
    files = [os.path.relpath(f, workdir) for f in workdir_files(workdir)]
    listing = "\n".join(f"- {f}" for f in files) or "- (none)"
    return _PROMPT.format(files=listing, question=task.question, finish=finish)


def workdir_files(workdir: str) -> List[str]:
    out = []
    for dirpath, _, names in os.walk(workdir):
        out.extend(os.path.join(dirpath, n) for n in sorted(names))
    return sorted(out)


def extract_final_answer(text: str) -> str:
    """
    Return what follows the last `FINAL ANSWER:` marker, or "" if there is none.
    """
    matches = re.findall(r"FINAL ANSWER:\s*(.*)", text or "", re.IGNORECASE)
    if not matches:
        return ""
    return matches[-1].strip().strip("`").strip()


# #############################################################################
# TaskAgent
# #############################################################################


class TaskAgent(abc.ABC):
    """
    Base class of every agent that answers benchmark tasks.
    """

    name: str = "task_agent"

    @abc.abstractmethod
    def answer(self, task: radsbben.Task, workdir: str) -> TaskResult:
        """
        Answer `task`, reading its data files from `workdir`.
        """

    def timed_answer(self, task: radsbben.Task, workdir: str) -> TaskResult:
        """
        Run `answer`, timing it and recording a crash as a failed result.
        """
        start = time.perf_counter()
        try:
            res = self.answer(task, workdir)
        except Exception:  # pylint: disable=broad-except
            res = TaskResult(error=traceback.format_exc(limit=5))
        res.runtime_s = round(time.perf_counter() - start, 2)
        return res


class OracleAgent(TaskAgent):
    """
    Answer with the gold answer: checks the pipeline end to end without an LLM.

    Never report its score as a result.
    """

    name = "oracle"

    def answer(self, task: radsbben.Task, workdir: str) -> TaskResult:
        return TaskResult(answer=radsbben.gold_as_prediction(task), model="none")


# #############################################################################
# Claude Code
# #############################################################################


class ClaudeCodeAgent(TaskAgent):
    """
    Claude Code running headless through the Claude Agent SDK.

    It uses the user's Claude Code login or `ANTHROPIC_API_KEY`. User settings and
    CLAUDE.md files are not loaded, so every run starts from the same state.
    """

    name = "claude_code"

    def __init__(
        self, max_turns: int = 30, timeout_s: int = 600, max_budget_usd: float = 1.0
    ) -> None:
        self.max_turns = max_turns
        self.timeout_s = timeout_s
        self.max_budget_usd = max_budget_usd
        self.model = os.environ.get(MODEL_ENV) or None

    def answer(self, task: radsbben.Task, workdir: str) -> TaskResult:
        return asyncio.run(
            asyncio.wait_for(self._answer(task, workdir), timeout=self.timeout_s)
        )

    async def _answer(self, task: radsbben.Task, workdir: str) -> TaskResult:
        import claude_agent_sdk as cas

        options = cas.ClaudeAgentOptions(
            cwd=workdir,
            allowed_tools=["Bash", "Read", "Write", "Edit", "Glob", "Grep"],
            max_turns=self.max_turns,
            max_budget_usd=self.max_budget_usd,
            model=self.model,
            setting_sources=[],
        )
        final: Optional[cas.ResultMessage] = None
        async for msg in cas.query(prompt=build_prompt(task, workdir), options=options):
            if isinstance(msg, cas.ResultMessage):
                final = msg
        if final is None:
            return TaskResult(error="No result message from Claude Code")
        usage = final.usage or {}
        raw = final.result or ""
        return TaskResult(
            answer=extract_final_answer(raw),
            raw=raw,
            tokens_in=_sum_keys(usage, "input_tokens", "cache_read_input_tokens",
                                "cache_creation_input_tokens"),
            tokens_out=usage.get("output_tokens"),
            cost_usd=final.total_cost_usd,
            error=f"Claude Code: {final.subtype}" if final.is_error else None,
            model=self.model or ",".join(sorted(final.model_usage or {})) or None,
            extra={"num_turns": final.num_turns},
        )


def _sum_keys(d: Dict, *keys: str) -> Optional[int]:
    vals = [d[k] for k in keys if isinstance(d.get(k), int)]
    return sum(vals) if vals else None


# #############################################################################
# smolagents
# #############################################################################


class SmolagentsAgent(TaskAgent):
    """
    Hugging Face smolagents `CodeAgent`: it acts by writing Python code.

    The model is a LiteLLM id from `BAKEOFF_MODEL` (e.g. an `anthropic/...` id);
    the matching API key must be set (e.g. `ANTHROPIC_API_KEY`).
    """

    name = "smolagents"

    _IMPORTS = [
        "pandas", "numpy", "scipy", "scipy.*", "statsmodels", "statsmodels.*",
        "sklearn", "sklearn.*", "json", "math", "statistics", "re", "os",
        "os.path", "collections", "datetime", "itertools",
    ]

    def __init__(self, max_steps: int = 20) -> None:
        self.max_steps = max_steps
        self.model_id = os.environ.get(MODEL_ENV)

    def answer(self, task: radsbben.Task, workdir: str) -> TaskResult:
        if not self.model_id:
            return TaskResult(error=f"Set {MODEL_ENV} to a LiteLLM model id")
        import smolagents

        agent = smolagents.CodeAgent(
            tools=[],
            model=smolagents.LiteLLMModel(model_id=self.model_id),
            additional_authorized_imports=self._IMPORTS,
            max_steps=self.max_steps,
            verbosity_level=0,
        )
        prompt = build_prompt(task, workdir, finish=FINISH_TOOL)
        # The local executor resolves relative paths against the process cwd.
        cwd = os.getcwd()
        os.chdir(workdir)
        try:
            out = agent.run(prompt)
        finally:
            os.chdir(cwd)
        counts = agent.monitor.get_total_token_counts()
        return TaskResult(
            answer=str(out).strip(),
            raw=str(out),
            tokens_in=counts.input_tokens,
            tokens_out=counts.output_tokens,
            cost_usd=_litellm_cost(self.model_id, counts.input_tokens, counts.output_tokens),
            model=self.model_id,
        )


def _litellm_cost(model: str, tokens_in: int, tokens_out: int) -> Optional[float]:
    try:
        import litellm

        cin, cout = litellm.cost_per_token(
            model=model, prompt_tokens=tokens_in, completion_tokens=tokens_out
        )
        return round(cin + cout, 6)
    except Exception:  # pylint: disable=broad-except
        return None


# #############################################################################
# Registry
# #############################################################################


TASK_AGENTS: Dict[str, Type[TaskAgent]] = {
    "claude_code": ClaudeCodeAgent,
    "smolagents": SmolagentsAgent,
    "oracle": OracleAgent,
}


def get_task_agent(name: str) -> TaskAgent:
    if name not in TASK_AGENTS:
        raise KeyError(f"Unknown agent '{name}'. Known: {sorted(TASK_AGENTS)}")
    return TASK_AGENTS[name]()
