"""
Deterministic tests for the housing analysis agent.

The language model is replaced by `ScriptedChatService`, which returns a
fixed sequence of function calls and a final answer. Everything else is real:
the `ChatCompletionAgent`, Semantic Kernel's automatic function invocation
loop, the housing plugin, the models, and the execution trace. The plugin
uses synthetic data, so no network access, Ollama server, or API key is
needed.
"""

import asyncio
import json
from collections.abc import Callable
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest
from pydantic import PrivateAttr
from semantic_kernel.connectors.ai.chat_completion_client_base import (
    ChatCompletionClientBase,
)
from semantic_kernel.contents import (
    AuthorRole,
    ChatHistory,
    ChatMessageContent,
    FunctionCallContent,
)

import housing_agent
import housing_kernel
from housing_plugins import housing_plugin
from housing_plugins.housing_plugin import HousingAnalysisPlugin

PLUGIN = housing_kernel.PLUGIN_NAME
N_ROWS = 60

# A scripted model turn: a list of function calls, or a function that builds
# the final answer from the chat history. A function call is a function name
# (called with `{}`) or a `(function name, raw JSON arguments)` pair.
Turn = list[str | tuple[str, str]] | Callable[[ChatHistory], str]


class ScriptedChatService(ChatCompletionClientBase):
    """Chat service that replays scripted turns instead of calling a model."""

    SUPPORTS_FUNCTION_CALLING: ClassVar[bool] = True
    _turns: list[Turn] = PrivateAttr(default_factory=list)
    _requests: int = PrivateAttr(default=0)
    _offered_functions: list[str] = PrivateAttr(default_factory=list)

    def __init__(self, turns: list[Turn]) -> None:
        super().__init__(ai_model_id="scripted-model")
        self._turns = list(turns)

    @property
    def requests(self) -> int:
        """Return the number of model requests received."""
        return self._requests

    @property
    def offered_functions(self) -> list[str]:
        """Return the functions advertised to the model as tools."""
        return self._offered_functions

    def _update_function_choice_settings_callback(self):
        def callback(configuration, settings, choice_type):
            self._offered_functions = sorted(
                function.fully_qualified_name
                for function in configuration.available_functions or []
            )

        return callback

    async def _inner_get_chat_message_contents(
        self, chat_history: ChatHistory, settings: Any
    ) -> list[ChatMessageContent]:
        turn = self._turns[self._requests]
        self._requests += 1
        if callable(turn):
            return [
                ChatMessageContent(
                    role=AuthorRole.ASSISTANT, content=turn(chat_history)
                )
            ]
        calls = []
        for index, call in enumerate(turn):
            name, arguments = call if isinstance(call, tuple) else (call, "{}")
            calls.append(
                FunctionCallContent(
                    id=f"call-{self._requests}-{index}",
                    plugin_name=PLUGIN,
                    function_name=name,
                    arguments=arguments,
                )
            )
        return [ChatMessageContent(role=AuthorRole.ASSISTANT, items=calls)]


def _synthetic_data() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        rng.normal(size=(N_ROWS, 6)),
        columns=[f"X{i}" for i in range(1, 7)],
    )
    y = pd.Series(
        3 * X["X1"] - 2 * X["X3"] + rng.normal(scale=0.1, size=N_ROWS),
        name="price",
    )
    return X, y


class CountingLoader:
    """Data loader that counts how often the plugin loads data."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> tuple[pd.DataFrame, pd.Series]:
        self.calls += 1
        return _synthetic_data()


def _failing_loader() -> tuple[pd.DataFrame, pd.Series]:
    raise ConnectionError("dataset download failed")


def _tool_results(chat_history: ChatHistory) -> list[str]:
    """Return the function results that were sent back to the model."""
    return [
        str(item.result)
        for message in chat_history.messages
        if message.role == AuthorRole.TOOL
        for item in message.items
    ]


def _run(prompt: str, turns: list[Turn], loader=None, **kwargs):
    service = ScriptedChatService(turns)
    plugin = HousingAnalysisPlugin(data_loader=loader or _synthetic_data)
    kernel = housing_kernel.create_kernel(plugin)
    result = asyncio.run(
        housing_agent.run_agent(
            prompt, service=service, kernel=kernel, **kwargs
        )
    )
    return result, service


def test_agent_offers_only_existing_housing_functions():
    """The model sees exactly the two existing plugin functions as tools."""
    _, service = _run("Hello", [lambda history: "Hi."])

    assert service.offered_functions == [
        f"{PLUGIN}-compare_models",
        f"{PLUGIN}-dataset_summary",
    ]


def test_agent_invokes_the_function_chosen_by_the_model():
    """A requested dataset_summary call runs and its result reaches the model."""
    loader = CountingLoader()

    def answer(history: ChatHistory) -> str:
        summary = json.loads(_tool_results(history)[-1])
        return f"The dataset has {summary['rows']} rows."

    result, service = _run(
        "How many rows does the dataset have?",
        [["dataset_summary"], answer],
        loader,
    )

    assert result.trace.function_names == [f"{PLUGIN}.dataset_summary"]
    assert result.trace.records[0].status == "ok"
    assert result.answer == f"The dataset has {N_ROWS} rows."
    # Only the selected function ran: compare_models was not invoked.
    assert loader.calls == 1
    assert service.requests == 2


def test_compare_models_result_is_returned_to_the_model():
    def answer(history: ChatHistory) -> str:
        scores = json.loads(_tool_results(history)[-1])
        return ", ".join(sorted(scores))

    result, _ = _run("Which model is better?", [["compare_models"], answer])

    assert result.trace.function_names == [f"{PLUGIN}.compare_models"]
    assert result.answer == "Linear Regression, Random Forest"
    assert "Random Forest" in result.trace.records[0].result_preview


def test_trace_records_multi_step_order():
    """Calls in consecutive model requests are recorded in order."""
    result, _ = _run(
        "Describe the data, then compare the models.",
        [["dataset_summary"], ["compare_models"], lambda history: "Done."],
    )

    assert result.trace.function_names == [
        f"{PLUGIN}.dataset_summary",
        f"{PLUGIN}.compare_models",
    ]
    assert [r.step for r in result.trace.records] == [1, 2]
    assert [r.request_index for r in result.trace.records] == [0, 1]
    assert all(r.status == "ok" for r in result.trace.records)


def test_trace_records_function_errors():
    """A failing function is traced as an error and reported to the model."""

    def answer(history: ChatHistory) -> str:
        return _tool_results(history)[-1]

    result, _ = _run(
        "Summarize the dataset.", [["dataset_summary"], answer], _failing_loader
    )

    record = result.trace.records[0]
    assert record.status == "error"
    assert record.error == "ConnectionError: dataset download failed"
    assert "dataset download failed" in result.answer


def test_answer_without_function_calls_has_empty_trace():
    result, _ = _run("What is your name?", [lambda history: "HousingAnalyst."])

    assert result.trace.records == []
    assert result.trace.format() == "No functions were invoked."
    assert result.answer == "HousingAnalyst."


def test_trace_format_lists_calls():
    result, _ = _run(
        "How many rows?", [["dataset_summary"], lambda history: "60."]
    )

    text = result.trace.format()

    assert text.startswith(f"1. {PLUGIN}.dataset_summary({{}}) [request 0]")
    assert '"rows": 60' in text


def test_trace_to_json_keeps_full_results():
    result, _ = _run(
        "Compare the models.", [["compare_models"], lambda history: "Done."]
    )

    records = json.loads(result.trace.to_json())

    assert records[0]["function"] == "compare_models"
    assert records[0]["status"] == "ok"
    # The full result is kept, not only the preview.
    assert json.loads(records[0]["result"]).keys() == {
        "Linear Regression",
        "Random Forest",
    }


def test_agent_uses_explicit_auto_invoke_limit():
    agent = housing_agent.create_agent(
        ScriptedChatService([]), housing_kernel.create_kernel()
    )

    behavior = agent.function_choice_behavior
    assert behavior.maximum_auto_invoke_attempts == 5
    assert behavior.auto_invoke_kernel_functions


def test_auto_invoke_limit_stops_repeated_calls():
    """A model that keeps requesting tools is stopped after the limit."""
    turns = [["dataset_summary"]] * 2 + [lambda history: "Stopped."]

    result, service = _run(
        "Loop.", turns, max_auto_invoke_attempts=2
    )

    # Two function-calling requests, then one final request without tools.
    assert len(result.trace.records) == 2
    assert service.requests == 3
    assert result.answer == "Stopped."


def test_plugin_caches_dataset_and_model_results(monkeypatch):
    """Repeated calls load the data and train the models only once."""
    loader = CountingLoader()
    trainings = []
    original_train_models = housing_plugin.train_models

    def counting_train_models(X, y):
        trainings.append(1)
        return original_train_models(X, y)

    monkeypatch.setattr(housing_plugin, "train_models", counting_train_models)
    plugin = HousingAnalysisPlugin(data_loader=loader)

    summaries = [plugin.dataset_summary() for _ in range(2)]
    comparisons = [plugin.compare_models() for _ in range(2)]

    assert loader.calls == 1
    assert len(trainings) == 1
    assert summaries[0] == summaries[1]
    assert comparisons[0] == comparisons[1]


def test_plugin_does_not_cache_failed_loads():
    attempts = []

    def flaky_loader():
        attempts.append(1)
        if len(attempts) == 1:
            raise ConnectionError("temporary failure")
        return _synthetic_data()

    plugin = HousingAnalysisPlugin(data_loader=flaky_loader)

    with pytest.raises(ConnectionError):
        plugin.dataset_summary()
    summary = json.loads(plugin.dataset_summary())

    assert summary["rows"] == N_ROWS
    assert len(attempts) == 2


def test_multi_step_run_loads_data_once():
    loader = CountingLoader()

    result, _ = _run(
        "Describe the data, then compare the models.",
        [["dataset_summary"], ["compare_models"], lambda history: "Done."],
        loader,
    )

    assert len(result.trace.records) == 2
    assert loader.calls == 1


def test_rejected_arguments_are_recorded_as_attempts():
    """
    Reproduce the Groq case: `{"": {}}` arguments for a function without
    parameters are rejected by Semantic Kernel before any filter runs; the
    model then retries with `{}`.
    """
    result, service = _run(
        "How many rows?",
        [
            [("dataset_summary", '{"": {}}')],
            [("dataset_summary", "{}")],
            lambda history: "60 rows.",
        ],
    )

    # Only the second request executed the function.
    assert result.trace.function_names == [f"{PLUGIN}.dataset_summary"]
    assert result.trace.records[0].request_index == 1
    assert service.requests == 3
    attempts = result.trace.attempts
    assert [(a.request_index, a.outcome) for a in attempts] == [
        (0, "rejected"),
        (1, "executed"),
    ]
    rejected = attempts[0]
    assert rejected.name == f"{PLUGIN}-dataset_summary"
    assert rejected.arguments == '{"": {}}'
    assert rejected.status is None
    assert "unexpected argument" in rejected.reason
    assert attempts[1].status == "ok"
    assert "Rejected function calls (not executed):" in result.trace.format()


def test_unknown_function_is_recorded_as_rejected():
    result, _ = _run(
        "Do something.",
        [[("not_a_function", "{}")], lambda history: "Cannot do that."],
    )

    assert result.trace.records == []
    assert len(result.trace.rejected_attempts) == 1
    attempt = result.trace.rejected_attempts[0]
    assert attempt.name == f"{PLUGIN}-not_a_function"
    assert "not part of the provided tools" in attempt.reason
    assert result.trace.format().startswith("No functions were invoked.")


def test_failed_execution_is_not_a_rejected_attempt():
    """A function that ran and raised is `executed` with status `error`."""
    result, _ = _run(
        "Summarize.",
        [["dataset_summary"], lambda history: "Failed."],
        _failing_loader,
    )

    assert [(a.outcome, a.status) for a in result.trace.attempts] == [
        ("executed", "error")
    ]
    assert result.trace.rejected_attempts == []


def test_normal_run_has_no_rejected_section():
    result, _ = _run(
        "How many rows?", [["dataset_summary"], lambda history: "60 rows."]
    )

    assert [a.outcome for a in result.trace.attempts] == ["executed"]
    assert "Rejected" not in result.trace.format()
    records = result.trace.attempts_to_dicts()
    assert records[0]["call_id"] == result.trace.records[0].call_id


def test_run_result_includes_metric_check():
    def answer(history: ChatHistory) -> str:
        summary = json.loads(_tool_results(history)[-1])
        return f"The dataset has {summary['rows']} rows and 7 features."

    result, _ = _run(
        "How many rows and features?", [["dataset_summary"], answer]
    )

    checks = {c.metric: c.status for c in result.metric_check.checks}
    assert checks["rows"] == "verified"
    assert checks["features"] == "mismatch"
    assert not result.metric_check.ok


@pytest.mark.parametrize("prompt", ["", "   "])
def test_run_agent_rejects_empty_prompt(prompt):
    with pytest.raises(ValueError, match="must not be empty"):
        asyncio.run(
            housing_agent.run_agent(prompt, service=ScriptedChatService([]))
        )
