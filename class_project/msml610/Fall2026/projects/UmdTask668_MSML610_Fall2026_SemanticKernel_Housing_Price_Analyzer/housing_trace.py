"""
Execution trace of the kernel functions invoked by the agent.

Two kernel filters record each function call that the model chooses and
Semantic Kernel invokes:

- an auto function invocation filter, which records the call and the model
  request it belongs to
- a function invocation filter, which records exceptions raised by the
  function (Semantic Kernel turns them into a message for the model, so the
  first filter does not see them)

Semantic Kernel rejects some calls before any filter runs, e.g., an unknown
function name or arguments that do not match the signature. These calls
never run. After a run, `record_attempts()` finds them in the conversation
history.

Import as:

import housing_trace
"""

import contextvars
import dataclasses
import json
import logging
import time
from collections.abc import Awaitable, Callable, Iterable
from typing import Any

from semantic_kernel import Kernel
from semantic_kernel.contents import (
    AuthorRole,
    ChatMessageContent,
    FunctionCallContent,
    FunctionResultContent,
)
from semantic_kernel.filters import (
    AutoFunctionInvocationContext,
    FilterTypes,
    FunctionInvocationContext,
)

_LOG = logging.getLogger(__name__)

# Length of `result_preview`, shown by `format()`. The full result is always
# kept in `result`.
RESULT_PREVIEW_CHARS = 200

# Record of the auto-invoked function currently running in this task.
_CURRENT_RECORD: contextvars.ContextVar["FunctionCallRecord | None"] = (
    contextvars.ContextVar("housing_trace_current_record", default=None)
)


@dataclasses.dataclass
class FunctionCallRecord:
    """
    One function call chosen by the model and invoked by the kernel.

    :param step: 1-based position of the call in the trace
    :param request_index: 0-based index of the model request that asked
        for the call (calls requested together share the same index)
    :param plugin: plugin name
    :param function: function name
    :param arguments: arguments the function was invoked with
    :param status: `ok` or `error`
    :param duration_ms: wall-clock duration of the call
    :param result: full result returned to the model
    :param result_preview: beginning of the result, for display
    :param error: error message if the function raised an exception
    :param call_id: identifier of the model's function call request
    """

    step: int
    request_index: int
    plugin: str
    function: str
    arguments: dict[str, Any]
    status: str = "running"
    duration_ms: float = 0.0
    result: str = ""
    result_preview: str = ""
    error: str | None = None
    call_id: str | None = None

    @property
    def qualified_name(self) -> str:
        return f"{self.plugin}.{self.function}"


@dataclasses.dataclass
class ToolCallAttempt:
    """
    One function call requested by the model, whether or not it ran.

    :param request_index: 0-based index of the model request that asked
        for the call
    :param call_id: identifier of the function call request
    :param name: function name exactly as requested by the model
    :param arguments: arguments exactly as requested by the model
    :param outcome: `executed` if the kernel invoked the function, or
        `rejected` if Semantic Kernel refused the call before invoking it
    :param status: for executed calls, the execution status (`ok` or
        `error`); None for rejected calls
    :param reason: for rejected calls, the message Semantic Kernel returned
        to the model
    """

    request_index: int
    call_id: str | None
    name: str
    arguments: Any
    outcome: str
    status: str | None = None
    reason: str | None = None


class ExecutionTrace:
    """
    Collect the function calls made during one agent run.

    `records` lists the executed calls (successful or failed); `attempts`
    lists every call the model requested, including rejected ones, once
    `record_attempts()` has been called.
    """

    def __init__(self) -> None:
        self.records: list[FunctionCallRecord] = []
        self.attempts: list[ToolCallAttempt] = []

    @property
    def rejected_attempts(self) -> list[ToolCallAttempt]:
        """Return the requested calls that Semantic Kernel did not execute."""
        return [a for a in self.attempts if a.outcome == "rejected"]

    def record_attempts(self, messages: Iterable[ChatMessageContent]) -> None:
        """
        List the function calls the model requested in a conversation.

        Each assistant message with function calls is one model request. A
        call is `executed` if this trace recorded an execution with the same
        call identifier, and `rejected` otherwise, with the tool message that
        Semantic Kernel returned as the reason. Only calls in the
        conversation are listed.

        :param messages: conversation messages of the agent run, in order
        """
        messages = list(messages)
        executions = {
            record.call_id: record
            for record in self.records
            if record.call_id is not None
        }
        tool_results = {
            item.id: str(item.result)
            for message in messages
            if message.role == AuthorRole.TOOL
            for item in message.items
            if isinstance(item, FunctionResultContent)
        }
        self.attempts = []
        request_index = 0
        for message in messages:
            calls = [
                item
                for item in message.items
                if isinstance(item, FunctionCallContent)
            ]
            if message.role != AuthorRole.ASSISTANT or not calls:
                continue
            for call in calls:
                record = executions.get(call.id)
                if record is not None:
                    attempt = ToolCallAttempt(
                        request_index=request_index,
                        call_id=call.id,
                        name=call.name or "",
                        arguments=call.arguments,
                        outcome="executed",
                        status=record.status,
                    )
                else:
                    attempt = ToolCallAttempt(
                        request_index=request_index,
                        call_id=call.id,
                        name=call.name or "",
                        arguments=call.arguments,
                        outcome="rejected",
                        reason=tool_results.get(call.id),
                    )
                self.attempts.append(attempt)
            request_index += 1

    def attempts_to_dicts(self) -> list[dict[str, Any]]:
        """Return the requested calls as JSON-serializable dictionaries."""
        return [dataclasses.asdict(attempt) for attempt in self.attempts]

    def attach(self, kernel: Kernel) -> None:
        """
        Register the trace filters on a kernel.

        :param kernel: kernel used by the agent
        """
        kernel.add_filter(
            FilterTypes.AUTO_FUNCTION_INVOCATION, self._auto_function_filter
        )
        kernel.add_filter(
            FilterTypes.FUNCTION_INVOCATION, self._function_filter
        )

    @property
    def function_names(self) -> list[str]:
        """Return the qualified names of the invoked functions, in order."""
        return [record.qualified_name for record in self.records]

    def to_dicts(self) -> list[dict[str, Any]]:
        """Return the trace as a list of JSON-serializable dictionaries."""
        return [dataclasses.asdict(record) for record in self.records]

    def to_json(self, **kwargs: Any) -> str:
        """
        Return the trace as a JSON string.

        :param kwargs: options passed to `json.dumps()` (e.g., `indent`)
        """
        return json.dumps(self.to_dicts(), default=str, **kwargs)

    def format(self) -> str:
        """
        Return a human-readable description of the trace.

        Executed calls are listed first, in the same format as before;
        rejected calls, if any, follow in a separate section.
        """
        lines = []
        if not self.records:
            lines.append("No functions were invoked.")
        for record in self.records:
            arguments = json.dumps(record.arguments, default=str)
            lines.append(
                f"{record.step}. {record.qualified_name}({arguments})"
                f" [request {record.request_index}]"
                f" -> {record.status} in {record.duration_ms:.0f} ms"
            )
            if record.error is not None:
                lines.append(f"   error: {record.error}")
            else:
                lines.append(f"   result: {record.result_preview}")
        rejected = self.rejected_attempts
        if rejected:
            lines.append("Rejected function calls (not executed):")
            for attempt in rejected:
                lines.append(
                    f"- {attempt.name}({attempt.arguments})"
                    f" [request {attempt.request_index}]"
                )
                lines.append(f"   reason: {attempt.reason}")
        return "\n".join(lines)

    async def _auto_function_filter(
        self,
        context: AutoFunctionInvocationContext,
        next: Callable[[AutoFunctionInvocationContext], Awaitable[None]],
    ) -> None:
        # Semantic Kernel passes the continuation as a keyword argument
        # named `next`, so the parameter name must not change.
        function = context.function
        arguments = {
            parameter.name: context.arguments[parameter.name]
            for parameter in function.parameters
            if context.arguments is not None
            and parameter.name in context.arguments
        }
        record = FunctionCallRecord(
            step=len(self.records) + 1,
            request_index=context.request_sequence_index,
            plugin=function.plugin_name or "",
            function=function.name,
            arguments=arguments,
            call_id=context.function_call_content.id,
        )
        self.records.append(record)
        _LOG.info("Agent invoking %s", record.qualified_name)
        token = _CURRENT_RECORD.set(record)
        start = time.perf_counter()
        try:
            await next(context)
        finally:
            _CURRENT_RECORD.reset(token)
            record.duration_ms = (time.perf_counter() - start) * 1000
        record.status = "error" if record.error is not None else "ok"
        if context.function_result is not None:
            record.result = str(context.function_result.value)
            record.result_preview = record.result[:RESULT_PREVIEW_CHARS]

    async def _function_filter(
        self,
        context: FunctionInvocationContext,
        next: Callable[[FunctionInvocationContext], Awaitable[None]],
    ) -> None:
        try:
            await next(context)
        except Exception as exc:
            record = _CURRENT_RECORD.get()
            if record is not None:
                record.error = f"{type(exc).__name__}: {exc}"
            raise
