"""
Natural-language housing analysis agent built with Semantic Kernel.

The agent is a `ChatCompletionAgent` that sees the functions of the
`HousingAnalysis` plugin as tools. With `FunctionChoiceBehavior.Auto()`, the
language model decides which functions to call for each request, and
Semantic Kernel invokes them and returns their results to the model. Every
invoked function is recorded in an `ExecutionTrace`.

Usage:

> python housing_agent.py "How many rows does the dataset have?"

Import as:

import housing_agent
"""

import argparse
import asyncio
import dataclasses
import logging
import sys

from semantic_kernel import Kernel
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.chat_completion_client_base import (
    ChatCompletionClientBase,
)
from semantic_kernel.connectors.ai.function_choice_behavior import (
    FunctionChoiceBehavior,
)

import housing_kernel
import housing_llm
import housing_trace
import housing_verify
from housing_plugins.housing_plugin import HousingAnalysisPlugin

_LOG = logging.getLogger(__name__)

AGENT_NAME = "HousingAnalyst"
# Model requests per run that may call functions. After the limit, Semantic
# Kernel makes one final request without tools. Same as the Semantic Kernel
# 1.46 default, set explicitly so that an upgrade cannot change it silently.
MAX_AUTO_INVOKE_ATTEMPTS = 5
AGENT_INSTRUCTIONS = """\
You are a housing price analysis assistant for the UCI Real Estate Valuation \
dataset.
- Answer by calling the HousingAnalysis functions. Call only the functions \
that the question needs, and call each function at most once: its result \
does not change.
- Report numbers exactly as the functions return them; you may round to \
4 decimal places. Never estimate or invent values.
- If a function returns an error, say that the analysis could not be \
completed and give the error.
- If the available functions cannot answer the question, say so instead of \
guessing.
- Keep the answer short."""


@dataclasses.dataclass
class AgentRunResult:
    """
    Result of one agent run.

    :param prompt: user request
    :param answer: final answer of the agent
    :param trace: functions invoked while answering
    :param metric_check: comparison of the labeled metrics in the answer
        with the function results
    """

    prompt: str
    answer: str
    trace: housing_trace.ExecutionTrace
    metric_check: "housing_verify.MetricCheckReport"


def create_agent(
    service: ChatCompletionClientBase,
    kernel: Kernel,
    *,
    max_auto_invoke_attempts: int = MAX_AUTO_INVOKE_ATTEMPTS,
) -> ChatCompletionAgent:
    """
    Create the housing analysis agent.

    :param service: chat-completion service that runs the language model
    :param kernel: kernel with the housing plugin registered
    :param max_auto_invoke_attempts: maximum number of model requests that
        may ask for function calls in one run
    :return: agent that automatically invokes the plugin functions chosen by
        the model
    """
    return ChatCompletionAgent(
        service=service,
        kernel=kernel,
        name=AGENT_NAME,
        instructions=AGENT_INSTRUCTIONS,
        function_choice_behavior=FunctionChoiceBehavior.Auto(
            maximum_auto_invoke_attempts=max_auto_invoke_attempts,
            filters={"included_plugins": [housing_kernel.PLUGIN_NAME]},
        ),
    )


async def run_agent(
    prompt: str,
    *,
    service: ChatCompletionClientBase | None = None,
    kernel: Kernel | None = None,
    max_auto_invoke_attempts: int = MAX_AUTO_INVOKE_ATTEMPTS,
) -> AgentRunResult:
    """
    Answer one natural-language request and record the invoked functions.

    Each call starts a new conversation.

    :param prompt: user request
    :param service: chat-completion service; defaults to the service
        configured by `housing_llm.load_llm_config()`
    :param kernel: kernel with the housing plugin; defaults to
        `housing_kernel.create_kernel()`
    :param max_auto_invoke_attempts: maximum number of model requests that
        may ask for function calls
    :return: answer and execution trace
    :raises ValueError: if the prompt is empty
    """
    if not prompt or not prompt.strip():
        raise ValueError("The prompt must not be empty.")
    if service is None:
        service = housing_llm.create_chat_service(
            housing_llm.load_llm_config()
        )
    if kernel is None:
        kernel = housing_kernel.create_kernel()
    trace = housing_trace.ExecutionTrace()
    trace.attach(kernel)
    agent = create_agent(
        service, kernel, max_auto_invoke_attempts=max_auto_invoke_attempts
    )
    response = await agent.get_response(prompt)
    # The thread holds the full conversation, including function calls that
    # Semantic Kernel rejected before invoking any filter.
    messages = [message async for message in response.thread.get_messages()]
    trace.record_attempts(messages)
    answer = str(response.message.content or "").strip()
    return AgentRunResult(
        prompt=prompt,
        answer=answer,
        trace=trace,
        metric_check=housing_verify.check_reported_metrics(answer, trace),
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("prompts", nargs="+", help="natural-language requests")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="show debug logging"
    )
    return parser.parse_args(argv)


async def _main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    config = housing_llm.load_llm_config()
    print(f"LLM: provider={config.provider} model={config.model}")
    service = housing_llm.create_chat_service(config)
    # One plugin for the whole session, so that the dataset is downloaded
    # and the models are trained at most once across all prompts.
    plugin = HousingAnalysisPlugin()
    exit_code = 0
    for prompt in args.prompts:
        print(f"\n=== Prompt: {prompt}")
        try:
            result = await run_agent(
                prompt,
                service=service,
                kernel=housing_kernel.create_kernel(plugin),
            )
        except Exception as exc:  # Report LLM or connection failures.
            print(f"Agent run failed: {type(exc).__name__}: {exc}")
            exit_code = 1
            continue
        print(f"Answer:\n{result.answer}")
        print(f"Execution trace:\n{result.trace.format()}")
        print(f"Metric check:\n{result.metric_check.format()}")
    return exit_code


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
