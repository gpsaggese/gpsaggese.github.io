"""
Import as:

import research.Noesis.gateway.test.test_passthrough_proxy as rnttpp
"""

import logging
import pprint
from typing import Callable, List, Tuple

import helpers.hunit_test as hunitest
import research.Noesis.gateway.passthrough_proxy as rnopapro

_LOG = logging.getLogger(__name__)


def _echo_call_func(model: str, prompt: str) -> str:
    """
    Stand-in provider `call_func` that echoes `model` and `prompt` back.

    :param model: model identifier passed to the provider
    :param prompt: prompt text passed to the provider
    :return: deterministic response text derived from `model`/`prompt`
    """
    return f"[{model}] {prompt}"


def _upper_call_func(model: str, prompt: str) -> str:
    """
    Stand-in provider `call_func` that upper-cases `prompt`, ignoring `model`.

    :param model: model identifier passed to the provider (unused)
    :param prompt: prompt text passed to the provider
    :return: `prompt` upper-cased
    """
    _ = model
    return prompt.upper()


# #############################################################################
# TestGateway
# #############################################################################


class TestGateway(hunitest.TestCase):
    """
    Test `passthrough_proxy.Gateway`.
    """

    def helper(
        self,
        provider_configs: List[rnopapro.ProviderConfig],
        calls: List[Tuple[str, str, str]],
        clock_func: Callable[[], float],
        expected: str,
    ) -> None:
        """
        Register `provider_configs`, issue `calls` through a fresh
        `Gateway`, and compare the pretty-printed log to `expected`.

        :param provider_configs: providers to register before issuing calls
        :param calls: `(provider_name, model, prompt)` tuples passed to
            `Gateway.call()`, in order
        :param clock_func: deterministic clock injected into the `Gateway`
        :param expected: expected pretty-printed `get_log()` result
        """
        # Prepare inputs.
        gateway = rnopapro.Gateway(clock_func=clock_func)
        for provider_config in provider_configs:
            gateway.register_provider(provider_config)
        # Run test.
        for provider_name, model, prompt in calls:
            gateway.call(provider_name, model, prompt)
        # Check outputs.
        actual = pprint.pformat(gateway.get_log())
        self.assert_equal(actual, expected, dedent=True)

    def test1(self) -> None:
        """
        Test that a request routed to each of 3 configured providers is
        logged with the correct schema and fields.
        """
        # Prepare inputs.
        provider_configs = [
            rnopapro.ProviderConfig("openai_mock", _echo_call_func, 0.01),
            rnopapro.ProviderConfig("anthropic_mock", _echo_call_func, 0.02),
            rnopapro.ProviderConfig("local_mock", _upper_call_func, 0.0),
        ]
        calls = [
            ("openai_mock", "gpt-4o-mini", "2+2="),
            ("anthropic_mock", "claude-3-haiku", "capital of France?"),
            ("local_mock", "llama-3-8b", "hello"),
        ]
        clock_func = iter([0.0, 1.0, 2.0, 2.5, 3.0, 3.25]).__next__
        # Prepare outputs.
        expected = """
        [RequestLogEntry(request_id=0,
                         provider='openai_mock',
                         model='gpt-4o-mini',
                         prompt='2+2=',
                         response='[gpt-4o-mini] 2+2=',
                         latency_in_secs=1.0,
                         cost=0.22),
         RequestLogEntry(request_id=1,
                         provider='anthropic_mock',
                         model='claude-3-haiku',
                         prompt='capital of France?',
                         response='[claude-3-haiku] capital of France?',
                         latency_in_secs=0.5,
                         cost=1.06),
         RequestLogEntry(request_id=2,
                         provider='local_mock',
                         model='llama-3-8b',
                         prompt='hello',
                         response='HELLO',
                         latency_in_secs=0.25,
                         cost=0.0)]
        """
        # Run test.
        self.helper(provider_configs, calls, clock_func, expected)

    def test2(self) -> None:
        """
        Test that `query_log()` filters logged entries by provider and by
        model independently.
        """
        # Prepare inputs.
        gateway = rnopapro.Gateway(
            clock_func=iter([0.0, 0.0, 0.0, 0.0]).__next__
        )
        gateway.register_provider(
            rnopapro.ProviderConfig("openai_mock", _echo_call_func, 0.0)
        )
        gateway.register_provider(
            rnopapro.ProviderConfig("anthropic_mock", _echo_call_func, 0.0)
        )
        gateway.call("openai_mock", "gpt-4o-mini", "2+2=")
        gateway.call("anthropic_mock", "claude-3-haiku", "capital of France?")
        # Prepare outputs.
        expected_by_provider = """
        [RequestLogEntry(request_id=0,
                         provider='openai_mock',
                         model='gpt-4o-mini',
                         prompt='2+2=',
                         response='[gpt-4o-mini] 2+2=',
                         latency_in_secs=0.0,
                         cost=0.0)]
        """
        expected_by_model = """
        [RequestLogEntry(request_id=1,
                         provider='anthropic_mock',
                         model='claude-3-haiku',
                         prompt='capital of France?',
                         response='[claude-3-haiku] capital of France?',
                         latency_in_secs=0.0,
                         cost=0.0)]
        """
        # Run test.
        actual_by_provider = pprint.pformat(
            gateway.query_log(provider="openai_mock")
        )
        actual_by_model = pprint.pformat(
            gateway.query_log(model="claude-3-haiku")
        )
        # Check outputs.
        self.assert_equal(actual_by_provider, expected_by_provider, dedent=True)
        self.assert_equal(actual_by_model, expected_by_model, dedent=True)

    def test3(self) -> None:
        """
        Test that a fresh `Gateway` starts with an empty, queryable log.
        """
        # Prepare inputs.
        gateway = rnopapro.Gateway()
        # Prepare outputs.
        expected: List[rnopapro.RequestLogEntry] = []
        # Run test.
        actual = gateway.get_log()
        # Check outputs.
        self.assert_equal(str(actual), str(expected))

    def test4(self) -> None:
        """
        Test that `call()` to an unregistered provider raises
        `AssertionError`.
        """
        # Prepare inputs.
        gateway = rnopapro.Gateway()
        gateway.register_provider(
            rnopapro.ProviderConfig("openai_mock", _echo_call_func, 0.0)
        )
        # Run test and check output.
        with self.assertRaises(AssertionError):
            gateway.call("unknown_provider", "gpt-4o-mini", "2+2=")

    def test5(self) -> None:
        """
        Test that registering the same provider name twice raises
        `AssertionError`.
        """
        # Prepare inputs.
        gateway = rnopapro.Gateway()
        gateway.register_provider(
            rnopapro.ProviderConfig("openai_mock", _echo_call_func, 0.0)
        )
        # Run test and check output.
        with self.assertRaises(AssertionError):
            gateway.register_provider(
                rnopapro.ProviderConfig("openai_mock", _upper_call_func, 0.01)
            )

    def test6(self) -> None:
        """
        Test that `query_log()` with both `provider` and `model` set
        returns only entries matching both, not the union of either alone.
        """
        # Prepare inputs.
        gateway = rnopapro.Gateway(clock_func=iter([0.0] * 6).__next__)
        gateway.register_provider(
            rnopapro.ProviderConfig("openai_mock", _echo_call_func, 0.0)
        )
        gateway.register_provider(
            rnopapro.ProviderConfig("anthropic_mock", _echo_call_func, 0.0)
        )
        gateway.call("openai_mock", "gpt-4o-mini", "a")
        gateway.call("openai_mock", "gpt-4o", "b")
        gateway.call("anthropic_mock", "gpt-4o-mini", "c")
        # Prepare outputs.
        expected = """
        [RequestLogEntry(request_id=0,
                         provider='openai_mock',
                         model='gpt-4o-mini',
                         prompt='a',
                         response='[gpt-4o-mini] a',
                         latency_in_secs=0.0,
                         cost=0.0)]
        """
        # Run test.
        actual = pprint.pformat(
            gateway.query_log(provider="openai_mock", model="gpt-4o-mini")
        )
        # Check outputs.
        self.assert_equal(actual, expected, dedent=True)

    def test7(self) -> None:
        """
        Test that mutating a `get_log()` snapshot does not affect the
        `Gateway`'s internal log.
        """
        # Prepare inputs.
        gateway = rnopapro.Gateway(clock_func=iter([0.0, 0.0]).__next__)
        gateway.register_provider(
            rnopapro.ProviderConfig("openai_mock", _echo_call_func, 0.0)
        )
        gateway.call("openai_mock", "gpt-4o-mini", "a")
        # Run test.
        snapshot = gateway.get_log()
        snapshot.clear()
        # Check outputs.
        self.assertEqual(len(gateway.get_log()), 1)
