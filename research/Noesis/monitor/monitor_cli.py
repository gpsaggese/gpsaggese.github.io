"""
Validate or execute one standalone model-monitoring run.

Validation is the default and makes no database writes or provider calls.

Example:

> python3 -m research.Noesis.monitor.monitor_cli \
    --target groq=meta-llama/llama-3.3-70b-instruct
"""

import argparse
import asyncio
import logging
import os
import sys
from typing import Mapping, Optional, Sequence

import research.Noesis.checker.checker_canaries as rnochcan
import research.Noesis.common.noesis_db as rnonodb
import research.Noesis.common.noesis_settings as rnonoset
import research.Noesis.gateway.gateway_openrouter as rnogaope
import research.Noesis.monitor.monitor_models as rnmonmod
import research.Noesis.monitor.monitor_runtime as rnmonrun

DEFAULT_BANK_VERSION = "v1"
DEFAULT_BANK_DIR = os.path.join(rnonoset.NOESIS_DIR, "data", "canaries")


def parse_target(value: str) -> rnmonmod.MonitoringTarget:
    """
    Parse `provider_slug=model_id` without restricting valid model names.
    """
    provider_slug, separator, model_id = value.partition("=")
    if not separator or not provider_slug.strip() or not model_id.strip():
        raise argparse.ArgumentTypeError(
            "target must have the form provider_slug=model_id"
        )
    return rnmonmod.MonitoringTarget(provider_slug.strip(), model_id.strip())


def _combined_environment() -> Mapping[str, str]:
    dotenv_path = os.path.join(rnonoset.NOESIS_DIR, ".env")
    return {**rnonoset.read_dotenv(dotenv_path), **os.environ}


def _require_secret(environment: Mapping[str, str], name: str) -> str:
    value = environment.get(name, "")
    if not value:
        raise ValueError("%s is required; see env.example" % name)
    return value


async def _run(args: argparse.Namespace) -> int:
    environment = _combined_environment()
    settings = rnonoset.load_settings(environ=environment)
    policy = rnmonmod.MonitoringPolicy(
        bank_version=args.bank_version,
        canary_count=args.canary_count,
        max_tokens=min(args.max_tokens, settings.max_tokens_cap),
        timeout_s=args.timeout,
    )
    canaries = rnochcan.load_canary_bank(args.bank_dir, policy.bank_version)
    rnmonrun.validate_monitoring_inputs(canaries, args.target, policy)
    print(
        "Monitoring plan: %s canaries x 1 target = %s paid provider calls"
        % (policy.canary_count, policy.canary_count)
    )
    print("Target: %s=%s" % (args.target.provider_slug, args.target.model_id))
    if not args.execute:
        print("Validation only: no database writes or provider calls were made.")
        print("Review the plan, then repeat with --execute to run it.")
        return 0
    api_key = _require_secret(environment, "OPENROUTER_API_KEY")
    selection_secret = _require_secret(
        environment, "NOESIS_CANARY_SCHEDULE_SECRET"
    )
    provider = rnogaope.OpenRouterProvider(api_key)
    try:
        pool = await rnonodb.connect(settings.database_url)
        try:
            await rnonodb.apply_migrations(pool)
            run_id, execution = await rnmonrun.run_monitoring(
                pool,
                provider,
                args.target,
                policy,
                selection_secret,
            )
            print(
                "Monitoring run %s completed: calls=%s correct=%s"
                " ungradable=%s cost_usd=%s"
                % (
                    run_id,
                    execution.summary["total_calls"],
                    execution.summary["correct_calls"],
                    execution.summary["ungradable_calls"],
                    execution.summary["total_cost_usd"],
                )
            )
            return 0
        finally:
            await pool.close()
    finally:
        await provider.aclose()


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        type=parse_target,
        required=True,
        metavar="PROVIDER=MODEL",
    )
    parser.add_argument("--bank-version", default=DEFAULT_BANK_VERSION)
    parser.add_argument("--bank-dir", default=DEFAULT_BANK_DIR)
    parser.add_argument("--canary-count", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="make paid provider calls and persist the run",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(asyncio.run(_run(_parse_args())))
