"""
Calibrate the canary bank or run the scheduled-probe worker.

Calibration is a no-spend validation by default. Add `--execute` only after
reviewing the printed provider-call count.

Examples:

> python3 -m research.Noesis.prober.prober_cli calibrate \
    --healthy groq=meta-llama/llama-3.3-70b-instruct \
    --healthy together=meta-llama/llama-3.3-70b-instruct \
    --weak deepinfra=meta-llama/llama-3.1-8b-instruct
> python3 -m research.Noesis.prober.prober_cli probe --once
> python3 -m research.Noesis.prober.prober_cli probe
"""

import argparse
import asyncio
import logging
import os
import signal
import sys
from datetime import datetime, timezone
from typing import Mapping, Optional, Sequence

import research.Noesis.checker.checker_calibration_report as rnochcare
import research.Noesis.checker.checker_calibration_runtime as rnochcaru
import research.Noesis.checker.checker_canaries as rnochcan
import research.Noesis.checker.checker_models as rnochmod
import research.Noesis.checker.checker_store as rnochsto
import research.Noesis.common.noesis_db as rnonodb
import research.Noesis.common.noesis_settings as rnonoset
import research.Noesis.gateway.gateway_faults as rnogafau
import research.Noesis.gateway.gateway_openrouter as rnogaope
import research.Noesis.prober.prober_runtime as rnoprrun
import research.Noesis.prober.prober_worker as rnoprwor

_LOG = logging.getLogger(__name__)

DEFAULT_BANK_VERSION = "v1"
DEFAULT_MINIMUM_SURVIVORS = 150
DEFAULT_BANK_DIR = os.path.join(rnonoset.NOESIS_DIR, "data", "canaries")
DEFAULT_REPORT_PATH = os.path.join(
    rnonoset.NOESIS_DIR, "docs", "findings", "calibration.md"
)


def parse_target(
    value: str, role: rnochmod.CalibrationRole
) -> rnochmod.CalibrationTarget:
    """
    Parse `provider_slug=model_id` without restricting valid model names.
    """
    provider_slug, separator, model_id = value.partition("=")
    if not separator or not provider_slug.strip() or not model_id.strip():
        raise argparse.ArgumentTypeError(
            "target must have the form provider_slug=model_id"
        )
    return rnochmod.CalibrationTarget(
        role, provider_slug.strip(), model_id.strip()
    )


def _combined_environment() -> Mapping[str, str]:
    dotenv_path = os.path.join(rnonoset.NOESIS_DIR, ".env")
    return {**rnonoset.read_dotenv(dotenv_path), **os.environ}


def _require_secret(environment: Mapping[str, str], name: str) -> str:
    value = environment.get(name, "")
    if not value:
        raise ValueError("%s is required; see env.example" % name)
    return value


def _write_report(path: str, report: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file_:
        file_.write(report)


async def _run_calibrate(args: argparse.Namespace) -> int:
    canaries = rnochcan.load_canary_bank(args.bank_dir, args.bank_version)
    targets = tuple(
        parse_target(value, rnochmod.CalibrationRole.HEALTHY)
        for value in args.healthy
    ) + (parse_target(args.weak, rnochmod.CalibrationRole.WEAK),)
    policy = rnochmod.CalibrationPolicy(
        repetitions=args.repetitions,
        max_tokens=args.max_tokens,
        timeout_s=args.timeout,
        concurrency=args.concurrency,
    )
    rnochcaru.validate_calibration_inputs(canaries, targets)
    if not 1 <= args.minimum_survivors <= len(canaries):
        raise ValueError(
            "minimum_survivors must be between one and the candidate count"
        )
    call_count = len(canaries) * len(targets) * policy.repetitions
    print(
        "Calibration plan: %s candidates x %s targets x %s repetitions"
        " = %s paid provider calls"
        % (len(canaries), len(targets), policy.repetitions, call_count)
    )
    if not args.execute:
        print("Validation only: no database writes or provider calls were made.")
        print("Review the plan, then repeat with --execute to run it.")
        return 0
    environment = _combined_environment()
    api_key = _require_secret(environment, "OPENROUTER_API_KEY")
    settings = rnonoset.load_settings(environ=environment)
    provider = rnogaope.OpenRouterProvider(api_key)
    try:
        pool = await rnonodb.connect(settings.database_url)
        try:
            await rnonodb.apply_migrations(pool)
            run_id, execution = await rnochcaru.run_calibration(
                pool, provider, canaries, targets, policy
            )
            accepted = int(execution.summary["accepted_count"])
            activated = accepted >= args.minimum_survivors
            if activated:
                await rnochsto.activate_calibrated_bank(
                    pool,
                    run_id,
                    args.bank_version,
                    args.minimum_survivors,
                )
            report = rnochcare.render_calibration_report(
                run_id,
                args.bank_version,
                policy,
                execution.summary,
                args.minimum_survivors,
                activated,
            )
            _write_report(args.report, report)
            print(report)
            print("Wrote %r" % os.path.relpath(args.report))
            if not activated:
                print("Calibration completed, but the bank was not activated.")
                return 2
            return 0
        finally:
            await pool.close()
    finally:
        await provider.aclose()


async def _run_probe(args: argparse.Namespace) -> int:
    environment = _combined_environment()
    api_key = _require_secret(environment, "OPENROUTER_API_KEY")
    schedule_secret = _require_secret(
        environment, "NOESIS_CANARY_SCHEDULE_SECRET"
    )
    settings = rnonoset.load_settings(environ=environment)
    policy = rnochmod.ProbePolicy(
        bank_version=args.bank_version,
        model_id=settings.model_id,
        max_tokens=min(args.max_tokens, settings.max_tokens_cap),
        timeout_multiplier=settings.timeout_multiplier,
    )
    provider = rnogaope.OpenRouterProvider(api_key)
    faults = rnogafau.FaultRegistry()
    try:
        pool = await rnonodb.connect(settings.database_url)
        try:
            await rnonodb.apply_migrations(pool)
            if args.once:
                result = await rnoprrun.run_due_probes(
                    pool,
                    provider,
                    faults,
                    policy,
                    schedule_secret,
                    datetime.now(timezone.utc),
                )
                print(
                    "Probe cycle: executions=%s errors=%s"
                    % (len(result.executions), len(result.errors))
                )
                for error in result.errors:
                    print("  %s" % error)
                return 1 if result.errors else 0
            stop_event = asyncio.Event()
            loop = asyncio.get_running_loop()
            for signal_name in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(signal_name, stop_event.set)
            result = await rnoprwor.run_probe_worker(
                pool,
                provider,
                faults,
                policy,
                schedule_secret,
                stop_event,
                poll_interval_s=args.poll_interval,
            )
            print(
                "Probe worker stopped: cycles=%s executions=%s errors=%s"
                % (result.cycles, result.executions, result.errors)
            )
            return 0
        finally:
            await pool.close()
    finally:
        await provider.aclose()


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    calibrate = subparsers.add_parser(
        "calibrate", help="validate or execute live-provider calibration"
    )
    calibrate.add_argument("--bank-version", default=DEFAULT_BANK_VERSION)
    calibrate.add_argument("--bank-dir", default=DEFAULT_BANK_DIR)
    calibrate.add_argument(
        "--healthy",
        action="append",
        required=True,
        metavar="PROVIDER=MODEL",
        help="healthy target; repeat for each provider",
    )
    calibrate.add_argument("--weak", required=True, metavar="PROVIDER=MODEL")
    calibrate.add_argument("--repetitions", type=int, default=3)
    calibrate.add_argument("--max-tokens", type=int, default=64)
    calibrate.add_argument("--timeout", type=float, default=15.0)
    calibrate.add_argument("--concurrency", type=int, default=5)
    calibrate.add_argument(
        "--minimum-survivors", type=int, default=DEFAULT_MINIMUM_SURVIVORS
    )
    calibrate.add_argument("--report", default=DEFAULT_REPORT_PATH)
    calibrate.add_argument(
        "--execute",
        action="store_true",
        help="make paid provider calls and persist the run",
    )
    probe = subparsers.add_parser(
        "probe", help="run due probes once or until shutdown"
    )
    probe.add_argument("--bank-version", default=DEFAULT_BANK_VERSION)
    probe.add_argument("--max-tokens", type=int, default=64)
    probe.add_argument("--poll-interval", type=float, default=1.0)
    probe.add_argument(
        "--once", action="store_true", help="run one cycle and exit"
    )
    return parser.parse_args(argv)


async def _main(args: argparse.Namespace) -> int:
    if args.command == "calibrate":
        return await _run_calibrate(args)
    if args.command == "probe":
        return await _run_probe(args)
    raise ValueError("unknown command %r" % args.command)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(asyncio.run(_main(_parse_args())))
