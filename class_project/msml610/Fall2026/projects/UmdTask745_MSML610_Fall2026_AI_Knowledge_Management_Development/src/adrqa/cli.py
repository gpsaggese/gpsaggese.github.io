"""Command line for the offline half: doctor and build. The API key is never printed."""
import argparse
import sys

from .config import CLONE_DIR, DATA_DIR, SHA, get_settings


def cmd_doctor(_) -> int:
    """Checks what is needed to run the offline build. The model settings are optional, so they never fail the check."""
    settings = get_settings()
    ok = True

    def report(name: str, passed: bool, detail: str = "", required: bool = True) -> None:
        nonlocal ok
        ok = ok and (passed or not required)
        mark = "ok" if passed else ("MISSING" if required else "not set (optional)")
        print(f"[{mark}] {name}" + (f": {detail}" if detail else ""))

    report("clone at pinned commit", (CLONE_DIR / ".git").is_dir(), f"{SHA[:9]}; `build` creates it if missing", required=False)
    report("offline data", (DATA_DIR / "staleness.json").exists(), "run `build` if missing", required=False)
    report("OPENAI_API_KEY", bool(settings.openai_api_key), "enables the model relation extractor", required=False)
    report("OPENAI_CHAT_MODEL", bool(settings.chat_model), settings.chat_model or "needed together with the key", required=False)
    if not (settings.openai_api_key and settings.chat_model):
        print("Without both, `build` finds replacements with the rule-based parser only. Everything else is unchanged.")
    return 0 if ok else 1


def cmd_build(_) -> int:
    from .offline import build_offline

    settings = get_settings()
    chat = None
    if settings.openai_api_key and settings.chat_model:
        from .llm import ChatClient

        chat = ChatClient(settings)
    result = build_offline(chat=chat)
    print(f"records: {len(result['records'])}, links: {len(result['links'])}, "
          f"staleness: {sum(s['stale'] is not None for s in result['scores'])} scored, "
          f"{sum(s['stale'] is None for s in result['scores'])} null, gap modules: {len(result['gaps'])}")
    print(f"wrote JSON to {DATA_DIR}")
    return 0


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="adrqa")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="check what the offline build needs").set_defaults(fn=cmd_doctor)
    sub.add_parser("build", help="read the records, link them to code, build the graph, score staleness, rank gaps").set_defaults(fn=cmd_build)
    args = parser.parse_args()
    sys.exit(args.fn(args))
