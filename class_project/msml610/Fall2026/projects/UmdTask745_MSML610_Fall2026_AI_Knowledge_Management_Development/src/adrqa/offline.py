"""Run lane 1 of the workflow, except the embedding index, and write JSON under data/."""
import json

from .config import DATA_DIR, RECORD_ROOT, RECORDS
from .gaps import rank_gaps
from .gitstore import GitStore
from .graph import build_graph, save_graph
from .linker import Linker
from .parser import parse_record
from .relations import apply_model_relations
from .staleness import dead_identifiers, score_record


def _write(name: str, value) -> None:
    (DATA_DIR / name).write_text(json.dumps(value, indent=1, ensure_ascii=False), encoding="utf-8")


def build_offline(log=print, chat=None) -> dict:
    DATA_DIR.mkdir(exist_ok=True)
    GitStore.ensure_clone()
    store = GitStore()
    linker = Linker(store)
    texts = store.read_many([f"{RECORD_ROOT}{r}/README.md" for r in RECORDS])
    text_of = {r: texts[f"{RECORD_ROOT}{r}/README.md"] for r in RECORDS}
    corpus = set(RECORDS)

    log("parsing records")
    records = [parse_record(r, text_of[r], corpus) for r in RECORDS]
    if chat is not None:
        log("finding replacement relations with the model")
        apply_model_relations(records, text_of, chat, log)
    else:
        log("no chat model configured: using rule-based replacement detection only")
    _write("records.json", records)

    log("linking records to code")
    links, commit_report = [], []
    for rec in records:
        links += linker.link_record(rec["id"], text_of[rec["id"]])
        commit_links, report = linker.adr_commit_links(rec["id"], rec["path"])
        links += commit_links
        commit_report.append(report)
    _write("links.json", links)
    _write("adr_commit_signal.json", commit_report)

    graph = build_graph(records, links)
    save_graph(graph, DATA_DIR / "graph.json")

    log("scoring staleness (follows renames per file, this takes a few minutes)")
    scores = []
    for rec in records:
        files = sorted({l["file_id"] for l in links if l["record_id"] == rec["id"]})
        dead = dead_identifiers(store, text_of[rec["id"]], linker.classes, linker.dirs_by_name)
        scores.append(score_record(store, rec, files, dead))
    _write("staleness.json", scores)

    log("ranking documentation gaps")
    governed = {l["file_id"] for l in links}
    gaps = rank_gaps(store, governed)
    _write("gap_report.json", gaps)
    return {"records": records, "links": links, "scores": scores, "gaps": gaps, "commit_report": commit_report}
