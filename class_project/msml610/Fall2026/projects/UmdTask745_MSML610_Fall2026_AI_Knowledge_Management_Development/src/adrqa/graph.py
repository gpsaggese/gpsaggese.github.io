"""The decision graph: records and files, with supersedes, references and governs edges."""
import json
from pathlib import Path

import networkx as nx


def build_graph(records: list[dict], links: list[dict]) -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    for r in records:
        g.add_node(r["id"], kind="record")
    for r in records:
        for target in r["references"]:
            g.add_edge(r["id"], target, kind="references")
        for s in r["supersedes"]:
            g.add_edge(r["id"], s["target"], kind="supersedes", scope=s["scope"], statement=s["statement"])
    for link in links:
        g.add_node(link["file_id"], kind="file")
        if not g.has_edge(link["record_id"], link["file_id"]):
            g.add_edge(link["record_id"], link["file_id"], kind="governs", signal=link["signal"])
    return g


def save_graph(g: nx.MultiDiGraph, path: Path) -> None:
    path.write_text(json.dumps(nx.node_link_data(g, edges="edges"), indent=1), encoding="utf-8")


def load_graph(path: Path) -> nx.MultiDiGraph:
    return nx.node_link_graph(json.loads(path.read_text(encoding="utf-8")), directed=True, multigraph=True, edges="edges")


def _out(g: nx.MultiDiGraph, node: str, kind: str):
    return [(v, d) for _, v, d in g.out_edges(node, data=True) if d["kind"] == kind]


def _in(g: nx.MultiDiGraph, node: str, kind: str):
    return [(u, d) for u, _, d in g.in_edges(node, data=True) if d["kind"] == kind]


def linked_files(g: nx.MultiDiGraph, record_id: str) -> list[str]:
    return sorted(v for v, _ in _out(g, record_id, "governs"))


def references(g: nx.MultiDiGraph, record_id: str) -> list[str]:
    return sorted(v for v, _ in _out(g, record_id, "references"))


def superseded_by(g: nx.MultiDiGraph, record_id: str) -> list[tuple[str, dict]]:
    """Newer records that replace all or part of this one."""
    return _in(g, record_id, "supersedes")


def supersedes(g: nx.MultiDiGraph, record_id: str) -> list[tuple[str, dict]]:
    return _out(g, record_id, "supersedes")


def governing_records(g: nx.MultiDiGraph, file_or_dir: str) -> list[str]:
    """Records governing this file, or any file under this directory, oldest first."""
    prefix = file_or_dir.rstrip("/") + "/"
    found = set()
    for node, data in g.nodes(data=True):
        if data["kind"] == "file" and (node == file_or_dir or node.startswith(prefix)):
            found.update(u for u, _ in _in(g, node, "governs"))
    return sorted(found)
