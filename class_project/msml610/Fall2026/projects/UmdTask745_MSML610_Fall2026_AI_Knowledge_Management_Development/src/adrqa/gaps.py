"""Rank modules by complexity x (1 - coverage), the complex code no decision covers."""
import collections

import lizard

from .gitstore import GitStore, is_test


def rank_gaps(store: GitStore, governed: set[str], top: int = 20) -> list[dict]:
    java = [f for f in store.tree if f.endswith(".java") and "/src/main/" in f and not is_test(f)]
    texts = store.read_many(java)
    complexity: collections.Counter = collections.Counter()
    files: collections.Counter = collections.Counter()
    covered: collections.Counter = collections.Counter()
    for path, text in texts.items():
        module = path.split("/src/main/")[0]
        files[module] += 1
        covered[module] += path in governed
        info = lizard.analyze_file.analyze_source_code(path, text)
        complexity[module] += sum(f.cyclomatic_complexity for f in info.function_list)
    entries = []
    for module in files:
        coverage = covered[module] / files[module]
        entries.append({"module": module, "complexity": complexity[module], "coverage": round(coverage, 3),
                        "gap": round(complexity[module] * (1 - coverage), 2)})
    return sorted(entries, key=lambda e: (-e["gap"], e["module"]))[:top]
