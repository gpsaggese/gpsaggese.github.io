"""Resolve references in a record to code at the pinned SHA (specs/03-offline-pipeline.md)."""
import collections
import os
import re
from functools import cached_property

from .gitstore import GitStore, is_test

CODE_EXT = re.compile(r"\.(java|kts|sql)$")
NAMED_FILE = re.compile(r"\.(java|kts|sql|toml|ya?ml|properties)$")
NON_CODE = re.compile(r"\.(mdx?|rst|txt|adoc|png|jpe?g|svg|drawio|puml)$|(^|/)(docs?|documentation)/", re.I)
# Stay linked for retrieval, but excluded from churn: commits touching them are nearly always unrelated.
SHARED_CONFIG = re.compile(r"^(gradle/libs\.versions\.toml|settings\.gradle\.kts|build\.gradle\.kts|gradle\.properties)$")
_CAMEL = re.compile(r"\b[A-Z][a-z]+(?:[A-Z][A-Za-z0-9]*)+\b")


def extract_tokens(text: str) -> dict[str, set[str]]:
    ticks = set(re.findall(r"`([^`\n]+)`", text))
    camel = {t for t in ticks if re.fullmatch(r"[A-Z][A-Za-z0-9]+", t)} | set(_CAMEL.findall(text))
    kebab = {t for t in ticks if re.fullmatch(r"[a-z][a-z0-9]*(-[a-z0-9]+)+", t)}
    slashed = {t.strip("./") for t in ticks if "/" in t}
    named = {t for t in ticks if "/" not in t and NAMED_FILE.search(t)}
    return {"camel": camel, "kebab": kebab, "slashed": slashed, "named": named}


class Linker:
    def __init__(self, store: GitStore):
        self.store = store

    @cached_property
    def code(self) -> list[str]:
        return [n for n in self.store.tree if not n.startswith("docs/") and CODE_EXT.search(n)]

    @cached_property
    def tree_set(self) -> set[str]:
        return set(self.store.tree)

    @cached_property
    def classes(self) -> dict[str, list[str]]:
        found = collections.defaultdict(list)
        for n in self.code:
            if n.endswith(".java"):
                found[os.path.basename(n)[:-5]].append(n)
        return found

    @cached_property
    def dirs_by_name(self) -> dict[str, set[str]]:
        found = collections.defaultdict(set)
        for n in self.store.tree:
            if n.startswith("docs/"):
                continue
            parts = n.split("/")
            for i in range(1, len(parts)):
                found[parts[i - 1]].add("/".join(parts[:i]))
        return found

    @cached_property
    def all_dirs(self) -> set[str]:
        return {d for found in self.dirs_by_name.values() for d in found}

    @cached_property
    def basenames(self) -> dict[str, list[str]]:
        found = collections.defaultdict(list)
        for n in self.store.tree:
            if not n.startswith("docs/"):
                found[os.path.basename(n)].append(n)
        return found

    def _expand_dir(self, directory: str) -> list[str]:
        return [f for f in self.code if not is_test(f) and f.startswith(directory + "/")]

    def link_record(self, record_id: str, text: str) -> list[dict]:
        tokens = extract_tokens(text)
        links: dict[tuple[str, str], dict] = {}

        def add(file_id: str, signal: str, evidence: str) -> None:
            links.setdefault((file_id, signal), {
                "record_id": record_id, "file_id": file_id, "signal": signal, "evidence": evidence,
                "shared_config": bool(SHARED_CONFIG.match(file_id)),
            })

        for token in sorted(tokens["camel"]):
            if token in self.classes:
                main = [f for f in self.classes[token] if not is_test(f)]
                for f in main or self.classes[token]:
                    add(f, "explicit_class", token)
        for token in sorted(tokens["kebab"]):
            for d in self.dirs_by_name.get(token, ()):
                for f in self._expand_dir(d):
                    add(f, "explicit_module", token)
        for token in sorted(tokens["slashed"]):
            if token.startswith("docs/"):
                continue
            if token in self.tree_set:
                add(token, "explicit_path", token)
            elif token in self.all_dirs:
                for f in self._expand_dir(token):
                    add(f, "explicit_path", token)
        for token in sorted(tokens["named"]):
            if len(self.basenames.get(token, [])) == 1:
                add(self.basenames[token][0], "explicit_file", token)
        return list(links.values())

    def adr_commit_links(self, record_id: str, record_path: str) -> tuple[list[dict], dict]:
        """The diagram's 'files in ADR commit' signal, plus what it found for reporting."""
        first = self.store.history(record_path)[0][0]
        files = self.store.files_in_commit(first)
        code = [f for f in files if not NON_CODE.search(f)]
        live = [f for f in code if f in self.tree_set]
        links = [{"record_id": record_id, "file_id": f, "signal": "adr_commit", "evidence": first,
                  "shared_config": bool(SHARED_CONFIG.match(f))} for f in live]
        return links, {"record_id": record_id, "commit": first, "files": len(files),
                       "code_files": len(code), "code_files_still_present": len(live)}
