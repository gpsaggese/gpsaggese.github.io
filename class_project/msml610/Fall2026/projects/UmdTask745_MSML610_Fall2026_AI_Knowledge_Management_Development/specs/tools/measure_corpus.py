"""Regenerate the numbers quoted in 01-corpus.md from the pinned clone.

Usage: python measure_corpus.py <clone_dir> <output.json>

Reads git objects only. The clone has no working tree, because about 100 EDC paths exceed the
Windows path limit and a checkout would silently miss them.
"""
import collections
import json
import os
import re
import subprocess
import sys

SHA = "a66db8963acfecec5a7f3316115f42eaf54c0374"
RECORD_ROOT = "docs/developer/decision-records/"
RECORDS = [
    "2025-05-28-dataspace-profile-context",
    "2025-07-25-multiple-participant-identifiers",
    "2025-08-26-participant-context",
    "2025-10-15-participant-context-config",
    "2025-10-29-participant-identifiers",
    "2026-05-11-multi-profile-virtual-connector",
    "2026-09-22-per-participant-resource-ids",
    "2026-09-24-remove-atomikos",
    "2025-02-07-http-proxy-data-plane-deprecation",
    "2024-10-10-daps-deprecation",
]
CODE_EXT = re.compile(r"\.(java|kts|sql)$")
NAMED_FILE = re.compile(r"\.(java|kts|sql|toml|ya?ml|properties)$")
NON_CODE = re.compile(r"\.(mdx?|rst|txt|adoc|png|jpe?g|svg|drawio|puml)$|(^|/)(docs?|documentation)/", re.I)
# Linked for retrieval, but excluded from churn: nearly every commit touching these is unrelated.
SHARED_CONFIG = re.compile(r"^(gradle/libs\.versions\.toml|settings\.gradle\.kts|build\.gradle\.kts|gradle\.properties)$")


def is_test(path):
    return "/src/test/" in path or "/src/testFixtures/" in path


class Git:
    def __init__(self, repo):
        self.repo = repo

    def __call__(self, *args, ok_codes=(0,)):
        p = subprocess.run(["git", "-C", self.repo, *args], capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        if p.returncode not in ok_codes:
            raise RuntimeError(f"git {' '.join(args)}: {p.stderr.strip()}")
        return p.stdout


def main(repo, out):
    git = Git(repo)
    tree = git("ls-tree", "-r", "--name-only", SHA).splitlines()
    tree_set = set(tree)
    code = [n for n in tree if not n.startswith("docs/") and CODE_EXT.search(n)]
    classes = collections.defaultdict(list)
    for n in code:
        if n.endswith(".java"):
            classes[os.path.basename(n)[:-5]].append(n)
    dirs_by_name = collections.defaultdict(set)
    for n in tree:
        if n.startswith("docs/"):
            continue
        parts = n.split("/")
        for i in range(1, len(parts)):
            dirs_by_name[parts[i - 1]].add("/".join(parts[:i]))
    all_dirs = {d for found in dirs_by_name.values() for d in found}
    basenames = collections.defaultdict(list)
    for n in tree:
        if not n.startswith("docs/"):
            basenames[os.path.basename(n)].append(n)

    def existed_before(glob):
        return git("log", "-1", "--format=%h %cs", SHA, "--", f":(glob){glob}").strip() or None

    def still_used(token):
        return bool(git("grep", "-l", "-w", "-F", token, SHA, "--", ".", ":(exclude)docs/", ok_codes=(0, 1)).strip())

    results = {}
    for rec in RECORDS:
        path = f"{RECORD_ROOT}{rec}/README.md"
        text = git("show", f"{SHA}:{path}")
        ticks = set(re.findall(r"`([^`\n]+)`", text))
        camel = {t for t in ticks if re.fullmatch(r"[A-Z][A-Za-z0-9]+", t)}
        camel |= set(re.findall(r"\b[A-Z][a-z]+(?:[A-Z][A-Za-z0-9]*)+\b", text))
        kebab = {t for t in ticks if re.fullmatch(r"[a-z][a-z0-9]*(-[a-z0-9]+)+", t)}

        class_refs = sorted(t for t in camel if t in classes)
        module_refs = sorted(t for t in kebab if t in dirs_by_name)
        path_refs = sorted({t.strip("./") for t in ticks if "/" in t and t.strip("./") in tree_set | all_dirs
                            and not t.strip("./").startswith("docs/")})
        file_refs = sorted(t for t in ticks if "/" not in t and NAMED_FILE.search(t) and len(basenames.get(t, [])) == 1)

        linked = set()
        for c in class_refs:
            main_files = [f for f in classes[c] if not is_test(f)]
            linked.update(main_files or classes[c])
        expand = {d for m in module_refs for d in dirs_by_name[m]} | {p for p in path_refs if p in all_dirs}
        linked.update(f for f in code if not is_test(f) and any(f.startswith(d + "/") for d in expand))
        linked.update(p for p in path_refs if p in tree_set)
        linked.update(basenames[t][0] for t in file_refs)
        linked = sorted(linked)

        unresolved = {}
        for token in sorted(camel - set(class_refs)):
            unresolved[token] = existed_before(f"**/{token}.java")
        for token in sorted(kebab - set(module_refs)):
            unresolved[token] = existed_before(f"**/{token}/**")
        dead = sorted(t for t, seen in unresolved.items() if seen and not still_used(t))
        external = sorted(t for t in unresolved if t not in dead)

        history = git("log", "--follow", "--format=%H %cs", SHA, "--", path).splitlines()[::-1]
        first, first_date = history[0].split()
        last_date = history[-1].split()[1]
        add_files = [f for f in git("show", "--name-only", "--format=", first).splitlines() if f]
        add_code = [f for f in add_files if not NON_CODE.search(f)]
        edits = len(history) - 1

        def churn(files, follow=True):
            seen = set()
            for f in files:
                args = ["log", "--format=%H", f"{first}..{SHA}", "--", f]
                seen.update(git(*(args[:1] + (["--follow"] if follow else []) + args[1:])).split())
            return seen

        no_shared = [f for f in linked if not SHARED_CONFIG.match(f)]
        churn_code = len(churn(no_shared)) if linked else None
        results[rec] = {
            "path": path,
            "first_commit": first,
            "first_commit_date": first_date,
            "last_edit_date": last_date,
            "edits_after_first": edits,
            "resolved_refs": {"class": class_refs, "module": module_refs, "path": path_refs, "file": file_refs},
            "n_resolved_refs": len(class_refs) + len(module_refs) + len(path_refs) + len(file_refs),
            "dead_identifiers": dead,
            "unresolved_other": external,
            "adr_commit_files": len(add_files),
            "adr_commit_code_files": add_code,
            "linked_files": linked,
            "n_linked_files": len(linked),
            "churn_no_rename_following": len(churn(linked, follow=False)) if linked else None,
            "churn_with_shared_config": len(churn(linked)) if linked else None,
            "churn_code": churn_code,
            "stale": None if churn_code is None else round(churn_code / (1 + edits), 2),
            "stale_reason": "no_linked_files" if not linked else None,
        }
        r = results[rec]
        print(f"{rec}\n  refs={r['n_resolved_refs']} files={r['n_linked_files']} dead={dead}"
              f"\n  adr_commit: files={len(add_files)} code={add_code}"
              f"\n  churn plain={r['churn_no_rename_following']} follow={r['churn_with_shared_config']}"
              f" code={churn_code} edits={edits} stale={r['stale']}")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"sha": SHA, "commits": int(git("rev-list", "--count", SHA)), "records": results}, f, indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
