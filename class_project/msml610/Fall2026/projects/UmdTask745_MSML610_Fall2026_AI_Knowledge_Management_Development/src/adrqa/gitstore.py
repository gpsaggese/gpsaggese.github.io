"""All repository access. Reads git objects at the pinned SHA and never checks out.

About 100 EDC paths exceed the Windows path limit, so a checkout silently loses them.
"""
import subprocess
from functools import cached_property
from pathlib import Path

from .config import CLONE_DIR, REPO_URL, SHA


def is_test(path: str) -> bool:
    return "/src/test/" in path or "/src/testFixtures/" in path


class GitStore:
    def __init__(self, repo: Path = CLONE_DIR, sha: str = SHA):
        self.repo, self.sha = Path(repo), sha

    @staticmethod
    def ensure_clone(repo: Path = CLONE_DIR) -> None:
        if not (Path(repo) / ".git").is_dir():
            Path(repo).parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "clone", "-q", "--no-checkout", REPO_URL, str(repo)], check=True)

    def git(self, *args: str, ok: tuple[int, ...] = (0,)) -> str:
        p = subprocess.run(["git", "-C", str(self.repo), *args], capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        if p.returncode not in ok:
            raise RuntimeError(f"git {' '.join(args)}: {p.stderr.strip()}")
        return p.stdout

    @cached_property
    def tree(self) -> list[str]:
        return self.git("ls-tree", "-r", "--name-only", self.sha).splitlines()

    def read(self, path: str) -> str:
        return self.git("show", f"{self.sha}:{path}")

    def read_many(self, paths: list[str]) -> dict[str, str]:
        """Read many files in one `git cat-file --batch` process."""
        spec = "".join(f"{self.sha}:{p}\n" for p in paths).encode("utf-8")
        raw = subprocess.run(["git", "-C", str(self.repo), "cat-file", "--batch"], input=spec,
                             capture_output=True, check=True).stdout
        out, pos = {}, 0
        for p in paths:
            nl = raw.index(b"\n", pos)
            header = raw[pos:nl].split()
            if header[-1] == b"missing":
                raise KeyError(p)
            size = int(header[2])
            out[p] = raw[nl + 1:nl + 1 + size].decode("utf-8", "replace")
            pos = nl + 1 + size + 1
        return out

    def history(self, path: str) -> list[tuple[str, str]]:
        """(sha, date) of commits touching path, oldest first, following renames."""
        lines = self.git("log", "--follow", "--format=%H %cs", self.sha, "--", path).splitlines()
        return [tuple(l.split()) for l in reversed(lines)]

    def commits_since(self, commit: str, path: str, follow: bool = True) -> set[str]:
        args = ["log", "--format=%H", f"{commit}..{self.sha}"]
        if follow:
            args.append("--follow")
        return set(self.git(*args, "--", path).split())

    def files_in_commit(self, commit: str) -> list[str]:
        return [f for f in self.git("show", "--name-only", "--format=", commit).splitlines() if f]

    def last_touch(self, glob: str) -> str | None:
        return self.git("log", "-1", "--format=%h %cs", self.sha, "--", f":(glob){glob}").strip() or None

    def used_outside_docs(self, token: str) -> bool:
        out = self.git("grep", "-l", "-w", "-F", token, self.sha, "--", ".", ":(exclude)docs/", ok=(0, 1))
        return bool(out.strip())

    def blame_latest(self, path: str, start: int, end: int) -> dict | None:
        """Most recent commit touching these lines: provenance for the answer context."""
        out = self.git("blame", "--line-porcelain", "-L", f"{start},{end}", self.sha, "--", path, ok=(0, 128))
        best = None
        current: dict = {}
        for line in out.splitlines():
            head = line.split(" ", 1)
            if len(head[0]) == 40 and all(c in "0123456789abcdef" for c in head[0]):
                current = {"sha": head[0][:9]}
            elif head[0] == "committer-time":
                current["time"] = int(head[1])
            elif head[0] == "summary":
                current["summary"] = head[1] if len(head) > 1 else ""
                if best is None or current["time"] > best["time"]:
                    best = dict(current)
        return best
