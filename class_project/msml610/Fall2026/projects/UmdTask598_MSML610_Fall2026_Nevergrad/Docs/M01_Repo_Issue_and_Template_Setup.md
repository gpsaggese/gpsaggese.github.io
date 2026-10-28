# Milestone 1 — Repo, Issue & Template Setup

> **GitHub Milestone**
> **Title:** `M1 — Repo, Issue & Template Setup`
> **Timeframe:** Week 1 (Sep 28 – Oct 4)
> **Depends on:** none
> **Description:**
> Do the open-source-contribution setup before any code is written. Fork/star/watch the course repos, install a container engine, confirm your issue assignment, create the tagged GitHub issue that becomes the project's discussion thread, cut the working branch off `master`, and copy the course Docker project template into the correct project directory. This is the substrate every later milestone commits into via PRs — there is no technical work here, but it is the one true prerequisite for everything that follows (§5). Confirm the exact term label (`Fall2026`) and course path with your TA/instructor before creating the issue and branch, since the guideline template literally shows `Spring{year}`.

## Notes
- **Confirm the term with your TA/instructor first.** The deadline (Mon Dec 7, 2026) places this in Fall 2026, but the guideline's tag pattern shows `Spring{year}`; lock the exact label and course path before creating the issue and branch (§5 term note).
- **Fork-and-PR, not direct-clone.** You have pull-only access to upstream, so clone your fork and wire the `upstream` remote once; never push to upstream `master` (§5.0). Use `GIT_LFS_SKIP_SMUDGE=1` on the clone to get past upstream's broken LFS object.
- **Upstream slug is `gpsaggese/gpsaggese.github.io`**, not the old `umd_classes` name the guidelines use (§5.1, GOTCHAS §1).
- **Files go only under the project directory:** `{GIT_ROOT}/class_project/msml610/Fall2026/projects/{branch_name}` — note the **lowercase** `msml610` even though the tag is uppercase (§5.4, GOTCHAS §5).
- **Reviewer latency is not idle time.** Plan to open the draft PR early (Week 2) and keep moving even if review is slow (§5.6).

---

## Issues

### `M1-1` — Prerequisites: repos, Docker, issue assignment
- **Assignee:** @jake
- **Labels:** `setup`, `repo`, `docker`, `core`
- **Blocked by:** none
- **Blocks:** `M1-2`, `M1-3`

**Context.** The one-time setup that gates the whole project: watch/star/fork `gpsaggese/gpsaggese.github.io` (the course repo, called `umd_classes` in the guidelines) and `causify-ai/helpers`, install a container engine, and confirm you are assigned on the issues tracker. Nothing here is code, but it must be done before the issue and branch exist.

**Deliverables**
- `gpsaggese.github.io` and `helpers` watched, starred, and forked to your account.
- A container engine installed and verified — **Docker** (Linux / Docker Desktop on macOS) or Apple's `container` tool (§4.1).
- Issue assignment confirmed on the `gpsaggese.github.io` issues tracker — or, if the Assignees field 403s under pull-only access, the documented fallback used (tag yourself as `Author:` in the body) and collaborator access requested (GOTCHAS §3).

**Definition of Done**
- [ ] Both repos forked and starred; forks visible under your account.
- [ ] Container engine installed and a hello-world image runs.
- [ ] Issue assignment confirmed (or the author-tag fallback used and access requested).

---

### `M1-2` — Create the tagged GitHub issue
- **Assignee:** @jake
- **Labels:** `setup`, `github`, `core`
- **Blocked by:** `M1-1`
- **Blocks:** `M1-3`

**Context.** Create the GitHub issue that becomes the discussion thread for the whole project, titled with the project tag. The tag format is `{Class}_{Term}{Year}_{project_title_without_spaces}` (§5.2); confirm the term label with your TA/instructor before committing to it.

**Deliverables**
- Project tag finalized: `MSML610_Fall2026_Nevergrad` (term confirmed).
- Issue created with that tag as its title, the project description, and a link to the spec (`.../MSML610/nevergrad_Project_Description.md`) and this plan (§5.3).
- Issue assigned to yourself (or author-tagged per the fallback); issue number recorded for the branch name — **#598**.

**Definition of Done**
- [ ] Term label (`Fall2026`) and course path confirmed with TA/instructor.
- [ ] Issue created with the project tag as its title and the spec/plan linked.
- [ ] Issue self-assigned (or author-tagged); issue number recorded (#598).

---

### `M1-3` — Create the branch & copy the template into the project directory
- **Assignee:** @jake
- **Labels:** `setup`, `repo`, `docker`, `core`
- **Blocked by:** `M1-1`, `M1-2`
- **Blocks:** `M2-1`

**Context.** Sync off the latest upstream `master`, cut the working branch, and copy the course Docker template into the correct project directory. The template supplies the `Dockerfile`, the `docker_*.sh` helper scripts, config files, and the notebook templates — the skeleton the next milestone builds on. Never commit to `master`; push branches to your fork (`origin`).

**Deliverables**
- Branch `UmdTask598_MSML610_Fall2026_Nevergrad` created off an up-to-date `master` (`git fetch upstream && git checkout master && git merge upstream/master`).
- Template copied to `{GIT_ROOT}/class_project/msml610/Fall2026/projects/{branch_name}` via `cp -r class_project/project_template <dir>` (§4.2, §5.4). Do **not** use `create_project.py --action create_links` (symlinks would leave the submission non-self-contained).
- Initial commit of the untouched template on the branch.

**Definition of Done**
- [ ] Branch created off the latest `master` with the correct `UmdTask598_...` name.
- [ ] Template present at the correct **lowercase** `msml610` path with all helper scripts and notebook templates.
- [ ] Files live only under the project directory; nothing added to `master`.
- [ ] Template committed to the branch as a clean starting point.
