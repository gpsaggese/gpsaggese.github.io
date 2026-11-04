# Task: Tutorial for the Semantic Scholar API

Create a Jupyter tutorial that teaches the Semantic Scholar Academic Graph API
through small, runnable examples. Use the `notebook.*` skills listed below.

Do not commit. Do not push. Read and follow `.claude/rules.md` first.

## Background to Read First

- `research/Estimate_academic_paper_quality/semantic_scholar.md`: what the service
  is, which endpoints and fields exist, the rate limits, and the API key
- `research/Estimate_academic_paper_quality/epq_bib_sources.py`: a working client
  (throttle, retry on `429` and `5xx`, cache). Use it as a reference only. Do not
  import it. The tutorial must be self-contained.
- Official docs: https://api.semanticscholar.org/api-docs/
  - The docs page answered `429` on 2026-11-03. If it fails again, work from
    `semantic_scholar.md` and from a live call with `curl`.
  - Mark every claim that you could not verify with `TODO(gp): Check ...`

## Decisions (already made, do not ask again)

- Package name for the skills: `semantic_scholar`
- Tutorial dir: `tutorials/semantic_scholar/`
- Notebook:
  `tutorials/semantic_scholar/semantic_scholar.01.API.academic_graph.ipynb`
- Paired file: `tutorials/semantic_scholar/semantic_scholar.01.API.academic_graph.py`
  (Jupytext, `py:percent`)
- The tutorial covers the REST API with plain `requests` calls
  - It does not use the `semanticscholar` PyPI client
  - There are no classes of a library to introspect. Instead of
    `hintros.print_obj_info()`, give the link to the official docs of each
    endpoint, right after the first call to it.
- Services: the Academic Graph API (base URL `https://api.semanticscholar.org/graph/v1`)
  is the core. The Recommendations API is the last section and is optional. Skip the
  Datasets API: only name it in the overview.
- Toy objects: 2 or 3 well-known papers with a fixed id, for example
  `ARXIV:1706.03762` and `ARXIV:1810.04805`. Use no realistic dataset.
- API key
  - Read it from the environment variable `SEMANTIC_SCHOLAR_API_KEY`
  - The notebook must run without a key. Without a key it can get `429` from the
    shared pool: the notebook must explain this and must not crash.
  - Send the key only in the header `x-api-key`. Never print it, never write it
    in a file, never commit it.
  - Check that `docker_bash.sh` and `docker_jupyter.sh` pass the variable into the
    container. If they do not, tell the user. Do not hard code the key.
- Citation counts and search results change over time
  - Do not write exact counts in markdown cells
  - Write the expected shape of the outcome, for example "an integer, larger than
    0"

## Skills to Execute

Execute them in this order. Each skill follows its own `SKILL.md`.

1. `.claude/skills/notebook.create_api_intro/SKILL.md`
   - `<PACKAGE_NAME>` is `semantic_scholar`
   - `<FILE>` is the notebook name in the Decisions section
   - Content to cover, one primitive at a time, each with a mental model table
     (Object | Description | Additional Info):
     1. Overview: the three services, the base URL, and a diagram of the objects
        `Paper`, `Author`, `Citation`, `Venue`
     2. A paper by id: `GET /paper/{id}` with the id forms `paperId`, `DOI:<doi>`,
        and `ARXIV:<id>`
        - Show that the default answer has few fields, and that `fields=` selects
          more: `title`, `year`, `authors`, `externalIds`, `openAccessPdf`,
          `citationCount`, `publicationDate`, `publicationTypes`
     3. Search: `GET /paper/search` with `query`, `limit`, `offset`, `fields`
        - Show the keys `total`, `offset`, `next`, `data`, and one page change
     4. Batch lookup: `POST /paper/batch` with a list of ids
     5. Authors: `GET /author/search`, `GET /author/{id}`, and
        `GET /author/{id}/papers`
     6. Citations and references: `GET /paper/{id}/citations` and
        `GET /paper/{id}/references`, and the wrapper keys `citingPaper` and
        `citedPaper`
     7. Errors and limits: `404` for a missing paper, `429` for the rate limit,
        the retry with a wait of 2, 4, 8, then 16 seconds
     8. Minimal end-to-end: search a title, take the best hit, read its citation
        count, list its first 3 references
     9. (Optional) Recommendations API: papers similar to a given paper
   - Keep every example within 10 to 20 lines. Add a comment before each call and a
     comment with the outcome after it, as the skill says.
   - Keep the number of live calls low (about 20 in the whole notebook), and wait
     1 second between calls, so a key with the 1 request per second limit works.
2. `.claude/skills/notebook.fix_packages/SKILL.md`
   - The Docker files come from `tutorials/project_template`. Copy them if
     `tutorials/semantic_scholar/` does not exist yet.
   - Set `IMAGE_NAME=umd_project_semantic_scholar` in `docker_name.sh`
   - Install `requests` and any other needed package in the notebook
3. `.claude/skills/notebook.fix_logging/SKILL.md`
4. `.claude/skills/notebook.split_cells/SKILL.md`

## Verification

- Pair the notebook with Jupytext:
  `jupytext.py --action pair --files tutorials/semantic_scholar/semantic_scholar.01.API.academic_graph.ipynb`
- Run the notebook end to end in the container, following `## Testing Notebook` in
  `.claude/skills/notebook.rules.md`
- Run it twice: with the key and without the key. Both runs must finish without an
  error. Without a key, a `429` must show a clear message.
- Lint the touched files, following the lint rules in `.claude/rules.md`

## Out of Scope

- The `semantic_scholar.example.ipynb` notebook, the README, the blog post, and
  unit tests. Ask the user before you add any of them.
- Any change to `research/Estimate_academic_paper_quality/`

## Done When

- The notebook and its paired `.py` file exist and are in sync
- The notebook runs end to end in the container, with and without a key
- Every endpoint above has a runnable example and a link to the official docs
- No key, no exact citation count, and no `tmp.*` file in the output
- Nothing is committed
- The final report lists: files created, the number of live calls, the endpoints
  that you could not verify against the docs, and any step left undone

## Plan

- [x] Read rules and background
  - [x] `.claude/rules.md`, `notebook.rules.md`, 4 skill files, API templates
  - [x] `semantic_scholar.md`, `epq_bib_sources.py` (reference only)
- [x] Pre-checks
  - [x] `docker_bash.sh` and `docker_jupyter.sh` do **not** pass
        `SEMANTIC_SCHOLAR_API_KEY` into the container: tell the user in the
        final report, do not edit the scripts
  - [x] Live calls of each endpoint, to verify field names and wrapper keys
        (the docs page answers `429`, so the official OpenAPI spec at
        `/graph/v1/swagger.json` and `/recommendations/v1/swagger.json` is the
        reference)
- [x] Skill 1: `notebook.create_api_intro`
  - [x] Copy the Docker files of `tutorials/project_template` to
        `tutorials/semantic_scholar`
  - [x] Write `semantic_scholar.01.API.academic_graph.py` (`py:percent`)
  - [x] Parts 1 to 9 as listed above, 18 live calls, 1 s between calls
  - [x] Write `semantic_scholar_utils.py` with `init_loggers()`
  - [x] Create the `.ipynb` and pair it with Jupytext (done in the container,
        since `jupytext` is not installed on the host)
- [x] Skill 2: `notebook.fix_packages`
  - [x] `IMAGE_NAME=umd_project_semantic_scholar` in `docker_name.sh`
  - [x] Pinned `!pip install -q requests==2.34.2 graphviz==0.21` cell
- [x] Skill 3: `notebook.fix_logging` (`utils.init_loggers(_LOG)` in the last
      setup cell)
- [.] Skill 4: `notebook.split_cells`
- [ ] Verification
  - [ ] Notebook and `.py` in sync
  - [ ] Run in container without key
  - [ ] Run in container with key (pass it with `docker run -e
        SEMANTIC_SCHOLAR_API_KEY`, name only, so the value is never printed)
  - [ ] Lint touched files per `.claude/rules.md`
  - [ ] No key, no exact count, no `tmp.*` file in the output
- [ ] `git add` the new files, do not commit
