# Running Example: `shortlink`

- Decision: the whole book uses `shortlink`, a URL shortener service, as its running
  example
  - Chapters 01-06 use it to show how to steer a model
  - Chapters 07-11 use it as the code to write, review, and debug
  - Chapters 12-13 use it as the feature to plan and architect
  - Chapter 14 uses it as the code to refactor by design principles
  - Chapter 15 uses it as the code to harden with robustness techniques
  - Chapters 16-18 use it as the product to track, containerize, and release
- Why `shortlink` won over `ledgerlite` and `docsearch`
  - It is the only candidate that is a deployed service
    - Chapters 17 and 18 get a real container, a health check, and a release gate
  - It fills gaps that `book_map.md` marks as not covered
    - Canary analysis, automated rollback, and production incident loops (chapter 18)
    - Service boundaries and scalability trade-offs (chapter 13)
  - Its security flaws give chapters 09 and 11 high-stakes review and debug targets
- The rejected candidates are in `Alternatives Considered`

# The Example

## What It Is

- A small Python HTTP service with SQLite storage
  - `POST /links`: create a short code for a URL
  - `GET /{code}`: redirect to the stored URL
  - `GET /links/{code}/stats`: click count
  - `GET /health`: liveness check
- Starter state `v0`
  - Create, redirect, and stats endpoints
  - An optional `expires_at` field on a link
  - Three layers: API, service, storage
  - Tests with a test client
- Features added in Part III
  - Custom aliases
  - Rate limiting
  - Click analytics
- Delivery in Part IV
  - Container image
  - CI pipeline with fast and slow test tiers
  - Release on tag and a canary check

## Seeded Flaws in `v0`

- Open redirect: `POST /links` accepts any URL scheme
- Race condition: two requests can get the same short code
- Wrong time zone: `expires_at` is compared in local time, not UTC
- Missing authentication: the stats endpoint is open
- Import cycle between the service layer and the storage layer in one legacy module

## Oracles

- Each oracle decides without a human whether a prompt worked
  - `pytest` with a test client for the endpoints
  - Golden JSON responses for the API
  - `pyright` and the linter for style
  - A smoke test against the running container: `/health` and one redirect
  - A reproducer test per seeded flaw: it fails on `v0` and passes after the fix

## Prompts as Unit Tests

- Supports the todo "Use example as unit test"
  - Keep the `v0` starter repo under a tag
  - Keep one branch per chapter, each starting from `v0`
  - Store the chapter prompt next to its branch
  - Reset to `v0`, run the prompt, and score the output with the oracles above
- Scoring one run
  - Tests pass or fail
  - Seeded flaw found or missed (chapters 09 and 10)
  - Diff size and cost (chapter 03)

## Chapter Fit

- Chapter numbers follow `book_map.md`

| Chap | Task on `shortlink`                                                  | Oracle                                         |
| :--- | :------------------------------------------------------------------- | :--------------------------------------------- |
| 01   | Generate the short-code generator from one spec with two models      | Same unit tests for both outputs               |
| 02   | Write a skill that adds an endpoint under the repo rules             | Skill run on a sample endpoint, diff reviewed  |
| 03   | Run the task "add click counts to stats" in three harnesses          | Tests pass, diff size, cost                    |
| 04   | Implement redirect with the investigate, plan, edit, test loop       | `pytest` green after each step                 |
| 05   | Write a spec for custom aliases with acceptance criteria, non-goals  | Golden JSON responses, alias edge cases        |
| 06   | Delegation matrix: CRUD to the AI, redirect policy and auth to human | Confirmation points in the decision log        |
| 07   | Lint, type hints, golden-file API tests, coverage gate               | Linter and `pyright` clean, coverage met       |
| 08   | OpenAPI docstrings, README, C4 diagram, `CLAUDE.md`                  | Commands in the README run as written          |
| 09   | Review a PR that redirects to any URL (open redirect)                | Seeded flaw found by the AI, then the human    |
| 10   | Explain the collision-handling path and map its imports              | Call graph shows the race window and the cycle |
| 11   | Fix a failing test: a link expires one hour early                    | Reproducer fails before the fix, passes after  |
| 12   | Write PRD, design, and plan for aliases, rate limiting, analytics    | Docs follow the project templates              |
| 13   | Set the layers and write an ADR: SQLite versus Postgres              | No import cycle, dependency direction holds    |
| 14   | Refactor the collision-handling code into one deep module            | Tests stay green, the public interface shrinks |
| 15   | Add input validation, a storage timeout, and an idempotent create    | Bad input fails fast, a retry makes one link   |
| 16   | Issues, sub-issues, and stacked PRs for the Part III features        | Each PR passes the tests on its own            |
| 17   | Build the container image and run it                                 | Smoke test passes inside the container         |
| 18   | CI with fast and slow tiers, coverage gate, release on tag, canary   | A planted failure is blocked by the gates      |

## Risks and Mitigations

- Higher setup cost than a CLI: server, database, and HTTP test client
  - Use SQLite and the framework test client: no external service
  - Ship a `Makefile` or `invoke` target that runs everything in one command
- Security flaws can pull the discussion away from the workflow topic
  - Plant flaws only in chapters 09 and 11
  - Keep the fix of each flaw to a few lines
- Framework boilerplate makes generated diffs noisy and hard to compare
  - Pin one framework and its version in the starter repo
  - Keep `v0` to a few small modules
- Less reuse of existing repo material than `docsearch`
  - Reuse the Docker and CI docs cited in chapters 17 and 18 of `book_map.md`
- Borrow from `ledgerlite`
  - Property tests on invariants, e.g., short codes are unique and decode back to the
    URL
  - Mutation testing of those invariants (chapter 07, not covered in `book_map.md`)

# Selection Criteria

- Derived from the chapter goals in `book_map.md`
  - **Chapter coverage**: the system gives a concrete task for each of chapters 01-18
  - **Deterministic oracle**: tests, golden files, or a linter decide if a prompt
    worked
    - Needed to reuse the example as "unit tests" for the book prompts
  - **Seedable defects**: a bug or a design flaw can be planted for chapters 09, 10,
    and 11
  - **Architecture richness**: enough modules and layers for chapter 13, not so many
    that a reader gets lost
  - **Delivery fit**: the system maps onto the Docker and CI/CD flows of chapters
    17 and 18
  - **Low setup cost**: a reader runs it with no paid service and no cloud account
  - **Reuse of repo material**: the repo already holds docs, skills, or tools about
    the topic
- Constraints for every candidate
  - Python, so the repo rules in `.claude/skills/coding.rules.md` apply as they are
  - Final size of a few modules, not a framework
  - A `v0` starter repo with planted flaws, then one branch per chapter
  - A reset to `v0` before each run, so a prompt can run again on the same input

# Alternatives Considered

## Comparison

- Scores are my judgment, from 1 (weak) to 5 (strong)
  - For "Low setup cost", 5 means the cheapest setup

| Criterion              | `shortlink` (chosen) | `ledgerlite` | `docsearch` |
| :--------------------- | :------------------- | :----------- | :---------- |
| Chapter coverage       | 5                    | 4            | 4           |
| Deterministic oracle   | 4                    | 5            | 2           |
| Seedable defects       | 4                    | 5            | 3           |
| Architecture richness  | 4                    | 4            | 3           |
| Delivery fit           | 5                    | 3            | 3           |
| Low setup cost         | 3                    | 5            | 3           |
| Reuse of repo material | 2                    | 3            | 5           |
| **Total (max 35)**     | **27**               | **29**       | **23**      |

- `ledgerlite` scored higher on paper
  - The decision weights delivery fit and the gaps in chapters 13 and 18 above the
    cheaper setup

## `ledgerlite`: a Double-Entry Ledger CLI

- What it is: a Python library and CLI that record money movements as balanced
  postings
  - Commands: `add`, `import-csv`, `balance`, `report`
  - Invariant: the postings of one transaction sum to zero
- Strengths:
  - Strongest oracles: golden-file reports and property tests on the invariant
  - Lowest setup cost: no server, no key, no cloud account
  - A CLI matches the dockerized executables of chapter 17
- Why not chosen:
  - No running service, so chapter 17 monitoring and rollback stay uncovered
  - Dry domain
  - Single-user tool: no concurrency or security stories

## `docsearch`: Question Answering over Markdown Docs

- What it is: a CLI that indexes a directory of Markdown files and answers a question
  with cited passages
  - `index`: chunk the files and build a BM25 index
  - `ask`: retrieve passages and call an LLM behind a cached client
- Strengths:
  - Hallucination shows up as an answer with no supporting passage (chapter 01)
  - The answer prompt is a versioned artifact (chapter 05)
  - Reuses `helpers_root/docs/tools/helpers/all.hllm.explanation.md` for cached and
    tested LLM calls
- Why not chosen:
  - Needs an LLM key and a budget for the first run
  - Output is not fully deterministic without a replay cache
  - The subject (AI) and the tool (AI) blur: readers can confuse the two
  - Retrieval tuning invites scope creep
