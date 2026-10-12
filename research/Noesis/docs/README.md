# Docs

- All docs of `research/Noesis`, grouped by kind
- Start with `onboarding.md`, then `architecture.md`
- Component READMEs live next to the code: `../checker/`, `../common/`, `../gateway/`,
  `../market/`, `../noesis_platform/`

## Structure of the Dir

| Dir         | Description                                                    |
| :---------- | :------------------------------------------------------------- |
| `plans/`    | What to build and when: roadmaps, MVP plan, progress, issue plans |
| `specs/`    | What and why: MVP requirements, decisions, per-PR implementation specs |
| `notes/`    | Code walkthroughs, AI prompts, dev notes, TODOs                |
| `findings/` | Raw measurements, e.g., the OpenRouter provider spike          |

## Description of Files

| File                | Description                                                        | Cluster    |
| :------------------ | :----------------------------------------------------------------- | :--------- |
| `architecture.md`   | C4-model architecture (context, container, component) of the code  | Design     |
| `checker_design.md` | Checker evaluation flow, window policy, and what is not built      | Design     |
| `gateway_design.md` | Gateway design: request flow, blame rules, faults, endpoints, config | Design   |
| `glossary.md`       | Terms used across the code and docs                                | Onboarding |
| `how_we_work.md`    | Branch, PR, test, and review conventions                           | Onboarding |
| `onboarding.md`     | Setup, run, test, and first tasks                                  | Onboarding |
| `plans/mvp_plan.md` | MVP milestones, owners, and task breakdown                         | Plans      |
| `plans/mvp_progress.md` | Live MVP status and history                                    | Plans      |
| `plans/plan.Noesis.md` | Protocol implementation roadmap (`PR_M*`, `PR_S*`, `PR_P*`)     | Plans      |
| `plans/plan.marketing.md` | Bootstrapping supply and demand liquidity                    | Plans      |
| `plans/plan.reorg.md` | Issue plan: reorganize the dir into components                   | Plans      |
| `specs/mvp_prd.md`  | MVP product requirements                                           | Specs      |
| `specs/mvp_decisions.md` | MVP decisions and the alternatives considered                 | Specs      |
| `specs/spec.PR_P2.md` | Implementation spec: containerized cloud deployment              | Specs      |
| `specs/spec.PR_P2b.md` | Implementation spec: Postgres backend                           | Specs      |
| `specs/spec.reorg.md` | Implementation spec: component layout                            | Specs      |
| `notes/explain_code.*.md` | Walkthroughs of the original market and platform modules     | Notes      |
| `notes/coding.create_specs.md` | Prompt template for the next `plan.Noesis.md` spec      | Notes      |
| `notes/dev_notes.md` | Build and OpenAPI-generation notes                                | Notes      |
| `notes/gp_todos.md` | Open TODOs                                                         | Notes      |
