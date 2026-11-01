# Docs

- All docs of `research/Noesis`, in reading order
- Start with `09-onboarding.md`, then `02-design.architecture.md`
- Component READMEs live next to the code: `../checker/`, `../common/`, `../gateway/`,
  `../market/`, `../noesis_platform/`, `../prober/`

## Structure of the Dir

| Dir         | Description                                                    |
| :---------- | :------------------------------------------------------------- |
| `findings/` | Raw measurements, e.g., the OpenRouter provider spike          |

- All other docs sit at the top level of `docs/`
  - The number at the start of a name is the slot of the doc and gives the reading
    order, see `.claude/skills/project.rules.md` section "Name the Docs with a Slot
    Number"
  - The docs without a number come last in the table below

The canary calibration command generates `findings/calibration.md`; a pending
placeholder is committed until the reviewed paid run is executed.

## Description of Files

| File                                     | Description                                                                              | Cluster    |
| :--------------------------------------- | :--------------------------------------------------------------------------------------- | :--------- |
| `02-design.architecture.md`              | C4-model architecture (context, container, component) of the code                        | Design     |
| `02.1-design.gateway.md`                 | Gateway design: request flow, blame rules, faults, endpoints, config                     | Design     |
| `02.2-design.checker.md`                 | Checker evaluation flow, thresholds, window policy, canaries, and what is not built      | Design     |
| `03-decisions.md`                        | MVP decisions and the alternatives considered                                            | Specs      |
| `04-plan.md`                             | Roadmap (`PR_M*`, `PR_S*`, `PR_P*`) and MVP plan (claim, rules, criteria, risks, tasks)  | Plans      |
| `06-spec.PR_M16.md`                      | Implementation spec: economic surplus of Noesis vs a posted-price baseline               | Specs      |
| `06-spec.PR_P2.md`                       | Implementation spec: containerized cloud deployment                                      | Specs      |
| `06-spec.PR_P2b.md`                      | Implementation spec: Postgres backend                                                    | Specs      |
| `06-spec.PR_S4.md`                       | Implementation spec: contract-window runtime and market-reporting boundary               | Specs      |
| `07-progress.md`                         | Live MVP status and history                                                              | Plans      |
| `08-how_we_work.md`                      | Branch, PR, test, and review conventions                                                 | Onboarding |
| `09-onboarding.md`                       | Setup, run, test, and first tasks                                                        | Onboarding |
| `10-glossary.md`                         | Terms used across the code and docs                                                      | Onboarding |
| `run_model_qos_demo.how_to_guide.md`     | Commands that demonstrate canary calibration, probing, grading, and evidence persistence | Demo       |
| `run_market_supply_demo.how_to_guide.md` | Commands that show OpenRouter prices becoming asks and clearing against a bid            | Demo       |
| `marketing_plan.md`                      | Bootstrapping supply and demand liquidity                                                | Plans      |
| `dev_notes.md`                           | Build and OpenAPI-generation notes                                                       | Notes      |
| `gp_todos.md`                            | Open TODOs                                                                               | Notes      |
