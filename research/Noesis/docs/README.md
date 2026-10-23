# Docs

- All docs of `research/Noesis`, grouped by kind
- Start with `onboarding.md`, then `design.architecture.md`
- Component READMEs live next to the code: `../checker/`, `../common/`, `../gateway/`,
  `../market/`, `../noesis_platform/`, `../prober/`

## Structure of the Dir

| Dir         | Description                                                    |
| :---------- | :------------------------------------------------------------- |
| `findings/` | Raw measurements, e.g., the OpenRouter provider spike          |

- All other docs sit at the top level of `docs/`, grouped by the cluster in the
  table below

The canary calibration command generates `findings/calibration.md`; a pending
placeholder is committed until the reviewed paid run is executed.

## Description of Files

| File                           | Description                                                                           | Cluster    |
| :----------------------------- | :------------------------------------------------------------------------------------ | :--------- |
| `design.architecture.md`       | C4-model architecture (context, container, component) of the code                     | Design     |
| `design.checker.md`            | Checker evaluation flow, thresholds, window policy, canaries, and what is not built   | Design     |
| `design.gateway.md`            | Gateway design: request flow, blame rules, faults, endpoints, config                  | Design     |
| `glossary.md`                  | Terms used across the code and docs                                                   | Onboarding |
| `how_we_work.md`               | Branch, PR, test, and review conventions                                              | Onboarding |
| `onboarding.md`                | Setup, run, test, and first tasks                                                     | Onboarding |
| `run_market_supply_demo.how_to_guide.md` | Commands that show OpenRouter prices becoming asks and clearing against a bid     | Demo       |
| `plan.md`                      | Roadmap (`PR_M*`, `PR_S*`, `PR_P*`) and MVP plan (claim, rules, criteria, risks, tasks) | Plans      |
| `progress.md`                  | Live MVP status and history                                                           | Plans      |
| `marketing_plan.md`            | Bootstrapping supply and demand liquidity                                             | Plans      |
| `decisions.md`                 | MVP decisions and the alternatives considered                                         | Specs      |
| `spec.PR_M16.md`               | Implementation spec: economic surplus of Noesis vs a posted-price baseline            | Specs      |
| `spec.PR_P2.md`                | Implementation spec: containerized cloud deployment                                   | Specs      |
| `spec.PR_P2b.md`               | Implementation spec: Postgres backend                                                 | Specs      |
| `spec.PR_S13.md`               | Implementation spec: standalone monitoring and model-nerfing detection                | Specs      |
| `dev_notes.md`                 | Build and OpenAPI-generation notes                                                    | Notes      |
| `gp_todos.md`                  | Open TODOs                                                                            | Notes      |
