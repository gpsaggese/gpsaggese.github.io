# Noesis MVP PRD: Open Items

- The rest of the former MVP PRD (v0.2) is merged into `plan.md`,
  `design.checker.md`, `decisions.md`, and `glossary.md`
- Only the statements below remain: each one contradicts another doc
- Resolve each item, then delete this file

## 1. Error Code for Another Buyer's Contract

- **PRD (G2)**: "Gateway rejects (HTTP 409) if the contract isn't ACTIVE or doesn't
  belong to the caller"
- **Other docs**: `design.gateway.md` (One Request, End to End, step 2) and
  `decisions.md` D-7: another buyer's contract returns 404, the same as a missing
  one. 409 is only for a contract that is not `ACTIVE`
- **Question**: keep 404? Then G2 loses the "doesn't belong to the caller" clause

## 2. Monthly Spend Cap

- **PRD (G8)**: "Hard spend cap: gateway refuses upstream calls once cumulative real
  cost >= `SPEND_CAP_USD` for the calendar month"
- **PRD (section 6)**: spend cap of $50/month (Proposed, D9); a 1-hour demo is about
  1 to 3 M tokens, about $1 to $3 at 70B prices, to verify in week 1
- **PRD (D9)**: "Us, $50/mo cap enforced in gateway (G8). Apply for
  OpenRouter/provider startup credits"
- **Other docs**:
  - `decisions.md` D-9: no gateway-side cap yet; the OpenRouter key's credit limit is
    the cap
  - `design.gateway.md` (Configuration): `NOESIS_SPEND_CAP_USD` is "Read but not
    enforced yet"
  - `plan.md` T1.4: spend cap deferred
- **Question**: drop G8, or reinstate the cap? No doc covers the action "apply for
  startup credits"; where should it go?

## 3. Standalone Repo With a `helpers` Shim

- **PRD (section 4, In)**: "Standalone repo: GP's `helpers` dependency replaced by a
  small local shim"
- **Other docs**: `plan.md` T0.2: this repo uses GP's real `helpers`; the stand-in
  lives only in the team's backup repo
- **Question**: delete the PRD bullet?

## 4. When the Contract Window Starts

- **PRD (K6)**: "Activated immediately -> `ACTIVE` with `window_start`"
- **PRD (K7)** and `design.checker.md` (Evaluation flow, step 1): the window starts
  on the first valid buyer request
- **Question**: is `window_start` set at activation or at the first valid buyer
  request? (`window_start` is nullable in `common/migrations/001_init.sql`)

## 5. Window Definition in the Glossary

- **PRD (K7)** and `design.checker.md` (Window policy): the window starts on the
  first valid buyer request and closes when its targets are served or its deadline
  elapses (60 s for the demo, at most 300 s otherwise)
- **`glossary.md` (Window)**: "The requests a contract covers: its task count or
  60 s, whichever comes first"
- **Question**: update the glossary row to match the checker design?

## 6. Owner of the Dashboard

- **PRD (D11)**: the third owner covers "market + reputation + dashboard"
- **`plan.md` (Owners, T4.2)**: `L` owns the dashboard and `G` only defines what it
  must tell a non-technical viewer
- **Question**: which assignment is right?

## 7. How Contract Fulfillment Works

- **PRD (K8)**: "Replace `mock_fulfill` coin-flip: contract result comes from the
  metering verdict. Keep `mock_fulfill` for unit tests only". The v0.1 to v0.2 change
  list says `dispatch_contract()` returns a bool right away, which doesn't fit
- **Other docs**: `plan.md` `PR_M8` still says to replace `mock_fulfill()` with a
  real call through `Gateway.call()`, translating each `Contract` into Gateway
  calls. Its task `T3.3` and `decisions.md` D-24 say the result is the checker's
  verdict after the window closes
- **Question**: rewrite the `PR_M8` text to match D-24?
