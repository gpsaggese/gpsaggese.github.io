# Causal Analysis of Hedge Fund Performance

## Status

- **Status**: draft
- **Complete Specs**: TBD

## Core Idea

- Quantify the contribution of manager skill versus luck to hedge fund
  returns
- Analyze performance across market conditions, fund strategies, and time
  periods
- Apply causal inference techniques to isolate skill from random variation
- Compare hedge fund managers to establish the skill distribution and its
  persistence
- Study how fund characteristics (age, size, fee structure) relate to skill
  and luck
- Identify systematic underperformance and investigate its potential causal
  factors
- Project objective: build a causal model that separates skill from luck in
  hedge fund performance, using historical hedge fund data and causal
  inference methods to measure the degree to which manager outperformance
  is attributable to genuine skill versus favorable market conditions or
  randomness

## Formalization

- Mathematical notation, definitions, or pseudocode
- Use LaTeX math where helpful
  ```
  VC_eff = VC(H) + log(N_strategies_tested)
  ```

## Key Examples

- **[Example 1]**: [Concrete scenario illustrating the idea]
- **[Example 2]**: [Second scenario, possibly from a different domain]
- **[Example 3]**: [Edge case or failure mode]

## Questions

1. What fraction of hedge fund returns can be explained by manager skill,
   as opposed to market conditions or randomness?
2. How persistent is manager skill over time?
3. [Provocative implication: if true, what does this change?]

## Research Topics

- **Forward-looking skill predictor**: rank managers by estimated future
  performance
- **Fee-skill relationship**: investigate whether fund fees correlate with
  manager skill, or are simply transfers of returns
- **Manager turnover**: analyze the relationship between manager turnover
  and skill, and identify factors driving skilled-manager departures
- **Skill vs. luck dashboard**: build an interactive dashboard showing
  skill vs. luck attribution across different fund cohorts
- **Cross-vehicle comparison**: compare skill persistence in hedge funds
  versus traditional mutual funds or factor-based strategies
- **Survivorship bias**: study manager survivorship bias and its impact on
  estimated skill distributions

## Next steps

- [ ] Look for related research (what has already been done)
- [ ] Finalize the implementation plan
- [ ] GP to review / approve the plan
- [ ] Hack a quick end-to-end prototype (e.g., in 1-2 days) to show that you
      understood the problem and can make progress
- [ ] Break the problem down in phases and milestones
- [ ] Execute one step at the time

## Implementation plan

- Milestone 1: collect and clean data
  - Aggregate hedge fund returns from multiple sources, handle missing
    data, and standardize fund identifiers
  - Candidate datasets:
    - **CRSP Hedge Fund Database** (crsp.org): monthly returns, fund
      characteristics, strategy classification, fees, AUM; academic
      license required
    - **Morningstar Alternative Investments Database**: historical
      performance, factsheets, manager tenure, strategy allocation
    - **BarclayHedge Database**: monthly index returns by strategy;
      constituent fund data requires subscription
    - **EDHEC Alternative Indices** (edhec.edu): hedge fund indices by
      strategy, factor exposures, publicly downloadable

- Milestone 2: risk-adjust and decompose
  - Calculate risk-adjusted returns (Sharpe ratio, information ratio) and
    fund-specific factor exposures
  - Apply causal inference methods to estimate the skill component of
    returns, separate from market/factor effects

- Milestone 3: test persistence and benchmark
  - Test whether manager skill persists across time periods and market
    regimes
  - Compare skill metrics across fund strategies, sizes, and fee structures
  - Perform robustness checks and significance testing on skill estimates

## References

- Saggese et al., _Causal Analysis of Agent Skill and Luck_,
  https://github.com/gpsaggese/gpsaggese.github.io/blob/master/papers/Causal_Analysis_of_Agent_Skill_And_Luck/Causal_Analysis_of_Agent_Skill_And_Luck.pdf
  - Developed a causal framework to decompose agent performance into skill
    and luck components, applied to quantify skill versus randomness in
    competitive settings
- Fama and French, _Luck versus Skill in Mutual Fund Performance_ (2010),
  https://www.jstor.org/stable/25721416
  - Established baseline methods for separating manager skill from luck via
    statistical decomposition; found most mutual fund outperformance is
    attributable to luck rather than skill
- McLean and Pontiff, _Does Academic Research Destroy Stock Market
  Anomalies?_ (2015),
  https://onlinelibrary.wiley.com/doi/full/10.1111/jofi.12217
  - Studied how documented trading anomalies degrade after academic
    publication; relevant to survivorship bias and market efficiency
    effects on fund performance
- Arnott et al., _How Can 'Active' Investing Outperform? An Analysis of the
  Lies, Damned Lies, and Statistics of Outperformance_ (2020),
  https://www.researchaffiliates.com/documents/799-how-can-active-investing-outperform.pdf
  - Analyzed the components of active management outperformance (skill,
    beta, leverage, fees), quantifying the role of risk-taking and fee
    structures
