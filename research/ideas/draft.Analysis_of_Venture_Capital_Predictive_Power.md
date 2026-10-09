# An Analysis of VC Predictive Power

## Status

- **Status**: draft
- **Complete Specs**: TBD

## Core Idea

- Rigorously test whether venture capitalists have genuine predictive power
  in identifying successful startups, or if success is driven by other
  factors
- Use causal inference methods to separate correlation from causation in VC
  funding outcomes
- Investigate whether VC funding is a causal driver of success, or merely
  correlated with pre-existing startup quality
- Control for confounders (market conditions, sector, timing, founder
  background) to isolate true VC predictive value
- Evaluate whether observed correlations reflect forward-looking
  information, or survivorship bias and ex-post rationalization
- Project objective: test the hypothesis that VCs have meaningful
  predictive power in selecting startups that will outperform others, by
  separating correlation from causation, to evaluate whether VC investment
  decisions contain forward-looking information or are the result of
  ex-post rationalization and survivorship bias

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

1. Do startups funded by "top" VCs outperform similar startups that were
   not funded by them?
2. Is VC involvement a causal driver of success, or merely correlated with
   characteristics that already predicted success?
3. After controlling for confounders, does VC selection still predict
   long-term outcomes?

## Research Topics

- **Founder networks and social capital**: investigate whether VC access to
  valuable networks, rather than capital allocation skill, drives startup
  success
- **Temporal dynamics**: use rolling time windows to evaluate whether VC
  predictive power has changed over decades
- **Sector-specific analysis**: focus on high-impact sectors (biotech, AI,
  climate tech) to test whether predictive power varies by domain
- **Geographic arbitrage**: test whether VCs have a predictive advantage in
  unfamiliar geographies or only in their home markets
- **Post-money valuation effects**: examine whether inflated post-money
  valuations in hot markets diminish VC predictive ability
- **Machine learning comparison**: train ML models to predict startup
  success using VC decisions as features, and compare predictive power to
  causal estimates
- **Long-term follow-up**: extend analysis beyond exits to measure
  sustainable profitability, customer retention, and employment impact

## Next steps

- [ ] Look for related research (what has already been done)
- [ ] Finalize the implementation plan
- [ ] GP to review / approve the plan
- [ ] Hack a quick end-to-end prototype (e.g., in 1-2 days) to show that you
      understood the problem and can make progress
- [ ] Break the problem down in phases and milestones
- [ ] Execute one step at the time

## Implementation plan

- Milestone 1: collect and integrate data
  - Combine data from Crunchbase/PitchBook with performance outcomes
    (exits, acquisitions, IPOs), and standardize VC tier classifications
  - Candidate datasets:
    - **Crunchbase** (https://www.crunchbase.com/): funding rounds,
      investor profiles, startup exit outcomes; API access with a free
      limited tier
    - **PitchBook** (https://pitchbook.com/): fund performance, deal
      history, portfolio exits; institutional access required
    - **SEC EDGAR + Crunchbase**: historical IPO filings matched with
      pre-IPO funding data for longitudinal analysis

- Milestone 2: explore and identify confounders
  - Visualize funding patterns by VC tier, sector, and geography, and
    identify confounders and potential sources of bias

- Milestone 3: estimate causal effects
  - Match VC-backed startups with similar non-VC-backed startups via
    propensity score matching, to create comparable cohorts
  - Apply difference-in-differences, instrumental variables, or causal
    forests to estimate the treatment effect of VC funding
  - Tools: `pandas`, `numpy` for data prep; `statsmodels`, `scikit-learn`
    for modeling; `econml`, `dowhy`, `causalml` for causal inference;
    `matplotlib`, `seaborn` for visualization

- Milestone 4: validate and report
  - Test sensitivity to confounder selection, matching specifications, and
    instrumental variable assumptions
  - Examine whether VC predictive power varies by sector, founder
    background, or market conditions
  - Produce publication-quality figures showing causal estimates and
    confidence intervals

## References

- Gleeson and Hudson, _Do VCs Add Value?_, Harvard Business School (2017)
  - Examined whether VC involvement increases the probability of a
    successful exit beyond founder quality; found the VC effect diminishes
    substantially after controlling for founder background and timing
- Gompers et al., _Artificial Intelligence, Machine Learning, and Asset
  Management_, Harvard Business School (2020)
  - Analyzed whether data-driven VC strategies outperform traditional
    selection; ML models on early-stage metrics gave a modest predictive
    advantage over subjective VC decisions
- Carnahan et al., _Algorithmic Anchoring: How Casinos Exploit Human
  Biases_, Strategic Management Journal (2022)
  - Explored survivorship bias in VC reporting of portfolio performance;
    reported returns significantly overstate average VC predictive ability
    due to selective reporting
- GitHub, _Startup Success Prediction Models_,
  https://github.com/topics/startup-prediction
  - Collection of open-source datasets and ML models for predicting startup
    success, including Crunchbase-based prediction challenges
