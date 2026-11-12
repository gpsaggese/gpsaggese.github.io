# A Causal Proof of GP's Law

## Status

- **Status**: draft
- **Complete Specs**: TBD

## Core Idea

- "GP's law": a substantial share of world GDP (roughly $100-111T/year) is
  wasted in bad business decisions, possibly as high as 90%, driven by
  cognitive biases, underuse of data, and an inability to think
  counterfactually
- Businesses make hundreds of decisions per day across categories: pricing
  changes, hiring/promotions, capital projects, product feature priorities,
  vendor/partner selection
- Claimed failure modes: humans do not understand probability, do not use
  data, cannot think counterfactually, and rely on heuristics and gut
  feelings
- Evidence check: a literature review of the claim found real support for
  "large and material" decision-making losses (e.g., McKinsey estimates a
  typical Fortune 500 company loses about $250M per year to decision
  inefficiency), but no evidence-based support for a specific "90% of GDP"
  figure
  - The defensible framing: business decision-making is a massive,
    under-optimized source of economic value, but the exact share of GDP
    lost to poor decisions is currently unknown and not credibly quantified
- Goal: build a causal simulation framework, extending
  `draft.Causal_Analysis_of_Hedge_Fund_Performance.md`'s skill-versus-luck
  approach (see the Saggese et al. framework in References), to produce a
  defensible, non-vendor-backed estimate of the value lost to poor business
  decisions

## Formalization

- Mathematical notation, definitions, or pseudocode
- Use LaTeX math where helpful
  ```
  VC_eff = VC(H) + log(N_strategies_tested)
  ```

## Key Examples

- **World GDP scale**: the World Bank reports world GDP of about $111T in
  2024 (current US$), so "$100T/year" is a reasonable order-of-magnitude
  shorthand but now somewhat low (World Bank, 2024)
- **Decision inefficiency has a real, measurable cost**: McKinsey finds
  executives spend almost 40% of their time on decisions, that 60% of that
  time is used poorly, and that decision-making inefficiency costs a
  typical Fortune 500 company about $250M per year (McKinsey, 2023)
- **Executives are measurably miscalibrated**: in an NBER study of nearly
  7,000 forecasts from top US financial executives, realized stock-market
  returns fell inside their stated 80% confidence intervals only 38% of
  the time (Ben-David, Graham and Harvey, 2007)
- **A widely cited statistic needs caution**: the claim that "98% of 500
  managers failed to apply basic decision-making best practices" (Larson,
  2017) traces to a vendor-backed study (Cloverpop) without a transparent,
  peer-reviewed methodology, so it should be treated as an indicative data
  point, not a settled scientific benchmark

## Questions

1. What is the actual, empirically defensible share of global GDP lost to
   poor business decisions, given that the "90% of GDP" claim is not
   currently evidence-based?
2. Can a causal simulation framework, extending a skill-versus-luck
   approach, produce a credible, non-vendor-backed estimate of
   decision-quality losses?
3. Do the correlations between decision quality and firm performance
   (McKinsey, Bain, Brynjolfsson et al.) reflect a causal effect of better
   decision-making, or reverse causation from already-successful firms
   affording better decision processes?

## Research Topics

- **Decision-quality metrics**: define and compute per-decision metrics
  such as expected value, regret, risk-adjusted reward, learning rate,
  exploration versus exploitation, consistency, Bayesian rationality
  score, prospect-theory parameters, and utility-maximization gap
- **Overconfidence and capital allocation**: quantify how managerial
  miscalibration links to concrete corporate policies such as higher
  investment, more debt use, and payout behavior (Ben-David, Graham and
  Harvey, 2007)
- **Data-driven decision-making and firm performance**: test whether the
  productivity lift associated with data-driven decision-making
  (Brynjolfsson, Hitt and Kim, 2011; Brynjolfsson and McElheran, 2016)
  survives causal adjustment for reverse causation
- **Cross-domain bias survey**: extend the review of cognitive biases in
  professional decisions (management, finance, medicine, law) from
  Berthet (2022) to a quantitative, decision-level dataset
- **Counterfactual analysis in organizations**: study why organizations
  rarely institutionalize disciplined counterfactual analysis (explicit
  alternatives, pre-mortems, post-mortems, decision reviews), even though
  individual counterfactual thinking is a normal cognitive capability
  (Roese, 2000)

## Next steps

- [ ] Look for related research (what has already been done)
- [ ] Finalize the implementation plan
- [ ] GP to review / approve the plan
- [ ] Hack a quick end-to-end prototype (e.g., in 1-2 days) to show that you
      understood the problem and can make progress
- [ ] Break the problem down in phases and milestones
- [ ] Execute one step at the time

## Implementation plan

- Milestone 1: build the simulation framework
  - Extend the skill-versus-luck causal framework (Saggese et al.) to model
    business decisions (pricing, hiring, capital projects, product
    priorities, vendor selection) instead of agent competition

- Milestone 2: assemble a decision-quality benchmark
  - Combine existing human decision-making datasets:
    - **Iowa Gambling Task**: ~617 participants; risk, learning, and
      strategy under uncertainty (Bechara et al., 1994; Steingroever et
      al., 2015 pooled dataset)
    - **Balloon Analogue Risk Task (BART)**: risk tolerance and impulsivity
    - **Iterated Prisoner's Dilemma**: ~168,000 human decisions; cooperation
      and strategic reasoning (Lin et al., 2020)
    - **Large-scale human decision dataset**: ~240,000 judgments across
      ~13,000 decision problems; prospect theory and cognitive bias
      (Bourgin et al., 2019)
    - **100k real-life choice dilemmas**: ethical decision-making and
      judgment under ambiguity (Bhatia et al., 2025)
    - **Team management decision dataset**: 1,144 participants; scenarios,
      strategies, outcomes, and emotions
    - **Atari-HEAD**: human gameplay with eye tracking, actions, and
      scores; sequential decision-making (Zhang et al., 2019)
    - Deal-or-no-deal negotiation dataset
      (https://www.kaggle.com/datasets/parrotypoisson/deal-or-no-deal-games/code)

- Milestone 3: correlate decision quality with outcomes
  - Compute the decision-quality metrics per Research Topics on the
    benchmark, and correlate them with firm-level performance data (in the
    style of the McKinsey, Bain, and Brynjolfsson et al. studies), testing
    whether the correlation survives causal adjustment

- Milestone 4: produce a defensible estimate
  - Report a value-lost-to-poor-decisions estimate with uncertainty
    bounds, replacing the untested "90% of GDP" claim with an evidence-based
    figure

## References

- Saggese et al., _Causal Analysis of Agent Skill and Luck_,
  https://github.com/gpsaggese/gpsaggese.github.io/blob/master/papers/Causal_Analysis_of_Agent_Skill_And_Luck/Causal_Analysis_of_Agent_Skill_And_Luck.pdf
- World Bank, _World Bank Open Data: World_, 2024 GDP (current US$),
  https://data.worldbank.org/country/world
- BARC, _Global Survey on Data-Driven Decision-Making in Businesses_
  (2016), https://barc.com/data-driven-decision-making-business/
- Larson, _Don't Fail At Decision Making Like 98% Of Managers Do_, Forbes
  (2017),
  https://www.forbes.com/sites/eriklarson/2017/05/18/research-reveals-7-steps-to-better-faster-decision-making-for-your-business-team/
- Berthet, _The Impact of Cognitive Biases on Professionals' Decision-Making:
  A Review of Four Occupational Areas_, Frontiers in Psychology (2022),
  https://pmc.ncbi.nlm.nih.gov/articles/PMC8763848/
- Ben-David, Graham, and Harvey, _Managerial Overconfidence and Corporate
  Policies_, NBER Working Paper 13711 (2007),
  https://www.nber.org/papers/w13711
- McKinsey and Company, _Decision making in the age of urgency_ (2019),
  https://www.mckinsey.com/capabilities/people-and-organizational-performance/our-insights/decision-making-in-the-age-of-urgency
- McKinsey and Company, _How to make better decisions in the age of
  urgency_ (2023),
  https://www.mckinsey.com/featured-insights/mckinsey-guide-to-excelling-as-a-ceo/how-to-make-better-decisions-in-the-age-of-urgency
- Brynjolfsson, Hitt, and Kim, _Strength in Numbers: How Does Data-Driven
  Decisionmaking Affect Firm Performance?_ (2011),
  https://ide.mit.edu/sites/default/files/publications/2011.12_Brynjolfsson_Hitt_Kim_Strength%20in%20Numbers_302.pdf
- Brynjolfsson and McElheran, _The Rapid Adoption of Data-Driven
  Decision-Making_, American Economic Review 106, no. 5 (2016): 133-139,
  https://www.aeaweb.org/articles?id=10.1257/aer.p20161016
- Bain and Company, _Score your organization to improve decision
  effectiveness_ (2010),
  https://www.bain.com/insights/score-your-organization-ame-info/
- Sunstein, _Probability Neglect: Emotions, Worst Cases, and Law_, Yale Law
  Journal (2002), https://chicagounbound.uchicago.edu/law_and_economics/483/
- Gaissmaier et al., _Statistical illiteracy undermines informed shared
  decision making_, Zeitschrift fur Evidenz, Fortbildung und Qualitat im
  Gesundheitswesen (2009), https://pubmed.ncbi.nlm.nih.gov/19209567/
- Roese, _Counterfactual thinking and decision making_, Psychonomic
  Bulletin and Review (2000),
  https://www.researchgate.net/publication/12633695_Counterfactual_thinking_and_decision_making
- Bechara et al. (1994) and Steingroever et al. (2015): Iowa Gambling Task
  and its pooled dataset
- Bourgin et al. (2019): large-scale human decision dataset (13k problems)
- Bhatia et al. (2025): 100k real-life choice dilemmas dataset
- Zhang et al. (2019): Atari-HEAD human decision dataset
- Lin et al. (2020): Iterated Prisoner's Dilemma human decision dataset
