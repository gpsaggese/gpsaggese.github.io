# A Causal Analysis of Age and Startup Success

## Status

- **Status**: draft
- **Complete Specs**: TBD

## Core Idea

- Causal inference project investigating the relationship between founder
  age and startup success metrics
- Popular narratives emphasize very young founders, but large-scale
  empirical data (2.7 million founders, US Census data 2007-2014) shows
  startup founders are typically middle-aged, with an average founder age
  of 41.9 years and an average age of 45 among the top 0.1% fastest-growing
  firms (Azoulay et al., 2018)
- Hypothesis to test: is founder age a true causal driver of startup
  success, or merely correlated with success because it proxies for
  accumulated human capital (industry experience), social capital
  (networks), and financial capital (savings, credit)?
- Multi-source data integration combining founder demographics, funding
  information, and exit outcomes (acquisition/IPO/failure), controlling for
  confounders (industry, team size, education, market conditions) via
  counterfactual reasoning: what would happen if a founder of a different
  age started the same company?
- Real-world impact: results inform age bias in venture capital and
  startup ecosystems, and challenge the "young founder myth" driven by
  media attention to exceptional outliers
- Use a causal framework such as Saggese et al., _Causal Analysis of Agent
  Skill and Luck_ (see References)

## Formalization

- Regression model, using founder age and its square to test nonlinearity:
  ```text
  Success_i = b0 + b1*Age_i + b2*Age_i^2 + b3*IndustryExperience_i
              + b4*Education_i + b5*StartupExperience_i
              + IndustryFE + YearFE + e_i
  ```
- Dependent variables (startup success): revenue growth, employment
  growth, venture capital funding, acquisition or IPO, firm survival,
  profitability
- Independent variables: founder age, founder age squared
- Control variables: industry, geography, founding year, founder
  education, prior startup experience, prior industry experience, team
  size, gender composition, funding status
- Interpretation:
  - If age loses significance after adding experience controls, age likely
    acts as a proxy for experience
  - If age remains significant, an independent life-stage effect may exist

## Key Examples

- **Large-scale Census evidence**: among 2.7 million US founders
  (2007-2014), a 50-year-old founder is about 1.8 times more likely than a
  30-year-old founder to build a top-growth firm, and older founders were
  roughly twice as likely to achieve a successful exit (Azoulay et al.,
  2018)
- **Meta-analysis nuance**: a 2021 meta-analysis of 102 samples found a
  weak positive linear relationship overall, but the sign flips by outcome
  metric: negative for firm growth, positive for financial performance,
  firm size, and subjective success, and not significant for survival
  (Zhao et al., 2021)
- **The young-founder myth**: media narratives highlight outlier
  young-dropout founders of billion-dollar companies, but population data
  shows most founders are in their 30s and 40s, and the highest-growth
  firms are more often founded by middle-aged entrepreneurs, a case of
  selection bias and media attention to exceptional cases
- **Nonlinearity across the age range**: young founders may benefit from
  risk tolerance and creativity, middle-aged founders from experience,
  networks, and resources, and very late founders may face declining risk
  tolerance or rising opportunity costs

## Questions

1. Does founder age have a positive association with high-growth startup
   success only up to middle age (i.e., is the relationship nonlinear)?
2. Does prior industry experience mediate the effect of age on startup
   success, so that age is mostly a proxy for experience?
3. Do the direction and magnitude of the age effect depend on which
   success metric is used (growth, financial performance, survival)?

## Research Topics

- **Instrumental variables**: use market conditions at founding time
  (e.g., recession, technology boom) as an instrument for founder age
  cohort effects
- **Time-to-exit analysis**: estimate the causal effect of age on time to
  IPO or acquisition using survival analysis
- **Heterogeneous treatment effects**: investigate whether the causal
  effect of age differs across industries (e.g., SaaS vs. biotech vs.
  hardware)
- **Machine learning integration**: use causal forests or similar ML-based
  causal inference to detect complex non-linear relationships and
  interactions
- **Historical longitudinal study**: track the same cohorts of founders
  over 5-10 years to estimate long-term causal effects
- **Bias in VC funding**: analyze whether VC funding decisions themselves
  introduce age bias, and how that mediates the age-success relationship
- **Founder replacement**: compare startups whose founders changed over
  time versus those with stable teams, to isolate age effects from founder
  quality
- **Mechanism decomposition**: test which channel mediates the age effect:
  human capital (industry expertise, management experience), social
  capital (investor and customer networks), financial capital (savings,
  credit access), or credibility and reputation with investors and
  customers

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
  - Collect founder age, founding date, industry, team size, and funding
    amounts from Crunchbase and Y Combinator; merge and handle missing age
    data
  - Create the company outcome variable: success (acquired/IPO) vs.
    failure/inactive
  - Candidate datasets:
    - **Crunchbase** (Kaggle mirror): founder names, ages, founding dates,
      funding rounds, industry, employee count, company status
    - **Y Combinator company database** (ycombinator.com, Kaggle mirror):
      company names, founding year, founders, batch year, status, industry
    - **AngelList free API**: startup profiles, founder information,
      funding history, company status
    - **US Census Bureau / BLS APIs**: industry growth rates, regional
      economic indicators, employment trends by age group (macro
      confounders)

- Milestone 2: explore and build the causal DAG
  - Analyze the distribution of founder ages, industries, funding rounds,
    and outcomes; visualize the raw age-success correlation and segment by
    industry, geography, and team composition
  - Construct a causal DAG identifying confounders (industry, team size,
    market timing, geographic region, investor stage), validated against
    domain knowledge and prior research

- Milestone 3: estimate the causal effect
  - Estimate propensity scores for early-age versus later-age founders
    based on confounders, and match founders with similar scores but
    different ages
  - Estimate the average treatment effect (ATE) of founder age on success
    probability
  - Tools: `dowhy` for causal graphs and treatment effect estimation

- Milestone 4: validate and report
  - Test whether results hold across industry subgroups and team sizes;
    check for hidden bias using Rosenbaum bounds or similar sensitivity
    analysis
  - Compare propensity score matching against inverse probability
    weighting and stratification
  - Visualize treatment effect heterogeneity, and report the magnitude,
    practical significance, limitations, and policy implications

## References

- Saggese et al., _Causal Analysis of Agent Skill and Luck_,
  https://github.com/gpsaggese/gpsaggese.github.io/blob/master/papers/Causal_Analysis_of_Agent_Skill_And_Luck/Causal_Analysis_of_Agent_Skill_And_Luck.pdf
- Azoulay, Jones, Kim, and Miranda, _Age and High-Growth Entrepreneurship_,
  NBER Working Paper No. 24489 (2018),
  https://www.nber.org/system/files/working_papers/w24489/w24489.pdf
  - Analysis of 2.7 million US founders (2007-2014); prior industry
    experience strongly predicts startup success
- Zhao, Seibert, and Lumpkin, _The Relationship of Age and Entrepreneurial
  Success: A Meta-Analysis_, Journal of Business Venturing (2021),
  https://www.sciencedirect.com/science/article/abs/pii/S0883902619302691
  - Meta-analysis of 102 independent samples; effect direction depends on
    the success metric used
- _Causal Inference: The Mixtape_, https://mixtape.scunning.com/
- _Introduction to Causal Inference_ (Brady Neal),
  https://www.bradyneal.com/causal-inference-book
- DoWhy library documentation, https://py-why.github.io/dowhy/
- Crunchbase API documentation,
  https://www.crunchbase.com/docs/api/overview
