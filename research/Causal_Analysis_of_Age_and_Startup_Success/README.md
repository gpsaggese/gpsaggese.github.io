# A Causal Analysis of Age and Startup Success

## 1. Project Overview

This project investigates whether **founder age has a causal effect on
startup success**, or whether the widely observed correlation between age
and success is explained by confounding factors — industry experience,
network/financial capital, and team composition.

Popular narratives about entrepreneurship celebrate very young founders, but
large-scale empirical research (Azoulay et al. 2018; Zhao et al. 2021)
suggests a more nuanced picture: the average successful founder is
middle-aged, and much of the apparent "age advantage" may be a proxy for
accumulated human, social, and financial capital rather than an independent
causal effect of age itself.

The goal is not simply to measure a correlation, but to apply causal
inference methods — regression adjustment, propensity score matching, DAG-based
confounder identification, and sensitivity analysis — to estimate how much of
the age–success relationship is likely causal, and how much is explained
away once relevant confounders are accounted for.

---

## 2. Background and Motivating Literature

### 2.1 Azoulay, Jones, Kim & Miranda (2020) — *Age and High-Growth
Entrepreneurship*, AER: Insights 2(1): 65–82
Analyzed 2.7 million U.S. founders (2007–2014) using linked IRS K-1, W-2,
Census LBD, and USPTO administrative microdata (peer-reviewed published
version; supersedes the earlier NBER working paper).

- Average founder age: 41.9 years
- Average age among the top 0.1% fastest-growing firms: 45.0 years
- A 50-year-old founder is ~1.8x more likely than a 30-year-old to build a
  top-growth firm, and ~2x more likely to achieve a successful exit
  (acquisition or IPO)
- Founders below age 25 succeed rarely; success probability rises sharply
  at 25, plateaus through the early 30s, then surges again from the
  mid-to-late 30s through the mid-50s
- **Founder identification methodology** — relevant to our own definitional
  choices: for S-corps/partnerships they use IRS Schedule K-1 ownership
  records; for C-corps (no K-1 data available) they proxy founders as the
  three highest-paid W-2 earners in the firm's first year, following Kerr
  & Kerr (2017)
- **Prior industry experience effect (their Table 3)** — founders with 3+
  years of experience in the *same* industry as their startup reach the
  top-0.1%-growth tier at 0.22–0.26% (varying by NAICS granularity), vs.
  0.11% for founders with no industry experience — roughly **2x higher**
  success rate. This is a concrete benchmark for our own H3 mediation test.
- **VC behavior puzzle (their §III.F)** — venture capitalists systematically
  fund *younger* founders despite younger founders having substantially
  lower success rates in this data. Directly relevant to our "Bias in VC
  Funding" stretch idea (Section 9).
- Individual-level exceptions (Jobs, Gates, Bezos, Musk) do not contradict
  the population-level middle-age advantage — their analysis of forward
  stock-price multiples shows none of these founders' companies "peaked"
  when the founder was very young (peaks at ages 36–52 across the four).

### 2.2 Zhao, Seibert & Lumpkin (2021) — Meta-analysis, Journal of Business Venturing
Synthesized 102 independent samples.

- Weak positive linear relationship between age and success overall
- Evidence of a U-shaped (nonlinear) relationship
- Effect direction depends heavily on the success metric used:

| Success Measure       | Relationship with Age |
|------------------------|------------------------|
| Firm Growth             | Negative               |
| Financial Performance   | Positive               |
| Firm Size               | Positive               |
| Subjective Success      | Positive               |
| Survival                 | No significant effect  |

### 2.3 Kanama, Ito, Muranaka & Watanabe (2025) — *Role of the Grey
Entrepreneur in Startups*, Journal of Economic Structures 14:8
Survey of 126 Japanese startups (up to 4 founders each), examining **founder
team age composition**, not just a single founder's age.

- **H1 confirmed**: more founders → higher gross sales (p<0.001)
- **Inverted U-shaped relationship** between founder age *diversity* and
  gross sales: some age spread among co-founders helps, but too much hurts.
  Sales peak when the team spans 2–3 "generations" (age-diversity score of
  2–3 on their 1–4 scale), consistent with a gap of roughly 20–30 years
  being optimal
- Teams with a **core of founders in their 30s**, supplemented by older
  co-founders, showed the strongest sales performance
- Explicitly frames the mechanism as a tension between resource dependence
  theory (diversity → more resources/networks → positive) and social
  identity theory (diversity → in-group friction → negative), with the
  inverted-U as the empirical reconciliation
- **Limitation, noted by the authors themselves**: proprietary, non-public
  survey data (n=126, single-country, industry-skewed toward B2B/healthcare);
  not directly replicable, but the *hypothesis* — team size/composition,
  not just individual age — is testable on our own data (see H5)

### 2.4 Takeaway
All three studies converge on the same conclusion: **age is likely a proxy
for other forms of capital** (human, social, financial) rather than a direct
causal driver, and **team composition** (size, age diversity) may matter as
much as, or more than, any single founder's age. This project tests these
hypotheses directly using causal inference methods rather than relying on
correlational summaries.

---

## 3. Problem Statement

> Do older founders succeed *because* they are older, or does age simply
> travel together with other things that really matter — industry know-how,
> bigger networks, more savings — which are the true drivers of success?

The core methodological challenge is **confounding**: age correlates with
industry experience, financial resources, and social capital, all of which
independently affect success. A naive correlation between age and success
conflates the effect of age itself with the effects of these accumulated
resources. Causal inference methods are used to strip away as much of this
confounding as the data allows, and to be explicit about what remains
unaddressed.

---

## 4. Causal Design

### 4.1 Treatment
**Founder age** — modeled as continuous, with a quadratic term (age²) to
capture the U-shaped relationship suggested by prior literature. Also
binarized (e.g. founded before 35 vs. founded 45+) as a robustness check for
matching-based methods.

### 4.2 Outcome
**Startup success** — primary definition: acquisition or IPO (binary).
Growth and funding are treated as secondary outcomes in separate models
rather than combined into a single composite measure.

### 4.3 Confounders (pre-treatment — control for these directly)
Variables that existed independently of the founder's age and could bias
both age and success:
- Industry
- Geography
- Founding year
- Team size
- Education

### 4.4 Mediators (post-treatment — analyzed separately, not controlled for directly)
Variables that are themselves *caused by* age and that go on to affect
success. Controlling for these in the same model as age would introduce
post-treatment bias and understate the true effect of age:
- Prior industry experience
- Network / financial capital

Testing how much the age coefficient shrinks when mediators are added is
itself a key finding (see Hypothesis H3 below) — it quantifies how much of
age's effect runs *through* these channels versus acting independently.

### 4.5 Unobserved Confounder
**Founder ability** (skill, judgment, drive) — not observable in any
available dataset. This cannot be controlled for directly. It is instead
addressed explicitly via sensitivity analysis (Rosenbaum bounds), which
quantifies how strong an unmeasured confounder would need to be to overturn
the estimated effect.

### 4.6 Causal DAG (conceptual structure)

```
                 Observed confounders
        (industry, geography, founding year,
              team size, education)
              /                      \
             v                        v
     Founder age  ── direct effect? ──>  Startup success
        |    \                          ^    ^
        |     \                        /      \
        |      v                     /         \
        |   Prior industry ---------            |
        |   experience (mediator)                |
        |                                        |
        v                                        |
   Network / financial ------------------------>-
   capital (mediator)

   Founder ability (UNOBSERVED) --(dashed)--> age AND success
   [handled via sensitivity analysis, not direct control]
```

Confounders point into both age and success and are controlled for directly.
Mediators are caused by age and go on to affect success — analyzed
separately to avoid post-treatment bias. Founder ability is an unobserved
common cause of both age and success, and is the central limitation of this
analysis — addressed via sensitivity analysis rather than ignored.

---

## 5. Research Hypotheses

- **H1** — Founder age has a positive association with high-growth startup
  success up to middle age.
- **H2** — The relationship between founder age and startup success is
  nonlinear (U-shaped).
- **H3** — Prior industry experience mediates the effect of age on startup
  success (i.e., the age coefficient shrinks substantially once experience
  is added to the model).
- **H4** — The direction and magnitude of the age effect depends on which
  success metric is used.
- **H5** — Founder team size (and, if age data becomes available, age
  diversity within the founding team) independently predicts success,
  following Kanama et al. (2025). Testable now using `founder_count` from
  the matched YC subset, without requiring individual founder ages.

---

## 6. Data Sources

| Source | Contains | Access | Role |
|---|---|---|---|
| Crunchbase (Kaggle export) | Founder names, funding rounds, industry, employee count, company status | Free Kaggle download | Core dataset |
| Y Combinator company dataset | Batch year, status, industry, founders | Kaggle / public scrape | Core dataset (cleaner ground truth on outcomes) |
| LinkedIn / public bios | Founder education end-date, used as an age proxy | Manual / semi-automated | Age proxy — hardest step, see limitations |
| Census Bureau / BLS APIs | Industry growth rates, regional economic indicators, employment trends | Free public API | Macro confounders |

**Known limitation:** neither Crunchbase nor the YC dataset provides
founder birthdate directly. Founder age will be estimated as a proxy
(founding year − estimated graduation year + ~22), which introduces noise
and missingness that will be documented explicitly as a limitation rather
than treated as ground truth.

**This is not unique to our project.** Every published study reviewed here
faces the same constraint: Azoulay et al. use restricted-access Census/IRS
microdata unavailable outside government research partnerships; Roche,
Conti & Rothaermel (2020, *Research Policy* 49(10):104062) build on
Crunchbase but supplement it with paid sources (VentureXpert, USPTO,
Scopus, Web of Science) plus manual LinkedIn lookups, and never released
their merged dataset; Ali Tamaseb's *Super Founders* manually collected
founder-level data for ~500 companies over 4 years. **Crunchbase is the
standard backbone dataset across this literature** — our project uses the
same base source as the published research it builds on; the gap is in the
proprietary/manual enrichment layers, not the core data.

**Given the time cost of manual age collection** (documented above), the
primary analysis proceeds on the full merged dataset using the confounders
that ARE available (industry, team size, funding, founding year, region),
with age treated as a documented gap rather than blocking the project. A
small hand-collected sample (30-50 founders) MAY be used to demonstrate the
age-proxy method, but is not required for the core causal analysis.

---

## 7. Procedure / Task Breakdown

### Step 1 — Data Collection & Integration
- Pull YC company list and Crunchbase founder/company data
- Estimate founder age via education-date proxy
- Merge datasets via fuzzy name matching (company + founder)
- Attach Census/BLS macro confounders by founding year × industry × region
- Define binary outcome variable (acquired/IPO = 1, else 0), with a
  documented cutoff date

### Step 2 — Exploratory Data Analysis (EDA)
- Distribution of founder ages, industries, funding rounds, outcomes
- Visualize raw age–success correlation and identify obvious confounders
- Segment by industry, geography, team composition for heterogeneity

### Step 3 — Confounder Identification & DAG Construction
- Formalize the DAG (Section 4.6) in code (e.g. via DoWhy)
- Validate variable classification (confounder vs. mediator) against domain
  knowledge and the literature in Section 2
- Document all causal assumptions explicitly

### Step 4 — Primary Modeling
- Regression: `success ~ age + age² + industry_FE + year_FE + team_size + geography + education`
- Mediation model: adds prior_industry_experience + network/financial
  capital, to test H3 (how much the age coefficient shrinks)

### Step 5 — Propensity Score Matching (robustness check)
- Binarize age (e.g. <35 vs. 45+)
- Estimate propensity scores on confounders only (not mediators)
- Match via nearest-neighbor within caliper
- Estimate Average Treatment Effect (ATE) on the matched sample
- Compare direction/magnitude against the regression estimate

### Step 6 — Sensitivity Analysis & Robustness Checks
- Rosenbaum bounds: quantify how strong an unobserved confounder (e.g.
  founder ability) would need to be to overturn the result
- Test whether results hold across industry subgroups and team sizes
- Compare PSM against alternative methods (inverse probability weighting,
  stratification) as available

### Step 7 — Interpretation & Reporting
- Visualize treatment effect heterogeneity across industries/demographics
- Report: does age have an independent causal effect, or is it explained
  away by confounders/mediators?
- Explicitly address magnitude, practical significance, limitations
  (especially the unobserved-ability problem), and policy implications

---

## 8. Repository Structure

```
research/Causal_Analysis_of_Age_and_Startup_Success/
├── README.md                  This file
├── requirements.txt
├── data/
│   ├── raw/                   Crunchbase, YC, Census pulls (untouched)
│   ├── interim/                Cleaned, deduplicated, name-matched
│   └── processed/              Final merged founder-level dataset
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_dag_and_model_spec.ipynb
│   ├── 03_regression.ipynb
│   ├── 04_psm.ipynb
│   └── 05_sensitivity.ipynb
└── src/
    ├── data_collection.py      Data pulling and merging
    ├── matching.py               Propensity score matching
    ├── models.py                  Regression specs (primary + mediation)
    └── sensitivity.py             Rosenbaum bounds
```

---

## 9. Bonus / Stretch Ideas (not in core scope)

- **Instrumental variables**: market conditions at founding (e.g. recession,
  tech boom) as an instrument for age-cohort effects — hard to defend on
  exclusion-restriction grounds, treated as a stretch goal only
- **Time-to-exit survival analysis**: causal effect of age on time to
  IPO/acquisition
- **Heterogeneous treatment effects**: causal forests to detect nonlinear/
  interaction effects across industries
- **VC funding bias**: whether investor decisions themselves introduce age
  bias, mediating the age–success relationship
- **Founder replacement analysis**: startups with founder turnover vs.
  stable teams, to help isolate age from founder quality
- **Founder team composition (H5)**: following Kanama et al. (2025), test
  whether founder count and (if age becomes available) age diversity within
  the team independently predict success, beyond any single founder's age

---

## 10. Known Limitations

- **Founder ability is unobserved** and cannot be directly controlled for;
  this is the central threat to causal identification and is addressed via
  sensitivity analysis rather than ignored.
- **Founder age is a derived proxy** (via education dates), not ground
  truth, and will carry measurement error and missingness.
- **Name-matching across datasets** (Crunchbase ↔ YC ↔ LinkedIn) is
  imperfect and may introduce merge errors.
- **Success is operationalized narrowly** (acquisition/IPO) as the primary
  outcome; other reasonable definitions of "success" may yield different
  conclusions (see H4).

---

## References

- Azoulay, P., Jones, B. F., Kim, J. D., & Miranda, J. (2020). *Age and
  High-Growth Entrepreneurship.* AER: Insights, 2(1): 65–82.
  https://doi.org/10.1257/aeri.20180582
- Zhao, H., Seibert, S. E., & Lumpkin, G. T. (2021). *The Relationship of
  Age and Entrepreneurial Success: A Meta-Analysis.* Journal of Business
  Venturing. https://www.sciencedirect.com/science/article/abs/pii/S0883902619302691
- Kanama, D., Ito, S., Muranaka, S., & Watanabe, T. (2025). *Role of the
  Grey Entrepreneur in Startups: An Empirical Study of the Impact of Age
  Diversity on Innovation Performance.* Journal of Economic Structures,
  14:8. https://doi.org/10.1186/s40008-025-00352-7
- Roche, M. P., Conti, A., & Rothaermel, F. T. (2020). *Different Founders,
  Different Venture Outcomes: A Comparative Analysis of Academic and
  Non-Academic Startups.* Research Policy, 49(10): 104062.
- Causal Inference: The Mixtape — https://mixtape.scunning.com/
- Introduction to Causal Inference (Brady Neal) — https://www.bradyneal.com/causal-inference-book
- DoWhy Library Documentation — https://py-why.github.io/dowhy/
