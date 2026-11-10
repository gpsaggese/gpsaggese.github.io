# Structural Break Challenge 2026: Multiple Testing Ideas

## Goal

- Link the ADIA Lab x Crunch Structural Break Challenge 2026
  - [ADIA Lab page](https://www.adialab.ae/adia-lab-x-crunch-the-structural-break-challenge-2026)
  - [CrunchDAO docs](https://docs.crunchdao.com/competitions/competitions/structural-break-real-time)
- Questions to investigate:
  - Are the leaderboard scores overfit?
  - Are the winners skilled or lucky?
  - How certain is the final ranking?
  - Which method families overfit more?
  - Does an ensemble of solutions improve the score?
- This file only collects ideas: nothing here is implemented yet

- Add probability (compute the probability), to have a better skill vs luck metric
  (log prob), compute that by voting in the ensemble

- Which time series have causal component?
  - Do they perform better?

## Competition Facts

- Source: the CrunchDAO docs page, read in full
  - The ADIA Lab page does not name the metric, the prize split, or the
    leaderboard rules
- Task: online detection of a permanent structural break in a univariate series
  - Historical segment: 1,000 to 5,000 break-free observations, given upfront
  - Online segment: 10 to 1,000 observations, revealed one at a time
  - After each observation, output a score in $[0, 1]$: the confidence that a
    break already happened
  - No look-ahead: each score is yielded before the next point arrives
- Data:
  - At most one break per series
  - Each series has probability 0.5 of a break at an unknown online position
  - Training covers breaks in mean, variance, distribution shape, and
    dependence structure
  - Each series is z-scored with its historical segment only
  - Training labels give `tau`, the 0-indexed break position, or `None`
- Metric: Time-Stratified AUC (TS-AUC)
  - At each step $t$, compute the AUC over the series still active
  - A series is positive if its break already happened by $t$, else negative
  - The final score averages the per-step AUCs with weight
    $w(t) = n_{pos}(t) \cdot n_{neg}(t)$
  - 0.5 is random, 1.0 is perfect
  - There is no separate latency term
- Splits:

  | Split               | Where        | Series | Labels     |
  | :------------------ | :----------- | :----- | :--------- |
  | Public training     | Local, Cloud | 10,000 | Yes        |
  | Public reduced test | Local        | 100    | Yes, local |
  | Public test         | Cloud        | 10,000 | No         |
  | Private test        | Cloud        | 10,000 | No         |

- Submission:
  - Code competition with `train()` and `infer()`
  - Output must be deterministic: a re-run on 10% of the data must match within
    1e-8
  - Execution limit of 15 hours per week
  - External data is allowed with a valid license
  - Yahoo Finance and TabPFN are not allowed
- Leaderboard and prizes:
  - Public leaderboard is frozen
  - A hidden leaderboard shows the public top 10 for comparison
  - Only the team leader is ranked, and rewards are split among members
  - Prizes in USDC: 1st $40K, 2nd $20K, 3rd $10K, 4th to 7th $5K each, 8th to 9th
    $3.5K each, 10th $3K
- Timeline:
  - Submission deadline: October 1, 2026
  - Winners announced at the ADIA Lab 2026 Symposium, October 26 to 28
  - Late submissions are open again but are not run on out-of-sample data
- Methods suggested by the docs:
  - Statistical tests (t-test, KS, CUSUM)
  - Online change-point detection
  - Incremental feature extraction with a classifier
  - Bayesian sequential models
  - Deep learning
  - Time-series foundation models
- No baseline is provided

## Statistical Toolbox

### Mapping from Backtests to Submissions

- The competition has the same structure as a backtest search

  | Finance                    | Competition                              |
  | :------------------------- | :--------------------------------------- |
  | Backtested strategy        | Submission or hyperparameter config      |
  | Sharpe ratio               | TS-AUC                                   |
  | Number of trials $N$       | Submissions across all teams             |
  | In-sample period           | Public leaderboard                       |
  | Out-of-sample period       | Private leaderboard                      |
  | Selected best strategy     | Winner                                   |

- The compute quota (15 hours per week) caps the trials of each team
  - A team with more submissions has more draws from the same noise

### Tools from Lopez de Prado and Coauthors

- **False Strategy Theorem**: the expected maximum of $N$ independent trials with
  zero true skill grows with $N$ even when every trial is noise
  - In z-score units, with $\gamma \approx 0.5772$ (Euler-Mascheroni constant):
    $$
    E[\max_N z] \approx (1 - \gamma) \Phi^{-1}(1 - 1/N)
    + \gamma \Phi^{-1}(1 - 1/(N e))
    $$
  - E.g., $N = 100$ gives about 2.5
  - E.g., $N = 1000$ gives about 3.3
  - TODO(gp): Check these two values against the paper: computed with the
    formula above, not taken from the source
- **Deflated Sharpe Ratio (DSR)**: tests the best Sharpe ratio against the
  expected maximum under the null
  - Analog here: a deflated AUC, using the standard error of the AUC
- **Probability of Backtest Overfitting (PBO)**: uses combinatorially symmetric
  cross-validation (CSCV)
  - Split the performance matrix (trials by blocks) into in-sample and
    out-of-sample halves
  - Count how often the best in-sample trial ranks below the median out-of-sample
- **Clustering of trials**: Lopez de Prado and Lewis cluster trials to estimate
  the number of independent ones
  - Many submissions are variants of CUSUM or gradient boosting
  - So the effective number of trials is much smaller than the nominal one
- **Minimum backtest length**: how much data separates a skilled strategy from
  the best of $N$ random ones
  - Analog here: how many test series separate the top ranks

### Precision of the Score

- The AUC standard error depends on the number of positives and negatives
  - Hanley-McNeil approximation, 5,000 positives and 5,000 negatives: about
    0.0055 for AUC between 0.5 and 0.6
  - With 500 per class: about 0.018
- Consequence: score gaps below about 0.005 between top teams are likely noise
  - Paired differences have a smaller standard error, since models share series
- TODO(gp): Check the standard errors: Hanley-McNeil formula recalled from
  memory, and TS-AUC pools many steps so its real variance differs

## Ideas from the Original Brainstorm

### 1. Multiple Hypothesis Testing to Detect Overfitting

- Question: how much of the top leaderboard scores is selection bias?
- Bootstrapping results:
  - Resample whole series, not steps, since steps in one series are correlated
  - Recompute TS-AUC on each resample
  - Get a confidence interval for each score and for each score difference
- Deflated AUC approach:
  - Estimate the number of trials $N$: submissions, or the effective number
    after clustering
  - Convert the best AUC to a z-score using its standard error
  - Compare with $E[\max_N z]$ from the False Strategy Theorem
  - Report a sensitivity table for $N$ from 10 to 10,000, since $N$ is uncertain
- Probability of backtest overfitting:
  - Rows are configs or teams
  - Columns are blocks: random series folds, or generator regimes
  - Regimes: break type, series length, early or late break
- Public vs private shift:
  - Compute Spearman and Kendall correlation of the two rankings
  - Regress the private score on the public score: the slope measures shrinkage
  - The hidden leaderboard with the public top 10 makes this a natural test
- Data needed: public and private leaderboard scores
  - TODO(gp): Check if these scores can be collected from the Crunch Hub after
    the final evaluation

### 2. Skill vs Luck for the Winners

- Question: would the same winner appear if the test series were drawn again?
- Null simulation:
  - Simulate $N$ detectors with zero or weak true skill
  - Score them on the public training set with TS-AUC
  - Get the distribution of the best score and of the winner margin
  - Compare with the real top scores
- Probability that the winner is truly best:
  - Monte Carlo with the AUC standard error and an error correlation $\rho$
    between teams
  - Report the result for $\rho$ from 0 to 0.9
- Trials per team:
  - Count the submissions of each team, when visible
  - Test if teams with more submissions score higher on public and drop more on
    private

### 3. Probabilistic Ranking

- Question: what is the probability of each rank, instead of a single rank?
- Bootstrap rank distribution:
  - Estimate $P(\text{rank} = k)$, $P(\text{top } 3)$, and $P(\text{top } 10)$ per
    team
  - Use rank confidence sets (Mogstad, Romano, Shaikh, and Wilhelm)
- Bayesian option:
  - Plackett-Luce model on per-fold results
  - Hierarchical model with a team effect
- Prize view:
  - Compute the expected prize of each team under rank uncertainty
  - Propose a payout rule that is robust to luck, e.g., shared prizes among teams
    whose scores are not statistically distinguishable
- Limit: the per-series predictions of all teams are needed
  - Only the organizers have them
  - Without them, use the parametric version with the standard error and $\rho$

### 4. Performance and Overfitting by Method

- Question: do some method families overfit more than others?
- Families: statistical tests, online change-point detection, features plus a
  classifier, Bayesian sequential, deep learning, foundation models
- Metrics per family:
  - Public-to-private score gap
  - Number of submissions
  - Score dispersion
  - Rank stability
- Hypothesis: high-capacity families show larger gaps, simple tests are stable
  but have a lower ceiling
- Own experiment:
  - Implement one detector per family on the 10,000 labeled training series
  - Use nested cross-validation: select hyperparameters on one fold, score on
    another
  - Compare the in-sample and out-of-sample TS-AUC per family
- Data needed: method labels for real submissions
  - TODO(gp): Check if the Crunch forum or public notebooks give the method
    of each team

### 5. Would an Ensemble of the Solutions Improve?

- Question: does combining detectors beat the best single one?
- Calibration:
  - TS-AUC compares scores across series at each step
  - So scores must be comparable across series
  - Combine with rank averaging or logit averaging per step
- Diversity:
  - Measure the correlation of score vectors between members
  - Ensembling helps only when members make different errors
- Selection bias:
  - An ensemble of leaderboard-selected members inherits the selection bias
  - Select members on fold A and evaluate the ensemble on fold B
- Limit: ensembling the real winners is possible only for the organizers
  - Own version: ensemble of one detector per method family (idea 4)

## New Ideas

### 6. Power Analysis for the Ranking

- Question: do 10,000 test series suffice to rank the top 10?
- Method:
  - Fix the effect: an AUC gap of 0.002 between two teams
  - Compute the number of series needed to detect it at the given $N$
  - Repeat for gaps of 0.001 to 0.01
- Output: a minimum-test-size curve, the analog of the minimum backtest length
- Use: tells the organizers how many series a future edition needs

### 7. Anytime-Valid Scores with E-Processes

- Question: can a sequential test give a better calibrated score?
- Idea:
  - Each step of the online segment is a hypothesis test, so monitoring is
    itself a multiple testing problem across time
  - A test martingale (e-process) for "no break" is valid at any stopping time
    (Ville's inequality)
  - Score = monotone function of the e-process
- Why it fits TS-AUC:
  - The e-value has the same meaning in every series
  - So the scores are comparable across series
- Baselines: CUSUM and Bayesian online change-point detection
- Open point: which null model for the e-process, given the z-scored history

### 8. Monotone Scores

- Question: does enforcing a non-decreasing score help?
- Reason: the break is permanent, so the confidence should not fall
- Variants:
  - Raw score
  - Running maximum
  - Exponentially smoothed score
- Risk: one early false alarm keeps a series high for the rest of the stream
- Test: compare the three on the training set with the cluster bootstrap

### 9. Meta-Labeling and Structural Break Features

- Question: can a second model filter the alarms of a simple detector?
- Meta-labeling (from Advances in Financial Machine Learning):
  - Primary model: CUSUM or Bayesian online change-point detection raises alarms
  - Secondary model: a classifier estimates the probability that an alarm is real
  - Output score = the secondary probability
- Features from the structural break chapter of the same book:
  - CUSUM tests
  - Supremum ADF type tests
- TODO(gp): Check the chapter numbers and the exact tests in the book: recalled
  from memory

### 10. Generator Audit and Leak Checks

- Question: do winners exploit artifacts of the data generator?
- Own generator:
  - Simulate series with the four break types
  - Vary break size, break position, series length, and noise type
  - Check if the ranking of methods stays stable when parameters shift
- Leak checks on the public training set:
  - Does the online length $T$ predict the presence of a break?
  - Does the break position correlate with $T$?
  - Are there near-duplicate series, within or across splits?
  - Does the 0.5 break prior leave an exploitable pattern in the scores?
- Link to PBO (idea 1): the generator regimes are the blocks of the matrix

### 11. Cross-Edition Stability

- Question: do the 2025 winners also rank high in 2026?
- 2025 edition: see `README.md` in this directory for the task and the links
  - TODO(gp): Check how the 2025 task and metric differ from the 2026 ones
- Method: rank correlation of teams present in both editions
- Data needed: the team lists of both leaderboards
  - TODO(gp): Check if both final leaderboards are public
- The docs say the previous dataset targets a different problem
  - So only team rankings are comparable, not scores

### 12. Submissions as Data for a Backtest Overfitting Detector

- Question: can we predict the private-rank drop from public information?
- Link: `README.md` in this directory, idea 2 (Backtest Overfitting Detector)
- Features per team:
  - Public score
  - Number of submissions
  - Method family
  - Score variance across submissions
- Target: private rank minus public rank, or the private score minus the public
  one
- Baseline to beat: deflated AUC from idea 1

## Suggested Order

- Cheap first, each step needs only the public training set:
  1. Local TS-AUC evaluator with the cluster bootstrap (idea 1)
  2. Null simulation of the best score and of the winner margin (idea 2)
  3. Power analysis (idea 6)
  4. Method family comparison with nested cross-validation (ideas 4 and 5)
  5. Generator audit and PBO over regimes (ideas 10 and 1)
- Needs leaderboard data, so after the final results:
  1. Public-private shift (idea 1)
  2. Rank probabilities (idea 3)
  3. Cross-edition stability and the overfitting detector (ideas 11 and 12)
- Needs new modeling work:
  1. E-process scores (idea 7)
  2. Monotone scores (idea 8)
  3. Meta-labeling (idea 9)

## References

### Verified Against Crossref

- Bailey, Borwein, Lopez de Prado, Zhu (2014): "Pseudo-Mathematics and Financial
  Charlatanism: The Effects of Backtest Overfitting on Out-of-Sample
  Performance", Notices of the AMS 61(5), 458, doi 10.1090/noti1105
- Bailey, Borwein, Lopez de Prado, Zhu: "The Probability of Backtest
  Overfitting", Journal of Computational Finance, doi 10.21314/jcf.2016.322
  - TODO(gp): Check the volume year: Crossref gives 2016, which is the online
    year
- Bailey and Lopez de Prado (2014): "The Deflated Sharpe Ratio: Correcting for
  Selection Bias, Backtest Overfitting and Non-Normality", Journal of Portfolio
  Management 40(5), 94-107, doi 10.3905/jpm.2014.40.5.094
- Lopez de Prado and Bailey (2021): "The False Strategy Theorem: A Financial
  Application of Experimental Mathematics", American Mathematical Monthly 128(9),
  825-831, doi 10.1080/00029890.2021.1965068
- Lopez de Prado (2019): "A Data Science Solution to the Multiple-Testing Crisis
  in Financial Research", Journal of Financial Data Science 1(1), 99-110,
  doi 10.3905/jfds.2019.1.099
- Lopez de Prado and Lewis (2018): "Detection of False Investment Strategies
  Using Unsupervised Learning Methods", SSRN, doi 10.2139/ssrn.3167017
- Harvey, Liu, Zhu (2016): "... and the Cross-Section of Expected Returns",
  Review of Financial Studies 29(1), 5-68, doi 10.1093/rfs/hhv059
- Harvey and Liu (2015): "Backtesting", Journal of Portfolio Management 42(1),
  13-28, doi 10.3905/jpm.2015.42.1.013
- White (2000): "A Reality Check for Data Snooping", Econometrica 68(5),
  1097-1126, doi 10.1111/1468-0262.00152
- Hansen (2005): "A Test for Superior Predictive Ability", Journal of Business
  and Economic Statistics 23(4), 365-380, doi 10.1198/073500105000000063
- Romano and Wolf (2005): "Stepwise Multiple Testing as Formalized Data
  Snooping", Econometrica 73(4), 1237-1282, doi 10.1111/j.1468-0262.2005.00615.x
- Hansen, Lunde, Nason (2011): "The Model Confidence Set", Econometrica 79(2),
  453-497, doi 10.3982/ecta5771
- Benjamini and Hochberg (1995): "Controlling the False Discovery Rate", Journal
  of the Royal Statistical Society B 57(1), 289-300,
  doi 10.1111/j.2517-6161.1995.tb02031.x
- Dwork, Feldman, Hardt, Pitassi, Reingold, Roth (2015): "The Reusable Holdout:
  Preserving Validity in Adaptive Data Analysis", Science 349(6248), 636-638,
  doi 10.1126/science.aaa9375
- Mogstad, Romano, Shaikh, Wilhelm (2020): "Inference for Ranks with Applications
  to Mobility across Neighborhoods and Academic Achievement across Countries",
  NBER Working Paper 26883, doi 10.3386/w26883
- Ramdas, Grunwald, Vovk, Shafer (2023): "Game-Theoretic Statistics and Safe
  Anytime-Valid Inference", Statistical Science 38(4), 576-601,
  doi 10.1214/23-sts894
- Page (1954): "Continuous Inspection Schemes", Biometrika 41(1-2), 100-115,
  doi 10.1093/biomet/41.1-2.100
- Killick, Fearnhead, Eckley (2012): "Optimal Detection of Changepoints With a
  Linear Computational Cost", JASA 107(500), 1590-1598,
  doi 10.1080/01621459.2012.737745
- Truong, Oudre, Vayatis (2020): "Selective Review of Offline Change Point
  Detection Methods", Signal Processing 167, 107299,
  doi 10.1016/j.sigpro.2019.107299

### Not Verified

- Lopez de Prado (2018): Advances in Financial Machine Learning, Wiley
  - TODO(gp): Check the chapter numbers for meta-labeling and structural break
    tests
- Lopez de Prado and Porcu (2026): "The Deflated Sharpe Ratio: A Unified
  Framework ...", Journal of Portfolio Management, doi 10.3905/jpm.2026.070
  - Seen only as a title in Crossref: read before citing
- Blum and Hardt (2015): "The Ladder: A Reliable Leaderboard for Machine Learning
  Competitions", ICML
  - Relevant to leaderboard overfitting: not found in Crossref
- Roelofs et al. (2019): "A Meta-Analysis of Overfitting in Machine Learning",
  NeurIPS
  - Relevant to leaderboard overfitting: not found in Crossref
- Adams and MacKay (2007): "Bayesian Online Changepoint Detection", arXiv
- Hanley and McNeil (1982): AUC standard error, Radiology

## Notes

- `README.md` in this directory lists the competition dates as May 4 to
  September 17, 2026
  - The CrunchDAO docs give May 6 start and October 1 submission deadline
  - TODO(gp): Check and fix the dates in `README.md`
- The two numeric estimates in "Statistical Toolbox" are computed, not sourced
