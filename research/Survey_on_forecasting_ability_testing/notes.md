# White's Reality Check and Beyond: Intuition and State of the Art

## 1. The intuition of White's Reality Check

White's Reality Check (White, 2000) answers one question: **if you try many models or
trading rules on the same data, is the best one actually good, or did it just get
lucky?**

### The problem: data snooping

Suppose you backtest 1,000 trading rules against a benchmark. Even if none of them
has any real edge, some will beat the benchmark in your sample by chance. The best of
1,000 will usually look impressive, and a standard t-test on it can be highly
"significant." That test assumes you picked the rule before looking at the data. You
didn't: you picked it *because* it won. It is like asking 1,000 people to flip a coin
ten times. Someone will probably get nine or ten heads, and that doesn't make them
skilled.

### The idea

The right benchmark for the best model is **how good the best of k models looks when
none of them has any real skill**. Comparing it to how good a single model looks by
chance is the wrong comparison.

1. **Relative performance.** For each model k and period t, compute its
outperformance over the benchmark, f_{k,t}, using return or forecast-loss
differences.
2. **Test statistic.** Take the best average outperformance: V = max_k sqrt(n) *
fbar_k.
3. **Null hypothesis.** The best model is no better than the benchmark: H0: max_k
E[f_k] <= 0.
4. **Bootstrap the maximum.** Resample the time series with the stationary bootstrap
(Politis and Romano, 1994). Recenter each model at its sample mean to impose "no
skill," and recompute the maximum in every resample.
5. **p-value.** The p-value is the share of bootstrap maxima that exceed the maximum
actually observed.

### Why it works

- **It accounts for the whole search.** Taking the maximum inside every bootstrap
  draw builds the act of picking the winner into the null distribution.
- **It respects correlation between models.** All models are resampled together, so
  near-duplicate rules are not treated as independent chances. This makes it far less
  conservative than a Bonferroni correction.
- **It respects time dependence.** Block resampling keeps autocorrelation and
  volatility clustering intact.

Sullivan, Timmermann and White (1999) applied it to thousands of technical trading
rules on about 100 years of Dow Jones data.

### Limitations

- **It is conservative when many bad models are included.** Recentering every model
  to a zero mean is the "least favorable" configuration, so clearly inferior models
  still inflate the bootstrap maximum.
- **It only says *whether* some model beats the benchmark,** not *which* ones do.
- **It needs the full set of models tried,** including discarded ones and those tried
  by earlier researchers on the same data.

**In short:** is your best result better than the best result pure luck would have
produced, given everything you tried?

## 2. State of the art

There isn't one single successor. The literature has split into several lines of
work, each fixing a different weakness.

### 2.1 Sharper tests of "is the best model better?"

- **Superior Predictive Ability (SPA)** (Hansen, 2005) is the standard replacement in
  applied work. It studentizes each model's statistic and leaves clearly inferior
  models out of the null distribution, which removes much of White's
  conservativeness.
- **Conditional SPA** (Li, Liao and Quaedvlieg, 2022) asks whether any model is
  better in some states of the world, for example high volatility, rather than only
  on average.
- **Multi-horizon SPA** (Quaedvlieg, 2021) tests the whole forecast path at once
  instead of each horizon separately, with uniform and average versions of superior
  ability.

### 2.2 Which models are good?

- **StepM** (Romano and Wolf, 2005) and **stepwise SPA** (Hsu, Hsu and Kuan, 2010)
  reject models one at a time while controlling the familywise error rate (FWER).
- **Model Confidence Set** (Hansen, Lunde and Nason, 2011) keeps discarding the worst
  model until the remaining set contains the best model with a given confidence
  level. It is now standard in forecast comparison.

### 2.3 From FWER to the false discovery rate (FDR)

With thousands of strategies, funds or factors, guarding against even one false
positive is too strict. FDR methods (Benjamini and Hochberg, 1995) instead control
the share of discoveries that are false.

- Early applications include mutual funds (Barras, Scaillet and Wermers, 2010) and
  technical trading rules (Bajgrowicz and Scaillet, 2012).
- **"Thousands of Alpha Tests"** (Giglio, Liao and Xiu, 2021) is the current
  frontier. It controls the FDR when the number of tests is large relative to the
  sample size, and stays robust to omitted factors and missing data.
- Qu, Timmermann and Zhu (2020) build FDR-controlling tests for panels of economic
  forecasters.

### 2.4 Type II errors and Bayesian approaches

- **Haircut Sharpe ratios and profit hurdles** (Harvey and Liu, 2015; Harvey, Liu and
  Zhu, 2016) adjust reported performance for the number of tests.
- **"False (and Missed) Discoveries"** (Harvey and Liu, 2020) chooses cutoffs by
  trading off false positives against missing managers who are genuinely skilled.
- **Bayesian / empirical-Bayes shrinkage** (Jensen, Kelly and Pedersen, 2023) models
  the whole set of factors jointly and finds that most of them replicate.

### 2.5 Practitioner tools

- **The deflated Sharpe ratio** (Bailey and López de Prado, 2014) and **the
  probability of backtest overfitting** via combinatorially symmetric
  cross-validation (Bailey, Borwein, López de Prado and Zhu, 2017) account for the
  number of strategy variants tried.

### 2.6 Open problems

- **Machine-learning models.** Classical predictive-ability theory (Diebold and
  Mariano, 1995; West, 1996) does not carry over directly to high-dimensional
  learners such as the Lasso; see Escanciano and Parra (2024).
- **Hidden search.** The true number of things tried is rarely known when there is
  hyperparameter tuning, LLM-generated strategies or shared datasets.
- **Ongoing refinements of alpha tests,** for example Massacci, Sarno, Trapani and
  Vallarino (2025).

## 3. Practical bottom line

| Question | Recommended tool | |---|---| | Does my best strategy beat the benchmark?
| SPA (Hansen, 2005) | | Which strategies are good? | StepM / stepwise SPA | | Which
forecasting models are best? | Model Confidence Set | | Thousands of funds or
factors? | FDR (Giglio–Liao–Xiu) or Bayesian shrinkage | | A single backtest in
practice? | Deflated Sharpe ratio |

Software: the Python `arch` package implements SPA, StepM and the Model Confidence
Set.

## References

- Bailey, D. H., Borwein, J., López de Prado, M. and Zhu, Q. J. (2017). The
  probability of backtest overfitting. *Journal of Computational Finance*, 20(4),
  39–69.
- Bailey, D. H. and López de Prado, M. (2014). The deflated Sharpe ratio: Correcting
  for selection bias, backtest overfitting, and non-normality. *Journal of Portfolio
  Management*, 40(5), 94–107.
- Bajgrowicz, P. and Scaillet, O. (2012). Technical trading revisited: False
  discoveries, persistence tests, and transaction costs. *Journal of Financial
  Economics*, 106(3), 473–491.
- Barras, L., Scaillet, O. and Wermers, R. (2010). False discoveries in mutual fund
  performance: Measuring luck in estimated alphas. *Journal of Finance*, 65(1),
  179–216.
- Benjamini, Y. and Hochberg, Y. (1995). Controlling the false discovery rate: A
  practical and powerful approach to multiple testing. *Journal of the Royal
  Statistical Society: Series B*, 57(1), 289–300.
- Diebold, F. X. and Mariano, R. S. (1995). Comparing predictive accuracy. *Journal
  of Business & Economic Statistics*, 13(3), 253–263.
- Escanciano, J. C. and Parra, R. (2024). Extending the scope of inference about
  predictive ability to machine learning methods. arXiv:2402.12838.
- Giglio, S., Liao, Y. and Xiu, D. (2021). Thousands of alpha tests. *Review of
  Financial Studies*, 34(7), 3456–3496.
- Hansen, P. R. (2005). A test for superior predictive ability. *Journal of Business
  & Economic Statistics*, 23(4), 365–380.
- Hansen, P. R., Lunde, A. and Nason, J. M. (2011). The model confidence set.
  *Econometrica*, 79(2), 453–497.
- Harvey, C. R. and Liu, Y. (2015). Backtesting. *Journal of Portfolio Management*,
  42(1), 13–28.
- Harvey, C. R. and Liu, Y. (2020). False (and missed) discoveries in financial
  economics. *Journal of Finance*, 75(5), 2503–2553.
- Harvey, C. R., Liu, Y. and Zhu, H. (2016). ... and the cross-section of expected
  returns. *Review of Financial Studies*, 29(1), 5–68.
- Hsu, P.-H., Hsu, Y.-C. and Kuan, C.-M. (2010). Testing the predictive ability of
  technical analysis using a new stepwise test without data snooping bias. *Journal
  of Empirical Finance*, 17(3), 471–484.
- Jensen, T. I., Kelly, B. and Pedersen, L. H. (2023). Is there a replication crisis
  in finance? *Journal of Finance*, 78(5), 2465–2518.
- Li, J., Liao, Z. and Quaedvlieg, R. (2022). Conditional superior predictive
  ability. *Review of Economic Studies*, 89(2), 843–875.
- Massacci, D., Sarno, L., Trapani, L. and Vallarino, P. (2025). A general randomized
  test for alpha. arXiv:2507.17599.
- Politis, D. N. and Romano, J. P. (1994). The stationary bootstrap. *Journal of the
  American Statistical Association*, 89(428), 1303–1313.
- Qu, R., Timmermann, A. and Zhu, Y. (2020). Do any economists have superior
  forecasting skills? Working paper.
- Quaedvlieg, R. (2021). Multi-horizon forecast comparison. *Journal of Business &
  Economic Statistics*, 39(1), 40–53.
- Romano, J. P. and Wolf, M. (2005). Stepwise multiple testing as formalized data
  snooping. *Econometrica*, 73(4), 1237–1282.
- Sullivan, R., Timmermann, A. and White, H. (1999). Data-snooping, technical trading
  rule performance, and the bootstrap. *Journal of Finance*, 54(5), 1647–1691.
- West, K. D. (1996). Asymptotic inference about predictive ability. *Econometrica*,
  64(5), 1067–1084.
- White, H. (2000). A reality check for data snooping. *Econometrica*, 68(5),
  1097–1126.

# Five Directions to Extend the Reality Check Literature

White's Reality Check (White, 2000) and its successors (SPA, StepM, Model Confidence Set, FDR methods and Bayesian shrinkage) leave several important gaps. Below are five research directions, each with the gap it addresses, possible approaches and the most relevant papers.

---

## 1. Correcting for search you can't see

**Gap.** Every method from White (2000) to Giglio, Liao and Xiu (2021) needs the number and correlation of the models tried. In practice that search is hidden: discarded specifications, hyperparameter sweeps, and earlier papers on the same data. Sullivan, Timmermann and White (1999) already stressed that the relevant universe includes rules tried by earlier researchers, yet it is almost never observed.

**Possible directions.**
- **Treat the size of the search as an unknown and estimate or bound it.** One way is to infer it from the distribution of published test statistics, extending Harvey, Liu and Zhu (2016), who estimate how many factors were tested, including unpublished ones. Another is a sensitivity analysis that reports how large the search would have to be for the result to stop being significant.
- **Build tests that stay valid over a whole range of possible search sizes,** like a robustness bound, instead of assuming one known K.

**Why it matters.** This turns a hidden assumption into an explicit robustness check that reviewers can evaluate.

---

## 2. Reality Checks for adaptive and machine-learning search

**Gap.** The Reality Check assumes a fixed set of K models chosen in advance. Modern search is sequential and adaptive: Bayesian optimization, genetic programming, and LLMs proposing strategies based on earlier results. The models themselves are also high-dimensional learners, and classical predictive-ability theory (Diebold and Mariano, 1995; West, 1996) does not apply to them directly (Escanciano and Parra, 2024).

**Possible directions.**
- **Model the search as a selection process** and use post-selection or selective inference (Lee, Sun, Sun and Taylor, 2016; Fithian, Sun and Taylor, 2014).
- **Bootstrap the entire search pipeline,** re-running the optimizer on each resample, instead of only bootstrapping the final set of models. Compute cost is the main obstacle, so cheaper approximations would be a contribution in themselves.
- **Use algorithmic stability or differential privacy** to bound how much the search can overfit, building on the "reusable holdout" (Dwork et al., 2015).

---

## 3. Testing as data arrives (anytime-valid methods)

**Gap.** The Reality Check (White, 2000) and SPA (Hansen, 2005) are run once, on a fixed sample. Practitioners monitor strategies live and repeatedly check whether they still work, which is data snooping across time.

**Possible directions.**
- **Build a sequential Reality Check** using e-values or test martingales (Vovk and Wang, 2021; Ramdas, Grünwald, Vovk and Shafer, 2023). These give guarantees that hold however often you look, while still taking the maximum over models.
- **Combine this with online FDR control** for a stream of new strategy proposals, using alpha-investing (Foster and Stine, 2008) or the online rules of Javanmard and Montanari (2018).

**Why it matters.** It fits how strategies are actually run, since you can decide to stop trading one at any time without invalidating the test.

---

## 4. Economic rather than statistical superiority

**Gap.** The tests ask whether mean outperformance is above zero. Investors care about outperformance after costs, capacity limits and risk, and possibly only in certain market regimes. Bajgrowicz and Scaillet (2012) show that transaction costs alone can eliminate apparent trading-rule profits.

**Possible directions.**
- **Extend conditional SPA** (Li, Liao and Quaedvlieg, 2022) to joint nulls over return, drawdown and turnover, for example "no model beats the benchmark on any of these measures after transaction costs."
- **Test whether some model beats the benchmark on a utility-based measure** for a range of risk-aversion levels.
- **Allow for nonstationarity:** test whether superiority holds within regimes, which ties into structural-break detection (Andrews, 1993).

---

## 5. Joining the frequentist and Bayesian views

**Gap.** FWER tests (White, 2000; Hansen, 2005; Romano and Wolf, 2005), FDR tests (Barras, Scaillet and Wermers, 2010; Giglio, Liao and Xiu, 2021) and Bayesian shrinkage (Jensen, Kelly and Pedersen, 2023) often reach very different conclusions on the same data, and practitioners have no principled way to choose between them. Harvey and Liu (2020) show that the choice of error rate matters economically because it determines how many skilled managers are missed.

**Possible directions.**
- **Give empirical-Bayes procedures frequentist error guarantees** in high-dimensional, cross-correlated settings, for example local-FDR estimates (Efron, 2010) with valid coverage under time dependence.
- **Develop decision-theoretic selection:** choose strategies to maximize expected out-of-sample performance after correcting for selection, not just to reject nulls. This could build on the winner's-curse corrections of Andrews, Kitagawa and McCloskey (2024) and on Tweedie's-formula shrinkage (Efron, 2011).
- **Run a large replication comparing all these methods** on common datasets (trading rules, factors, funds) with true out-of-sample validation, to show which one predicts future performance best.

---

## Where to start

- **Easiest to start with:** Directions 3 and 4 build on existing tools (e-values, conditional SPA) and offer clear empirical applications.
- **Highest impact:** Direction 2 is the most topical, given LLM- and ML-driven strategy search, but it is the hardest theoretically.

---

## References

- Andrews, D. W. K. (1993). Tests for parameter instability and structural change with unknown change point. *Econometrica*, 61(4), 821–856.
- Andrews, I., Kitagawa, T. and McCloskey, A. (2024). Inference on winners. *Quarterly Journal of Economics*, 139(1), 305–358.
- Bajgrowicz, P. and Scaillet, O. (2012). Technical trading revisited: False discoveries, persistence tests, and transaction costs. *Journal of Financial Economics*, 106(3), 473–491.
- Barras, L., Scaillet, O. and Wermers, R. (2010). False discoveries in mutual fund performance: Measuring luck in estimated alphas. *Journal of Finance*, 65(1), 179–216.
- Diebold, F. X. and Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253–263.
- Dwork, C., Feldman, V., Hardt, M., Pitassi, T., Reingold, O. and Roth, A. (2015). The reusable holdout: Preserving validity in adaptive data analysis. *Science*, 349(6248), 636–638.
- Efron, B. (2010). *Large-Scale Inference: Empirical Bayes Methods for Estimation, Testing, and Prediction*. Cambridge University Press.
- Efron, B. (2011). Tweedie's formula and selection bias. *Journal of the American Statistical Association*, 106(496), 1602–1614.
- Escanciano, J. C. and Parra, R. (2024). Extending the scope of inference about predictive ability to machine learning methods. arXiv:2402.12838.
- Fithian, W., Sun, D. and Taylor, J. (2014). Optimal inference after model selection. arXiv:1410.2597.
- Foster, D. P. and Stine, R. A. (2008). α-investing: A procedure for sequential control of expected false discoveries. *Journal of the Royal Statistical Society: Series B*, 70(2), 429–444.
- Giglio, S., Liao, Y. and Xiu, D. (2021). Thousands of alpha tests. *Review of Financial Studies*, 34(7), 3456–3496.
- Hansen, P. R. (2005). A test for superior predictive ability. *Journal of Business & Economic Statistics*, 23(4), 365–380.
- Harvey, C. R. and Liu, Y. (2020). False (and missed) discoveries in financial economics. *Journal of Finance*, 75(5), 2503–2553.
- Harvey, C. R., Liu, Y. and Zhu, H. (2016). ... and the cross-section of expected returns. *Review of Financial Studies*, 29(1), 5–68.
- Javanmard, A. and Montanari, A. (2018). Online rules for control of false discovery rate and false discovery exceedance. *Annals of Statistics*, 46(2), 526–554.
- Jensen, T. I., Kelly, B. and Pedersen, L. H. (2023). Is there a replication crisis in finance? *Journal of Finance*, 78(5), 2465–2518.
- Lee, J. D., Sun, D. L., Sun, Y. and Taylor, J. E. (2016). Exact post-selection inference, with application to the lasso. *Annals of Statistics*, 44(3), 907–927.
- Li, J., Liao, Z. and Quaedvlieg, R. (2022). Conditional superior predictive ability. *Review of Economic Studies*, 89(2), 843–875.
- Ramdas, A., Grünwald, P., Vovk, V. and Shafer, G. (2023). Game-theoretic statistics and safe anytime-valid inference. *Statistical Science*, 38(4), 576–601.
- Romano, J. P. and Wolf, M. (2005). Stepwise multiple testing as formalized data snooping. *Econometrica*, 73(4), 1237–1282.
- Sullivan, R., Timmermann, A. and White, H. (1999). Data-snooping, technical trading rule performance, and the bootstrap. *Journal of Finance*, 54(5), 1647–1691.
- Vovk, V. and Wang, R. (2021). E-values: Calibration, combination and applications. *Annals of Statistics*, 49(3), 1736–1754.
- West, K. D. (1996). Asymptotic inference about predictive ability. *Econometrica*, 64(5), 1067–1084.
- White, H. (2000). A reality check for data snooping. *Econometrica*, 68(5), 1097–1126.
