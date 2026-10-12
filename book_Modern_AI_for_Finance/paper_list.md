
### [ ] Gu, Kelly and Xiu’s “Empirical Asset Pricing via Machine Learning” (2020)
  - It found that trees and neural nets roughly doubled the predictive performance of
    linear models on US stocks
  - Momentum, liquidity and volatility were the most important inputs

Note: a link followed by `???` is not fully verified. SSRN blocks bots, so I
could not re-check the ID against the paper title. Title and authors came from
search results only.

# Causal Analysis for Finance

## Causal Inference Foundations

### [ ] 2025, Hünermund et al., "Causal Inference and Data Fusion in Econometrics"
    (https://arxiv.org/abs/1912.09104)
  - Reviews causal inference methods from the AI literature and their use in
      econometrics.
  - Targets hard data problems: unobserved confounding, selection-biased samples,
      surrogate experiments with imperfect compliance, and transfer of causal
      knowledge across different populations.
  - Argues that graph-based methods give one unified framework, algorithmic
      identification criteria, and a middle ground between structural
      econometrics and potential outcomes.
### [ ] 2020, Imbens, "Potential Outcome and Directed Acyclic Graph Approaches to
    Causality: Relevance for Empirical Practice in Economics"
    (https://arxiv.org/abs/1907.07271)
  - Essay that compares the potential outcome framework (Rubin, building on
      Neyman) with graphical approaches based on directed acyclic graphs.
  - Reviews the DAG literature, including Pearl and Mackenzie's The Book of Why.
  - Weighs the questions each framework answers well. Argues that much work in
      economics is closer to the potential outcome framework.
### [ ] 2020, Sharma et al., "DoWhy: An End-to-End Library for Causal Inference"
    (https://arxiv.org/abs/2011.04216)
  - Open-source Python library that makes causal assumptions first-class objects,
      based on causal graphs.
  - API follows four steps: model the data with a causal graph, identify the
      effect, estimate it, and refute the estimate.
  - Includes robustness checks: placebo tests, bootstrap tests, and tests for
      unobserved confounding. Works with EconML and CausalML for estimation.
### [ ] 2018, Pearl et al., "The Book of Why: The New Science of Cause and Effect"
    (https://bayes.cs.ucla.edu/WHY/)
  - Book for general readers on the basics of cause and effect, by Pearl and
      Mackenzie.
  - Argues that the old slogan "correlation is not causation" once led to a
      virtual ban on causal talk. Causal analysis now sits at the center of AI
      and other applications.
  - Chapters cover the ladder of causation, confounding and deconfounding,
      counterfactuals, and mediation.
### [ ] 2017, Peters et al., "Elements of Causal Inference: Foundations and
    Learning Algorithms"
    (https://mitp-content-server.mit.edu/books/content/sectbyfn?collid=books_pres_0&id=11283&fn=11283.pdf)
  - Concise, self-contained introduction to causal models and how to learn them
      from data. Free open access PDF from MIT Press.
  - Shows how to compute intervention distributions, infer causal models from
      observational and interventional data, and use causal ideas in machine
      learning.
  - Written for readers with a machine learning or statistics background.
      Includes exercises and code snippets.
### [ ] 2009, Pearl, "Causality: Models, Reasoning, and Inference"
    (https://bayes.cs.ucla.edu/BOOK-2K/)
  - Second edition. Comprehensive exposition of modern causal analysis. Unifies
      the probabilistic, manipulative, counterfactual, and structural approaches
      to causation.
  - Gives simple mathematical tools to link causal connections to statistical
      associations.
  - Chapters cover causal diagrams and identification of causal effects,
      structural models in social science and economics, and structure-based
      counterfactuals.

## Causality in Finance

### [ ] 2026, López de Prado et al., "Correcting the Factor Mirage: A Research
    Protocol for Causal Factor Investing" (https://ssrn.com/abstract=5931616)
  - Proves that specification errors can make factor strategies underperform and
      possibly lose money, even if all risk premia are constant and estimated
      with the correct sign.
  - Names the factor mirage: models that look strong statistically but are
      structurally flawed. Over-controlling for colliders raises the risk of
      few-shot p-hacking and bad outcomes.
  - Proposes changes to the econometric canon, using machine learning and causal
      inference. Published in The Journal of Portfolio Management.
### [ ] 2025, López de Prado et al., "Causality and Factor Investing: A Primer"
    (https://ssrn.com/abstract=5277078)
  - Notes that most factor strategies miss their in-sample promise. Names a less
      discussed cause: use of an econometric canon that ignores causal structure.
      P-hacking and backtest overfitting get more blame.
  - Introduces the factor mirage: a factor model that looks statistically valid
      but is causally misspecified.
  - Shows how collider bias and confounder bias in standard regressions give
      misleading inferences and poor out-of-sample performance.
### [ ] 2024, López de Prado et al., "The Case for Causal Factor Investing"
    (https://ssrn.com/abstract=4774522)
  - Factor model estimates of risk premia are unbiased only if the model is
      correctly specified. Correct specification needs knowledge of the causal
      graph of the data generating process.
  - Researchers pick specifications with associational arguments such as
      explanatory power, not with causal tools such as do-calculus. Models are
      likely misspecified and premia are biased.
  - Calls for rebuilding factor investing on causal foundations.
### [ ] 2023, López de Prado, "Causal Factor Investing: Can Factor Investing Become
    Scientific?" (https://ssrn.com/abstract=4205613)
  - Monograph, published as a Cambridge Elements title in 2023. The link is the
      SSRN version. Argues that almost all factor investing papers make
      associational claims and ignore the causal content of factor models.
  - Authors do not identify the causal graph, justify specifications with
      correlations, and propose no experiments to falsify causal mechanisms.
      Without a causal theory, findings are likely false.
  - Separates type-A and type-B spurious claims. Proposes fixes to make the field
      scientific.
### [ ] 2023, Jensen et al., "Is There a Replication Crisis in Finance?"
    (https://ssrn.com/abstract=3774514)
  - Answers claims that most finance studies cannot be replicated or rest on
      multiple testing of too many factors. Builds and estimates a Bayesian model
      of factor replication.
  - Finds that most asset pricing factors replicate. They cluster into 13 themes,
      and most themes are significant parts of the tangency portfolio.
  - Factors work out of sample in a new data set covering 93 countries. The large
      number of observed factors strengthens the evidence.
### [ ] 2022, Chen et al., "Open Source Cross-Sectional Asset Pricing"
    (https://ssrn.com/abstract=3604626)
  - Provides open data and code that reproduce nearly all cross-sectional stock
      return predictors. Covers 319 characteristics from earlier meta-studies.
  - Compares reproduced t-stats to the original papers. For the 161
      characteristics that were clearly significant, 98% of long-short portfolios
      have t-stats above 1.96.
  - A regression of reproduced t-stats on original t-stats gives a slope of 0.88
      and an R^2 of 82%.
### [ ] 2020, Hou et al., "Replicating Anomalies"
    (https://ssrn.com/abstract=2961979)
  - Builds a data library of 452 anomalies (447 in the SSRN draft). Limits
      microcap influence with NYSE breakpoints and value-weighted returns.
  - Finds that 65% of anomalies fail the single test hurdle of |t| 1.96. The
      failure rate rises to 82% with the multiple test hurdle of 2.78.
  - Replicated anomalies have economic magnitudes much smaller than originally
      reported. Concludes that capital markets are more efficient than previously
      recognized.
### [ ] 2016, Harvey et al., "...and the Cross-Section of Expected Returns"
    (https://ssrn.com/abstract=2249314)
  - Hundreds of factors try to explain expected returns. Given this data mining,
      the usual t-ratio above 2.0 is too weak for a new factor.
  - Introduces a multiple testing framework that allows for correlated tests and
      publication bias. Gives historical significance cutoffs from 1967 to today.
  - Estimates that a new factor needs a t-ratio above 3.0. Argues that most
      claimed findings in financial economics are likely false.

## Causal Structure from Text

### [ ] 2025, Saklad et al., "Can Large Language Models Infer Causal
    Relationships from Real-World Text?" (https://arxiv.org/abs/2505.18931)
  - Tests whether LLMs infer causal relations from real-world academic text,
      not from short synthetic text.
  - Builds a benchmark (ReCITE) from academic literature. Texts vary in
      length, complexity, and domain.
  - The best model reaches an average F1 of only 0.535.
### [ ] 2025, Garg et al., "Causal Claims in Economics"
    (https://arxiv.org/abs/2501.06873)
  - Maps each paper into an evidence-annotated claim graph. Nodes are
      standardized economic concepts. Edges are stated relationships, labeled
      by evidentiary basis.
  - Builds graphs for 44,852 economics papers from 1980-2023 with a
      multi-stage AI workflow.
  - The share of causal edges rises from 7.7% in 1990 to 31.7% in 2020.
      Causal narrative structure and novelty are positively associated with
      top-five publication and long-run citations.
### [ ] 2024, Zhang et al., "Causal Graph Discovery with Retrieval-Augmented
    Generation based Large Language Models" (https://arxiv.org/abs/2402.15301)
  - Proposes an LLM method (LACR) for causal graph recovery. It uses
      knowledge in the LLM, knowledge extracted from a scientific publication
      database, and experiment data.
  - Prompts the LLM to extract associations among factors. Then it verifies
      causality for each association.
  - Gives better graph quality than other LLM-based methods on benchmark
      datasets. It reacts to new evidence in the literature.
### [ ] 2023, Antonucci et al., "Zero-shot Causal Graph Extrapolation from Text
    via LLMs" (https://arxiv.org/abs/2312.14670)
  - Evaluates LLMs on inferring causal relations from natural language
      without training samples.
  - LLMs are competitive with traditional NLP and deep learning on a
      benchmark of pairwise relations.
  - Extends the approach to causal graphs through iterated pairwise queries.
      Preliminary test on biomedical abstracts with expert-validated graphs.
### [ ] 2023, Zecevic et al., "Causal Parrots: Large Language Models May Talk
    Causality But Are Not Causal" (https://arxiv.org/abs/2308.13067)
  - Argues that LLMs cannot be causal. Defines the meta SCM: a structural
      causal model whose variables encode causal facts about other SCMs.
  - Conjectures that LLM successes come from correlations between causal
      facts in the training text. The LLMs recite causal knowledge.
  - Empirical analysis supports this: current LLMs are weak "causal parrots".
### [ ] 2023, Jin et al., "Can Large Language Models Infer Causation from
    Correlation?" (https://arxiv.org/abs/2306.05836)
  - Proposes the Corr2Cause task: given correlational statements, find the
      causal relation between the variables.
  - Dataset has more than 200K samples. Evaluates seventeen LLMs.
  - Models score close to random. Finetuning helps in distribution but fails
      when variable names and wording change.
### [ ] 2023, Kiciman et al., "Causal Reasoning and Large Language Models:
    Opening a New Frontier for Causality" (https://arxiv.org/abs/2305.00050)
  - Behavioral study of how well LLMs generate causal arguments across many
      tasks.
  - GPT-3.5 and GPT-4 beat existing algorithms on pairwise causal discovery
      (97%), counterfactual reasoning (92%), and event causality (86%).
  - Failure modes are unpredictable. The authors point to combining LLMs
      with existing causal techniques.
### [ ] 2022, Yang et al., "A Survey on Extraction of Causal Relations from
    Natural Language Text" (https://arxiv.org/abs/2101.06426)
  - Surveys causality extraction. Covers explicit intra-sentential, implicit,
      and inter-sentential causality.
  - Lists benchmark datasets and assessment methods for causal relation
      extraction.
  - Reviews knowledge-based, statistical ML-based, and deep learning-based
      methods. Ends with open challenges.
### [ ] 2020, Mariko et al., "Financial Document Causality Detection Shared Task
    (FinCausal 2020)" (https://arxiv.org/abs/2012.02505)
  - Presents the FinCausal 2020 shared task on causality detection in
      financial documents and the FinCausal dataset.
  - Two sub-tasks: binary classification (Task 1) and relation extraction
      (Task 2).
  - 16 teams submitted runs. 13 of them wrote system description papers.

## Paper-to-Code and LLM Quant Agents

### [ ] 2026, Lin et al., "FactorEngine: A Program-level Knowledge-Infused
    Factor Mining Framework for Quantitative Investment"
    (https://arxiv.org/abs/2603.16365)
  - Mines alpha factors as Turing-complete code. It separates logic revision
      from parameter optimization, LLM-guided search from Bayesian search, and
      LLM use from local computation.
  - A bootstrapping module turns unstructured financial reports into
      executable factor programs. It uses a multi-agent extraction,
      verification, and code generation loop.
  - Backtests on real-world OHLCV data show higher IC/ICIR and better
      AR/Sharpe than baselines.
### [ ] 2025, Li et al., "R&D-Agent-Quant: A Multi-Agent Framework for
    Data-Centric Factors and Model Joint Optimization"
    (https://arxiv.org/abs/2505.15155)
  - Multi-agent framework (RD-Agent(Q)) that automates quant research and
      development by co-optimizing factors and models.
  - A Research stage forms hypotheses and tasks. A Development stage uses a
      code agent (Co-STEER) and real-market backtests. A feedback stage and a
      multi-armed bandit scheduler link the stages.
  - Reports up to 2X higher annualized returns than classical factor
      libraries, with 70% fewer factors.
### [ ] 2025, Seo et al., "Paper2Code: Automating Code Generation from
    Scientific Papers in Machine Learning" (https://arxiv.org/abs/2504.17192)
  - Presents PaperCoder, a multi-agent LLM framework that turns ML papers
      into code repositories.
  - Three stages: planning, analysis, and generation. Specialized agents
      handle each stage.
  - Evaluated with model-based and human evaluation, including the paper
      authors. Also beats strong baselines on PaperBench.
### [ ] 2025, Starace et al., "PaperBench: Evaluating AI's Ability to Replicate
    AI Research" (https://arxiv.org/abs/2504.01848)
  - Benchmark where agents replicate 20 ICML 2024 Spotlight and Oral papers
      from scratch: understand, write the codebase, run experiments.
  - Rubrics hold 8,316 gradable tasks, co-developed with the paper authors.
      An LLM-based judge grades the attempts.
  - The best agent scores 21.0% (Claude 3.5 Sonnet (New) with open-source
      scaffolding). Models do not yet beat the human baseline of ML PhDs.
### [ ] 2023, Blecher et al., "Nougat: Neural Optical Understanding for Academic
    Documents" (https://arxiv.org/abs/2308.13418)
  - Visual Transformer OCR model. It converts scientific document PDFs into a
      markup language.
  - Targets the loss of semantic information in PDFs, especially math
      expressions.
  - Evaluated on a new dataset of scientific documents. Models and code are
      released.

# Causal Macroeconomic Modeling

## Macro Variables in Asset Pricing

### [ ] 2023, Bryzgalova et al., "Bayesian Solutions for the Factor Zoo: We Just
    Ran Two Quadrillion Models" (https://ssrn.com/abstract=3481736)
  - Proposes a Bayesian framework for linear asset pricing models. It is simple,
      robust, and works in high dimensional problems.
  - For one model, it estimates prices of risk for tradable and non-tradable
      factors. It flags weakly identified factors.
  - For competing models, it picks the best one if a dominant one exists.
      Otherwise it builds a Bayesian model average SDF (BMA-SDF). Across 2.25
      quadrillion models, BMA-SDF beats existing models in-sample and
      out-of-sample.
### [ ] 2021, Giglio et al., "Asset Pricing with Omitted Factors"
    (https://ssrn.com/abstract=2865922)
  - Standard risk premia estimators are biased when priced factors are omitted
      from the model.
  - Proposes a three-pass method for the risk premium of an observable factor. It
      uses principal components of test asset returns to recover the factor
      space. Then it runs cross-sectional and time-series regressions.
  - It handles measurement error in the factor and detects spurious factors.
      Applied to a large panel of equity and non-equity portfolios.
### [ ] 2018, Barillas et al., "Comparing Asset Pricing Models"
    (https://ssrn.com/abstract=2676709)
  - Derives a Bayesian asset pricing test. It is computed in closed form from the
      standard F-statistic.
  - Extends the test to model probabilities for all models built from subsets of
      a set of candidate traded factors.
  - Finds that the Hou, Xue, and Zhang (2015) and Fama and French (2015, 2016)
      models are dominated by models with a momentum factor and monthly updated
      value and profitability factors.
### [ ] 2008, Welch et al., "A Comprehensive Look at the Empirical Performance of
    Equity Premium Prediction" (https://ssrn.com/abstract=517667) ???
  - Re-examines variables proposed to predict the equity premium. Examples:
      dividend price ratios, earnings-price ratios, book-market ratios, interest
      rates, and the consumption-based ratio cay. Tests are in-sample and
      out-of-sample, as of 2005.
  - Finds that over the last 30 years the prediction models failed both in-sample
      and out-of-sample. The models are unstable.
  - Finds that the models would not have helped an investor who had only the
      information available at the time to time the market.
### [ ] 2005, Bernanke et al., "What Explains the Stock Market's Reaction to
    Federal Reserve Policy?" (https://ssrn.com/abstract=890610)
  - Studies how unanticipated changes in the federal funds rate target move
      equity prices. Goal: size the typical reaction and find its sources.
  - Finds that an unanticipated 25 basis point rate cut goes with a rise of about
      1 percent in stock prices.
  - Uses the methods of Campbell and Ammer. Most of the effect comes through
      forecasted equity risk premiums. Very little comes through the real
      interest rate.
### [ ] 2001, Lettau et al., "Resurrecting the (C)CAPM: A Cross-Sectional Test When
    Risk Premia Are Time-Varying" (https://ssrn.com/abstract=935320)
  - Tests whether the CAPM and the consumption CAPM explain the cross-section of
      average stock returns. The pricing kernel is a conditional linear factor
      model, since risk premia vary over time.
  - The conditioning variable proxies for fluctuations in the log
      consumption-aggregate wealth ratio.
  - Conditional models beat unconditional ones. They do about as well as the
      Fama-French three-factor model on size and book-to-market portfolios.
### [ ] 2000, Pastor et al., "Comparing Asset Pricing Models: An Investment
    Perspective" (https://ssrn.com/abstract=217512) ???
  - Studies the portfolio choices of mean-variance investors. They use sample
      evidence to update prior beliefs centered on a risk-based or a
      characteristic-based pricing model.
  - With dogmatic beliefs and no limit on position size, optimal portfolios
      differ across models by economically significant amounts.
  - Modest doubt about the models' pricing ability shrinks the differences.
      Realistic limits on position size shrink them more, to zero in some cases.
### [ ] 1991, Ferson et al., "The Variation of Economic Risk Premiums"
    (https://ssrn.com/abstract=3281992)
  - Analyzes the predictable components of monthly stock and bond portfolio
      returns.
  - Most of the predictability links to sensitivity to economic variables in a
      rational asset pricing model with multiple betas.
  - The stock market risk premium matters most for stocks. Interest rate risk
      premiums matter most for bonds. Time variation in the premium for beta risk
      matters more than time variation in the betas.
### [ ] 1986, Chen et al., "Economic Forces and the Stock Market"
    (https://doi.org/10.1086/296344)
  - Tests whether innovations in macroeconomic variables are risks that the stock
      market rewards.
  - Variables: the spread between long and short interest rates, expected and
      unexpected inflation, industrial production, and the spread between high
      and low grade bonds. These sources of risk are significantly priced.
  - Neither the market portfolio nor aggregate consumption is priced separately.
      Oil price risk is not separately rewarded.

## Bayesian Macroeconometrics

### [ ] 2016, Herbst et al., "Bayesian Estimation of DSGE Models"
    (https://press.princeton.edu/books/hardcover/9780691161082/bayesian-estimation-of-dsge-models)
  - Book on computational methods for Bayesian analysis of DSGE models. Such
      models serve academic research, forecasting, and central bank policy work.
  - Covers MCMC for linearized DSGE models, sequential Monte Carlo for parameter
      inference, and particle filter likelihoods for nonlinear models.
  - Explains the theory behind the algorithms. Gives empirical applications and
      advice on tuning the methods and checking accuracy.
### [ ] 2015, Schorfheide et al., "Real-Time Forecasting With a Mixed-Frequency
    VAR" (https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2366046) ???
  - Builds a VAR for series observed at two frequencies: quarterly and monthly.
      Uses state-space form and Bayesian estimation with a Minnesota-style prior.
  - Shows how to compute the marginal data density. This lets the data pick the
      hyperparameters.
  - Real-time test against a quarterly VAR and MIDAS regressions: information
      that arrives within the quarter improves forecasts.
### [ ] 2015, Giannone et al., "Prior Selection for Vector Autoregressions"
    (https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2164591) ???
  - VARs have many parameters. Inference is unstable and out-of-sample forecasts
      are poor, especially with many variables.
  - Informative priors shrink the model toward a simpler benchmark. The paper
      treats the tightness of the prior as extra parameters in a hierarchical
      model and picks it optimally.
  - Reports good results for out-of-sample forecasts, factor models, and impulse
      response estimates.
### [ ] 2011, Del Negro et al., "Bayesian Macroeconometrics"
    (https://www.newyorkfed.org/medialibrary/media/research/economists/delnegro/B9AF4A64-F0AB-F7AF-1B0CB70D57957496.pdf)
  - Handbook chapter on Bayesian methods for macroeconomics. Starts from the
      challenges macroeconomists face in data-rich settings.
  - Covers reduced-form and structural VARs, linearized and nonlinear DSGE
      models, and ways to test how well DSGE models fit the data.
  - Ends with model uncertainty: how to choose and decide across several models.
### [ ] 2010, Koop et al., "Bayesian Multivariate Time Series Methods for Empirical
    Macroeconomics" (https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1514412)
  - Monograph on VARs, factor-augmented VARs, and time-varying parameter versions
      of these models, including multivariate stochastic volatility.
  - These models have many parameters, so over-parameterization can arise.
      Bayesian methods are a way to avoid it.
  - Explains Bayesian inference and the MCMC algorithms for state space models.
      Gives practical advice and empirical illustrations.
### [ ] 2010, Bańbura et al., "Large Bayesian vector auto regressions"
    (https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1292332)
  - Shows that a VAR with Bayesian shrinkage is a good tool for large dynamic
      models.
  - Shrinkage is set in relation to the cross-sectional dimension. Then small
      monetary VARs forecast better when extra macro variables and sectoral
      information are added.
  - Large VARs with shrinkage give credible impulse responses. They suit
      structural analysis.
### [ ] 2007, Gelman et al., "Data Analysis Using Regression and
    Multilevel/Hierarchical Models" (https://sites.stat.columbia.edu/gelman/arm/)
  - Textbook for applied researchers who fit linear, nonlinear, and multilevel
      regression models.
  - Covers regression, multilevel and hierarchical models, causal inference, and
      research design. Uses R and WinBUGS.
  - Examples come mostly from the authors' own applied research. The book site
      links to data, software, and errata.
### [ ] 2007, Smets et al., "Shocks and Frictions in US Business Cycles: A Bayesian
    DSGE Approach" (https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1687574)
  - Estimates a DSGE model of the US economy with Bayesian likelihood methods and
      seven macro time series.
  - The model has many real and nominal frictions and seven types of structural
      shocks. It competes with Bayesian VARs in out-of-sample prediction.
  - Tests which frictions matter. Uses the model to study the sources of business
      cycles, the effect of productivity on hours worked, and the Great
      Moderation.
### [ ] 2007, Ghysels et al., "MIDAS Regressions: Further Results and New
    Directions" (https://papers.ssrn.com/sol3/papers.cfm?abstract_id=885683)
  - Studies MIDAS regressions: regressions that use time series sampled at
      different frequencies.
  - Main focus is volatility. The method also applies in macroeconomics and
      finance. It joins new volatility estimators with the older distributed lag
      literature.
  - Compares lag structures that keep the model parsimonious. Proposes new
      extensions. The empirical part covers the risk-return tradeoff,
      microstructure noise, and volatility forecasting.
### [ ] 2006, Sims et al., "Were There Regime Switches in U.S. Monetary Policy?"
    (https://papers.ssrn.com/sol3/papers.cfm?abstract_id=579022)
  - Multivariate model identifies monetary policy and allows simultaneity and
      regime switching in coefficients and variances. Uses US data since 1959.
  - Best fit: only the variances of structural disturbances change. If
      coefficients may change too, the best fit changes only the policy rule,
      with three main regimes and one rare regime.
  - Regime differences are too small to explain the rise and fall of inflation in
      the 1970s and 1980s. Monetary targeting was central in the early 1980s.
### [ ] 2005, Primiceri, "Time Varying Structural Vector Autoregressions and
    Monetary Policy" (https://papers.ssrn.com/sol3/papers.cfm?abstract_id=352960)
  - Models monetary policy and the private sector as a time-varying structural
      VAR. Both the coefficients and the covariance matrix of the innovations
      vary over time.
  - Gives a simple way to model the law of motion of the covariance matrix.
      Proposes an efficient MCMC algorithm for the posterior.
  - Finds that systematic policy responded more aggressively to inflation and
      unemployment over forty years, with a negligible effect on the rest of the
      economy. Non-policy shocks matter more for the high inflation and
      unemployment episodes.
### [ ] 2003, Smets et al., "An Estimated Dynamic Stochastic General Equilibrium
    Model of the Euro Area"
    (https://papers.ssrn.com/sol3/papers.cfm?abstract_id=358102)
  - Builds and estimates a model of the euro area with sticky prices and wages,
      habit formation, capital adjustment costs, and variable capacity
      utilisation.
  - Bayesian estimation on seven macro series: GDP, consumption, investment,
      prices, real wages, employment, and the nominal interest rate. Ten
      structural shocks.
  - Uses the model to study the effects of the shocks, their role in euro area
      business cycles, and the output gap: actual output minus model-based
      potential output.
### [ ] 2003, Mariano et al., "A new coincident index of business cycles based on
    monthly and quarterly series"
    (https://papers.ssrn.com/sol3/papers.cfm?abstract_id=317983)
  - Common monthly coincident indexes, such as the composite index and the
      Stock-Watson index, ignore quarterly indicators like real GDP. They also
      lack economic interpretation.
  - Extends the Stock-Watson index. Applies maximum likelihood factor analysis to
      quarterly real GDP and monthly coincident indicators.
  - The new index relates to latent monthly real GDP.
### [ ] 1998, Bai et al., "Estimating and Testing Linear Models with Multiple
    Structural Changes" (https://dspace.mit.edu/handle/1721.1/63516)
  - Studies least squares estimation of linear models with several structural
      breaks at unknown dates, including the estimated break dates.
  - Tests whether change exists and how many breaks occur, including a test of l
      breaks against l+1 breaks. This supports a specific-to-general way to pick
      the number of breaks.
  - Covers partial structural change, where only some parameters shift. Derives
      convergence rates for the break fractions. Also describes a way to find the
      break points one at a time.
### [ ] 1989, Hamilton, "A New Approach to the Economic Analysis of Nonstationary
    Time Series and the Business Cycle"
    (https://www.econometricsociety.org/publications/econometrica/1989/03/01/new-approach-economic-analysis-nonstationary-time-series-and)
  - Proposes a tractable model of regime change. The parameters of an
      autoregression follow a discrete-state Markov process. The shifts are not
      observed, so the data must reveal them.
  - Gives a maximum likelihood algorithm and a basis for forecasting.
  - Applied to postwar US real GNP: switches from positive to negative growth
      recur over the cycle. This could give an objective rule to date recessions.
      A typical recession links to a permanent drop of about 3% in the level of
      GNP.
### [ ] 1986, Litterman, "Forecasting with Bayesian Vector Autoregressions: Five
    Years of Experience"
    (https://www.minneapolisfed.org/research/working-papers/forecasting-with-bayesian-vector-autoregressions-five-years-of-experience)
  - Reports five years of real forecasting with Bayesian vector autoregressions
      (BVARs).
  - The BVAR is inexpensive and reproducible. On average it is as accurate as the
      best known commercial forecasting services.
  - Covers the forecasting problem, the case for the Bayesian approach, how to
      implement it, and five years of results from one small BVAR model.
### [ ] 1984, Doan et al., "Forecasting and conditional projection using realistic
    prior distributions"
    (https://papers.ssrn.com/sol3/papers.cfm?abstract_id=305579) ???
  - Presents a forecasting method based on Bayesian estimation of VARs.
  - Applied to ten macro variables. Out-of-sample forecasts improve on univariate
      equations. The prior damps cross-variable effects, but the estimates still
      capture substantial interaction.
  - Shows how to make conditional projections to compare policy options, with a
      December 1982 Congressional Budget Office forecast as the example. Warns
      that such models describe statistical interdependence, not causal links.
### [ ] 1980, Sims, "Macroeconomics and Reality"
    (https://www.econometricsociety.org/publications/econometrica/1980/01/01/macroeconomics-and-reality)
  - Summarizes objections to existing econometric methods in macroeconomics. Some
      objections are recent and some are old.
  - Argues that, taken together, they make it unlikely that macroeconomic models
      are over-identified in the way standard statistical theory assumes.
  - Explores what follows from this. Gives an example of econometric work in a
      nonstandard style that answers the objections.

## Identification of Dynamic Causal Effects

### [ ] 2025, Ferreira et al., "Bayesian Local Projections"
    (https://www.bcb.gov.br/content/publicacoes/WorkingPaperSeries/WP581.pdf)
  - Proposes Bayesian local projections: informative priors regularize the local
      projection regressions. Targets the bias versus variance trade-off between
      local projections and VARs.
  - Impulse responses show richer adjustment dynamics than VAR ones, with
      comparable estimation uncertainty. Efficiency improves on standard and
      smooth local projections.
  - Also tested for forecasting: competitive with Bayesian VARs in a multivariate
      out-of-sample exercise.
### [ ] 2023, Bauer et al., "A Reassessment of Monetary Policy Surprises and
    High-Frequency Identification"
    (https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4086229)
  - Responds to concerns that high-frequency rate surprises around FOMC
      announcements may not be valid instruments for monetary policy effects.
  - Two fixes: add Fed Chair speeches to the announcements (roughly doubles their
      number and importance), and orthogonalize surprises against macro and
      financial data known before each announcement.
  - Asset price effects stay largely unchanged. Macro effects are larger and more
      significant than in earlier high-frequency studies.
### [ ] 2021, Miranda-Agrippino et al., "The Transmission of Monetary Policy
    Shocks" (https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2957644)
  - Combines a new identification that accounts for informational rigidities with
      a flexible method that bridges VARs and local projections.
  - Says much of the lack of robustness in earlier results comes from full
      information assumptions paired with severely misspecified models.
  - Finds a monetary tightening is contractionary, with no price puzzle and no
      output puzzle.
### [ ] 2021, Plagborg-Møller et al., "Local Projections and VARs Estimate the Same
    Impulse Responses" (https://www.mikkelpm.com/files/lp_var.pdf)
  - Proves that local projections and VARs estimate the same impulse responses.
      The result is nonparametric and needs only unrestricted lag structures.
  - So the two are dimension reduction techniques with a common target but
      different finite-sample properties. Short-run, long-run, and sign
      restriction identification work with either.
  - An instrument can be ordered first in a recursive VAR, even under
      non-invertibility. Linear VARs are as robust to non-linearities as linear
      local projections.
### [ ] 2018, Nakamura et al., "High-Frequency Identification of Monetary
    Non-Neutrality: The Information Effect"
    (https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2298991) ???
  - Treats unexpected interest rate changes in a 30-minute window around
      scheduled Fed announcements as news about monetary policy.
  - After a rate hike, nominal and real rates rise roughly one for one several
      years out. Expected inflation barely moves. Output growth forecasts rise,
      the opposite of standard models.
  - Argues Fed announcements also shift beliefs about other economic
      fundamentals. In their model this information effect matters for the output
      effect of policy shocks.
### [ ] 2018, Stock et al., "Identification and Estimation of Dynamic Causal
    Effects in Macroeconomics Using External Instruments"
    (https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3106657) ???
  - Explains external instruments: outside as-if random variation used to
      identify dynamic causal effects of macro shocks. It is the time series
      counterpart of microeconometric IV.
  - Gives conditions on instruments and controls for valid inference on
      structural impulse responses.
  - Compares a one-step IV regression with a two-step VAR method. Under a
      restrictive condition the one-step method holds even if the VAR is not
      invertible, so comparing the two tests invertibility.
### [ ] 2018, Arias et al., "Inference Based on Structural Vector Autoregressions
    Identified With Sign and Zero Restrictions: Theory and Applications"
    (https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2580264)
  - Argues that common algorithms for sign and zero restrictions add sign
      restrictions on variables meant to be unrestricted. This biases results and
      gives misleading confidence intervals.
  - Proposes a faster algorithm that adds no extra restrictions.
  - Applies it to optimism shocks and business cycles, and to deficit-financed
      tax cuts versus spending. Finds little support for the clear answers of
      earlier studies.
### [ ] 2016, Ramey, "Macroeconomic Shocks and Their Propagation"
    (https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2732455) ???
  - Handbook chapter. Starts with why identifying macro shocks is hard, then
      surveys recent identification methods.
  - Reviews monetary, fiscal, and technology shocks in detail. Each section adds
      new estimates that compare and synthesize the literature.
  - Asks how much of output and hours fluctuations the leading shocks explain.
      Concludes we are much closer to understanding the shocks than twenty years
      ago.
### [ ] 2015, Baumeister et al., "Sign Restrictions, Structural Vector
    Autoregressions, and Useful Prior Information"
    (https://www.nber.org/papers/w20741)
  - Gives a simpler analytical description and algorithm for Bayesian inference
      in SVARs. It covers over, just, and under identified models.
  - Shows the traditional sign restriction approach carries implicit informative
      priors. Their influence does not vanish as the sample grows.
  - Argues researchers should state and defend their prior beliefs. Illustrates
      with a simple model of the US labor market.
### [ ] 2015, Gertler et al., "Monetary Policy Surprises, Credit Costs, and
    Economic Activity"
    (https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2450924) ???
  - Combines a monetary VAR with high-frequency identification of policy shocks.
      Output and inflation responses match textbook theory and standard VAR
      analysis.
  - Small moves in short rates lead to large moves in credit costs and activity.
      Term premia and credit spreads drive much of the credit cost response.
  - Forward guidance matters for the strength of transmission.
### [ ] 2010, Rubio-Ramírez et al., "Structural Vector Autoregressions: Theory of
    Identification and Algorithms for Inference"
    (https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1296848)
  - Notes there were no workable rank conditions to check whether an SVAR is
      globally identified. Efficient small-sample inference algorithms were also
      missing for some restrictions, such as long-run ones.
  - Gives general rank conditions for over and exactly identified models, checked
      as a matrix filling exercise. For exactly identified models the check
      reduces to counting.
  - Develops efficient algorithms for small-sample estimation and inference.
### [ ] 2009, Kilian, "Not All Oil Price Shocks Are Alike: Disentangling Demand and
    Supply Shocks in the Crude Oil Market"
    (https://papers.ssrn.com/sol3/papers.cfm?abstract_id=975262) ???
  - Splits the real price of crude oil into four shocks: supply shocks from OPEC
      political events, other supply shocks, demand for industrial commodities,
      and oil market specific demand.
  - Uses a new measure of global real economic activity. Estimates the size,
      timing, and effects of each shock for 1975 to 2005.
  - Finds the source of an oil price rise matters for the effect on US real GDP
      and CPI inflation. Policy responses should account for the cause.
### [ ] 2005, Jordà, "Estimation and Inference of Impulse Responses by Local
    Projections" (https://www.aeaweb.org/articles?id=10.1257/0002828053828518)
  - Estimates a separate local projection at each horizon instead of
      extrapolating from a fitted VAR.
  - Uses simple regressions in standard software. Is more robust to
      misspecification, makes joint or pointwise inference simple, and allows
      flexible nonlinear specifications.
  - Backed by Monte Carlo evidence and an application to a simple closed economy
      New Keynesian model.
### [ ] 2005, Uhlig, "What are the effects of monetary policy on output? Results
    from an agnostic identification procedure"
    (https://papers.ssrn.com/sol3/papers.cfm?abstract_id=199742) ???
  - Agnostic identification: sign restrictions on the responses of prices,
      nonborrowed reserves, and the federal funds rate to a monetary policy
      shock. No restriction on real GDP.
  - Finds contractionary monetary policy shocks have an ambiguous effect on real
      GDP. Other results largely confirm earlier VAR findings.
  - A counterfactual with policy shocks set to zero after December 1979 stays
      close to actual real GDP. So the label "Volcker recession" looks misplaced.
### [ ] 2004, Romer et al., "A New Measure of Monetary Shocks: Derivation and
    Implications" (https://papers.ssrn.com/sol3/papers.cfm?abstract_id=428360) ???
  - Argues common policy measures, such as the federal funds rate, mix in
      non-policy forces and Fed responses to expected conditions. This biases
      estimated effects.
  - Builds a US shock series for 1969 to 1996. Controls for Fed forecasts of
      output and inflation and uses changes in the intended funds rate around
      scheduled FOMC meetings.
  - Finds large, fairly quick, and significant effects on output and inflation.
      Effects are stronger and faster than with earlier measures.

## Causal Graphs and Discovery for Time Series

### [ ] 2021, Lorch et al., "DiBS: Differentiable Bayesian Structure Learning"
    (https://arxiv.org/abs/2105.11839)
  - Bayesian structure learning: infers graph structure while reasoning about
      uncertainty. Uses a fully differentiable framework on a latent
      probabilistic graph.
  - Makes no assumption on the form of the local conditional distributions.
      Infers graph and parameters jointly, including nonlinear models such as
      neural networks.
  - Builds a variational inference method for distributions over structural
      models. Reports it beats related approaches on simulated and real data.
### [ ] 2020, Runge, "Discovering Contemporaneous and Lagged Causal Relations in
    Autocorrelated Nonlinear Time Series Datasets"
    (https://arxiv.org/abs/2003.03685)
  - PCMCI+: a conditional independence method for causal discovery in
      observational time series. Covers linear and nonlinear, lagged and
      contemporaneous links. Assumes causal sufficiency.
  - Extends PCMCI to contemporaneous links. Optimizes the conditioning sets to
      make CI tests more reliable under strong autocorrelation.
  - Reports higher adjacency detection power, better contemporaneous orientation
      recall, and better control of false positives. Runs much faster than the PC
      algorithm.
### [ ] 2020, Pamfil et al., "DYNOTEARS: Structure Learning from Time-Series Data"
    (https://arxiv.org/abs/2002.00498)
  - Score-based structure learning for dynamic Bayesian networks. Estimates
      contemporaneous (intra-slice) and time-lagged (inter-slice) relations
      together.
  - Minimizes a penalized loss under a smooth acyclicity constraint.
  - Beats other methods on simulated data, mostly in high dimensions. Applied to
      real data from finance and molecular biology.
### [ ] 2019, Runge et al., "Detecting and quantifying causal associations in large
    nonlinear time series datasets" (https://arxiv.org/abs/1702.07007)
  - PCMCI: reconstructs causal networks from large observational time series.
      Combines linear or nonlinear conditional independence tests with a causal
      discovery algorithm.
  - Targets data that is high-dimensional and nonlinear, with limited sample
      size.
  - Tested on a climate teleconnection between the tropical Pacific and
      extra-tropical temperatures, and on large synthetic data. Reports higher
      detection power than alternatives.
### [ ] 2015, Brodersen et al., "Inferring causal impact using Bayesian structural
    time-series models" (https://arxiv.org/abs/1506.00356)
  - Infers the causal impact of a market intervention on an outcome over time.
      Uses a diffusion-regression state-space model. A synthetic control gives
      the counterfactual response.
  - Versus difference-in-differences: shows how impact evolves, takes empirical
      priors in a Bayesian setup, and handles local trends, seasonality and
      covariates.
  - Uses MCMC for inference. Tested on simulated data and on an online
      advertising campaign. Implemented in the CausalImpact R package.
### [ ] 2013, Moneta et al., "Causal Inference by Independent Component Analysis:
    Theory and Applications" (https://www.lem.sssup.it/WPLem/documents/ciica.pdf)
  - Presents a method to estimate structural vector autoregressive (SVAR) models.
      It uses non-normality to recover the causal structure behind the data.
  - Applied to microeconomic data: firm growth and firm performance.
  - Applied to macroeconomic data: the effects of monetary policy.
### [ ] 2010, Hyvärinen et al., "Estimation of a Structural Vector Autoregression
    Model Using Non-Gaussianity" (https://jmlr.org/papers/v11/hyvarinen10a.html)
  - Combines a non-Gaussian model of instantaneous effects with autoregressive
      models. The result is a structural vector autoregression (SVAR) model.
  - Shows the model is identifiable without prior knowledge of the network
      structure.
  - Gives efficient estimation methods and tests for the significance of causal
      influences. Applied to financial and brain imaging data.
### [ ] 2010, White et al., "Granger Causality and Dynamic Structural Systems"
    (https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1578749) ???
  - Defines direct and total structural causality in a general dynamic structural
      system. Covers structural VARs and recursive time-series natural
      experiments.
  - Links Granger (G-)causality to structural causality: under conditional
      exogeneity, G-causality holds if and only if a matching form of structural
      causality holds.
  - Proposes new tests for G-causality and conditional exogeneity. Applies them
      to oil and gasoline prices, monetary policy and industrial production, and
      stock returns and macroeconomic announcements.
### [ ] 2003, Demiralp et al., "Searching for the Causal Structure of a Vector
    Autoregression" (https://ssrn.com/abstract=388840)
  - A structural VAR needs a causal order among the contemporaneous variables.
      Identifying transformations are not unique. It is widely believed the order
      must come from prior theory or other criteria not rooted in the data.
  - Applies graph-theoretic search based on conditional independence to select
      the causal order, or to narrow it to a small equivalence class. Builds on
      Swanson and Granger (1997).
  - Monte Carlo study of the PC algorithm shows how accuracy varies with signal
      strength at realistic sample lengths. Concludes that graph-theoretic
      methods may help SVAR analysis.
### [ ] 2003, Friedman et al., "Being Bayesian About Network Structure: A Bayesian
    Approach to Structure Discovery in Bayesian Networks"
    (https://ai.stanford.edu/~nir/Papers/FK1Full.pdf)
  - Computes the posterior probability of a feature of a Bayesian network: the
      total posterior of all models that contain it. Useful when data is modest
      and many structures have non-negligible posterior.
  - Shows how to sum efficiently over the exponential number of networks
      consistent with a fixed variable order.
  - Runs MCMC over orders, not structures. The order space is smaller and more
      regular. Compared with full model averaging, MCMC over structures, and a
      non-Bayesian bootstrap on synthetic and real data.
### [ ] 1997, Swanson et al., "Impulse Response Functions Based on a Causal
    Approach to Residual Orthogonalization in Vector Autoregressions"
    (https://doi.org/10.1080/01621459.1997.10473634)
  - Proposes a data-determined method to test structural models of the errors in
      a vector autoregression.
  - Standard t statistics can test the overidentifying restrictions that a model
      implies. The method also compares prior knowledge of the error structure
      with the properties of the data.
  - Aims to make error orderings for impulse response and variance decomposition
      analyses sensible, given the data. Illustrated with two economic examples.
### [ ] 1969, Granger, "Investigating Causal Relations by Econometric Models and
    Cross-spectral Methods"
    (https://jeti.uni-freiburg.de/studenten_seminar/stud_sem_SS_09/grangercausality.pdf)
  - Asks how to tell the direction of causality between two related variables,
      and whether feedback exists. Gives testable definitions of causality and
      feedback, shown on simple two-variable models.
  - Discusses apparent instantaneous causality. Says it often comes from slow
      information recording or from too narrow a set of possible causal
      variables.
  - Splits the cross spectrum into two parts, one per causal direction in a
      feedback system. This gives measures of causal lag and strength.
      Generalizes with the partial cross spectrum.

## Real-Time Macroeconomic Data

### [ ] 2021, McCracken et al., "FRED-QD: A Quarterly Database for Macroeconomic
    Research" (https://ssrn.com/abstract=3843838) ???
  - Presents FRED-QD, a large quarterly macro database. It is updated regularly
      through FRED. It is closely modeled on the Stock and Watson (2012) dataset.
  - Factors from FRED-QD behave like those from the original Stock and Watson
      dataset. The dominant factors are insensitive to outliers.
  - Studies how unit root tests guide the choice of transformation codes. Factors
      from the data help forecast many macro series. The choice of transformation
      codes changes forecast accuracy.
### [ ] 2016, McCracken et al., "FRED-MD: A Monthly Database for Macroeconomic
    Research" (https://ssrn.com/abstract=2646151)
  - Describes a large monthly macro database. Goal: a convenient starting point
      for empirical work that needs big data.
  - The data are updated monthly from FRED and are public. The database spares
      researchers from managing data changes and revisions.
  - Factors from the data have the same predictive content as factors from
      vintages of the Stock-Watson dataset. Diffusion indexes built from the
      factors may help study business cycle chronology.
### [ ] 2011, Croushore, "Frontiers of Real-Time Data Analysis"
    (https://ssrn.com/abstract=1118356)
  - Surveys research on data revisions. Topics: properties of revisions,
      structural modeling, forecasting, monetary policy analysis, and current
      analysis of business conditions.
  - Reports progress in building better real-time data sets around the world.
  - Says more research is needed in key areas, and that existing work points to
      more fruitful topics.
### [ ] 2001, Croushore et al., "A Real-Time Data Set for Macroeconomists"
    (https://ssrn.com/abstract=170531)
  - Presents a real-time data set. It holds vintages, or snapshots, of the major
      macro data available at quarterly intervals in real time.
  - Uses: test the robustness of published econometric results, analyze policy,
      and forecast.
  - Shows how variables behave across vintages. Re-runs key macro papers on
      different vintages. Shows how revisions can affect policy analysis and
      forecasts.
### [ ] 2001, Orphanides, "Monetary Policy Rules Based on Real-Time Data"
    (https://ssrn.com/abstract=70448) ???
  - Measures the information problems in simple monetary policy rules, using
      Taylor's rule as the example.
  - Real-time policy recommendations differ considerably from those based on ex
      post revised data. They are revised substantially even a year after the
      quarter.
  - Reaction functions estimated on revised data can mislead about historical
      policy. With Fed staff forecasts, forward-looking rules describe 1987 to
      1992 policy better than Taylor-type rules.
### [ ] 2006, Federal Reserve Bank of St. Louis, "ALFRED: Archival Federal Reserve
    Economic Data" (https://alfred.stlouisfed.org/)
  - Web database of data vintages. It retrieves each economic data release that
      was available on a specific date in history.
  - The page tagline reads: economic data time travel since 2006. For the most
      current release, the page points to FRED.
### [ ] n.d., Federal Reserve Bank of St. Louis, "FRED-MD and FRED-QD: Monthly and
    Quarterly Databases for Macroeconomic Research"
    (https://www.stlouisfed.org/research/economists/mccracken/fred-databases)
  - Web page hosting the monthly FRED-MD and quarterly FRED-QD databases. They
      are built for big data empirical analysis.
  - Both are updated in real time through FRED and are public. The St. Louis Fed
      Data Desk handles data changes and revisions.
  - Historical vintages: FRED-MD from 1999-08 and FRED-QD from 2018-05. The page
      also links the working papers, appendices, and Matlab code.
