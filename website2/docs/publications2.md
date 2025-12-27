# Publications

## 2025

### Causify Benchmark: A Benchmark of Causal AI for Predictive Maintenance
**Authors:** K. Taduri, S. Dhande, G.P. Saggese, P. Smith
**Publication:** arXiv preprint arXiv:2512.01149, 2025
**Links:** [arXiv](https://arxiv.org/abs/2512.01149)

This paper provides the first comprehensive benchmark comparing Bayesian structural causal models against traditional correlation-based machine learning for predictive maintenance on 10,000 CNC machines.

- **$49,500 Annual Advantage Over Best ML**: Bayesian Structural Causal Model (Model L6) outperformed the best correlation-based decision tree by $49,500 annually and PyMC BART ensemble by $18,000, demonstrating causal AI's superiority even against carefully tuned alternatives.
- **Highest Recall Through Explicit Failure Mechanisms**: Achieved 93.9% recall (62 out of 66 failures detected) by explicitly modeling four independent physical failure mechanisms (tool wear, heat dissipation, power, overstrain) combined via Noisy-OR gate, compared to implicit pattern recognition in ML models.
- **Superior to Cost-Optimized ML**: Demonstrates causal AI outperforms even carefully tuned, cost-aware correlation-based models on business-relevant metrics accounting for realistic cost asymmetry between false negatives (missed failures) and false positives (unnecessary maintenance).
- **Enhanced Interpretability**: Provides root-cause explanations through explicit structural equations for each failure mechanism, enabling targeted maintenance interventions (e.g., "replace worn tool") rather than generic inspections.
- **Business-Aligned Optimization Framework**: Validates critical importance of optimizing for actual business costs rather than statistical accuracy metrics, with properly calibrated models substantially outperforming accuracy-optimized approaches.

### Causify DataMap: Causal Probabilistic Reasoning
**Authors:** G.P. Saggese, P. Smith
**Publication:** Manuscript to be submitted to arXiv, 2025

This paper introduces a system that automatically generates mathematical models capable of reasoning and decision-making under uncertainty by integrating knowledge graphs, large language models (LLMs), and causal inference.

- **Automated Causal Model Generation**: Converts unstructured knowledge and data into executable causal probabilistic models, eliminating manual model design and enabling scalable reasoning systems.
- **Unified Framework**: Combines LLMs for extracting structured causal relations, knowledge graphs for semantic grounding, and Bayesian inference for uncertainty quantification—creating an integrated Causal AI pipeline.
- **Operational Decision-Making Under Uncertainty**: Demonstrated on real-world predictive maintenance problems, the system makes interpretable, data-driven decisions under uncertain conditions, showing superior robustness compared to correlation-based machine learning.
- **Probabilistic Theorem Proving**: Extends traditional theorem proving into probabilistic domains, enabling dynamic question generation, hypothesis evaluation, and decision optimization.

### Causify DataFlow: A Causal Model Simulator
**Authors:** G.P. Saggese, P. Smith
**Publication:** Manuscript to be submitted to arXiv, 2025

This paper presents a rigorous computational framework for simulating causal models that process time-series data, ensuring temporal correctness from research through production deployment.

- **Causality by Design**: DataFlow models computation as a directed acyclic graph (DAG) with explicit knowledge-time semantics, ensuring every output at time t depends only on information available at or before t. This architecture prevents future-peeking, data leakage, and non-causal bugs.
- **Tileability Guarantees Causal Correctness**: Computations are "tileable"—outputs remain invariant to data partitioning as long as the required context window is present. This property enables accurate causal simulation, unified batch/streaming execution, and efficient memory-performance trade-offs.
- **Unified Research-Production Framework**: The same DAG specification executes with identical semantics in historical simulation, real-time streaming, backtesting, training, and deployment. This consistency enables building complete causal AI systems—forecasting models, trading agents, anomaly detectors, and control pipelines—without rewriting logic or risking semantic drift.

### Causify DataPull: A Causal Data Layer
**Authors:** G.P. Saggese, P. Smith
**Publication:** Manuscript to be submitted to arXiv, 2025

This paper describes Causify's end-to-end framework for ingesting, storing, and serving time-series data with correct temporal semantics for downstream causal modeling and analytics.

- **Knowledge-Time-Aware, Bitemporal Data Layer**: DataPull preserves both event time (when something happened) and knowledge time (when we learned about it), ensuring models only use information actually available at decision time—foundational for reliable causal analysis.
- **Raw, Standardized, and Reproducible Time-Series Foundation**: Stores data exactly as received while unifying semantics across vendors and resolutions, providing a clean, unbiased, fully lineage-tracked substrate that causal models can trust.
- **Integrated QA and Automatic Domain-Aware Cleaning**: Systematic early-stage QA detects missing data, malformed intervals, inconsistencies, and schema drift, ensuring causal models learn from clean, coherent, and temporally valid inputs.

### Causify Sentinel: A Causal Failure Prediction Framework
**Authors:** C. Ma, S. Nikiforova, G.P. Saggese, P. Smith, K. Taduri
**Publication:** Manuscript to be submitted to arXiv, 2025

This paper presents a causal failure prediction framework for time-series systems achieving 100% recall with early warning capabilities ranging from weeks to months before failure events, validated on wind turbine main bearing failures.

**Performance Summary:**
- **100% Recall**: Detected all 9 main bearing failures in dataset (9/9)
- **Early Warning**: Weeks to months advance notice
- **False Positive Rate**: Low, with robust confounder adjustment

**Key Contributions:**
- **Physics-Informed Causal Models**: Combines domain knowledge of bearing failure mechanisms (degradation→friction→heat→temperature cascade) with data-driven Bayesian inference, ensuring robustness to distribution shifts and enabling transfer across different operational environments.
- **Cross-Sectional Anomaly Detection with Confounder Adjustment**: Uses propensity score weighting to construct confounder-adjusted baselines removing spurious associations from ambient temperature, operational load, turbine age, and seasonal patterns while preserving true causal signal from bearing degradation.
- **Probabilistic Alerting with Survival Analysis**: Maps smoothed residuals to failure probabilities using calibrated logistic regression with explicit confounder terms, computing rolling cumulative survival probabilities with threshold-based alerting robust to transient noise.
- **State-Space Forecasting and RUL Prediction**: Projects future degradation trajectories using linear Gaussian state-space models with Kalman filtering, estimating remaining useful life through exponential degradation model with continuous updates enabling proactive maintenance prioritization.
- **Novel Survival Forecast Score Metric**: Introduces time-weighted evaluation metric for remaining useful life predictions properly handling right-censored observations and emphasizing accuracy near failure events, supporting business-aligned model optimization.

### Causify Grid: Causal Inference in Energy Demand Prediction
**Authors:** C. Ma, G.P. Saggese, P. Smith
**Publication:** Manuscript to be submitted to arXiv, 2025

This paper demonstrates the importance of causal inference for energy demand prediction through a case study showing that naive modeling approaches ignoring causal relationships result in reduced robustness and biased parameter estimates.

**Key Results:**
- **Test MAPE**: 4.17% (Causal) vs 4.76% (Non-Causal) — **12.5% improvement**
- **Temperature Coefficient Bias**: Correct (Causal) vs +47.8% overestimate (Non-Causal)

**Key Contributions:**
- **47.8% Coefficient Bias Reduction**: Non-causal model overestimates temperature coefficient by 47.8% due to confounder bias (time-of-day affects both temperature and demand), while causal model correctly estimates the true causal effect using backdoor criterion adjustment.
- **Season-Dependent Temperature Sensitivity**: Causal model reveals energy demand responds to temperature fluctuations with season-dependent sensitivity, explaining observed heteroscedasticity through alignment/misalignment of causal effects (heating in winter, cooling in summer).
- **Full Bayesian Treatment**: Implements complete Bayesian causal model using Pyro with stochastic variational inference, providing calibrated uncertainty quantification for risk-aware planning.

### Causify Horizon: A Causal Demand Forecasting Framework
**Authors:** C. Ma, G. Pomazkin, G.P. Saggese, P. Smith, D. Tikhomirov, N. Trubacheva
**Publication:** Manuscript to be submitted to arXiv, 2025

This paper presents a causal demand forecasting framework integrating causal knowledge graphs with state-space models for supply chain optimization, inventory management, and resource allocation.

- **Causal Knowledge Graph Integration**: Combines offline knowledge graph construction (domain expertise, LLM-assisted extraction, empirical data) with online query-specific forecasting that compiles relevant subgraphs into Bayesian networks for interventional inference. For instance, when forecasting spare parts demand, the system automatically identifies relevant causal factors (device age, warranty status, environmental conditions) and constructs a custom model for each query.
- **State-Space Forecasting with Causal Semantics**: Models repair demand using normalized repair rates with variance-stabilizing transformations, capturing local level, seasonal patterns, and regression terms for device attributes through linear Gaussian state-space models with Kalman filtering.
- **Spare Parts Demand Forecasting**: Applied to spare parts demand prediction accounting for complex interactions between device aging, warranty policies, maintenance schedules, and environmental conditions, demonstrating superior robustness to distribution shifts.
- **Calibrated Uncertainty Quantification**: Provides epistemic uncertainty estimates enabling risk-aware inventory planning, with validation on historical natural experiments confirming correct prediction of causal effects from policy changes.

---

## 2022

### A Novel Module-Sign Low-Power Implementation for the DLMS Adaptive Filter With Low Steady-State Error
**Authors:** G. Di Meo, D. De Caro, G.P. Saggese, E. Napoli, N. Petra, A.G.M. Strollo
**Publication:** IEEE Transactions on Circuits and Systems I: Regular Papers, vol. 69, no. 1, pp. 297-308, 2022
**Links:** [DOI](https://doi.org/10.1109/TCSI.2021.3088913) | [DBLP](https://dblp.org/rec/journals/tcasI/MeoCSNPS22.bib)
**Note:** PDF not freely available

### Approximate Multipliers Using Static Segmentation: Error Analysis and Improvements
**Authors:** A.G.M. Strollo, E. Napoli, D. De Caro, N. Petra, G.P. Saggese, G. Di Meo
**Publication:** IEEE Transactions on Circuits and Systems I: Regular Papers, vol. 69, no. 6, pp. 2449-2462, 2022
**Links:** [DOI](https://doi.org/10.1109/TCSI.2022.3152921) | [PDF](https://ieeexplore.ieee.org/iel7/8919/9782465/09726786.pdf) | [DBLP](https://dblp.org/rec/journals/tcasI/StrolloNCPSM22.bib)

---

## 2011

### Automated Derivation of Application-Specific Error Detectors Using Dynamic Analysis
**Authors:** K. Pattabiraman, G.P. Saggese, D. Chen, Z. Kalbarczyk, R.K. Iyer
**Publication:** IEEE Transactions on Dependable and Secure Computing, vol. 8, no. 5, pp. 640-655, 2011
**Links:** [DOI](https://doi.org/10.1109/TDSC.2010.19) | [PDF](https://www.researchgate.net/publication/220068507_Automated_Derivation_of_Application-Specific_Error_Detectors_Using_Dynamic_Analysis) | [DBLP](https://dblp.org/rec/journals/tdsc/PattabiramanSCKI11.bib)

---

## 2006

### Dynamic Derivation of Application-Specific Error Detectors and their Implementation in Hardware
**Authors:** K. Pattabiraman, G.P. Saggese, D. Chen, Z. Kalbarczyk, R.K. Iyer
**Publication:** Sixth European Dependable Computing Conference (EDCC 2006), Coimbra, Portugal, pp. 97-108, 2006
**Links:** [DOI](https://doi.org/10.1109/EDCC.2006.9) | [PDF](https://tcipg.org/sites/default/files/papers/2006_Pattabiraman_et_al.pdf) | [DBLP](https://dblp.org/rec/conf/edcc/PattabiramanSCKI06.bib)

---

## 2005

### An Experimental Study of Soft Errors in Microprocessors
**Authors:** G.P. Saggese, N.J. Wang, Z. Kalbarczyk, S.J. Patel, R.K. Iyer
**Publication:** IEEE Micro, vol. 25, no. 6, pp. 30-39, 2005
**Links:** [DOI](https://doi.org/10.1109/MM.2005.104) | [PDF](https://www.researchgate.net/publication/220291001_An_Experimental_Study_of_Soft_Errors_in_Microprocessors) | [DBLP](https://dblp.org/rec/journals/micro/SaggeseWKPI05.bib)

### Microprocessor Sensitivity to Failures: Control vs Execution and Combinational vs Sequential Logic
**Authors:** G.P. Saggese, A. Vetteth, Z. Kalbarczyk, R.K. Iyer
**Publication:** 2005 International Conference on Dependable Systems and Networks (DSN 2005), Yokohama, Japan, pp. 760-769, 2005
**Links:** [DOI](https://doi.org/10.1109/DSN.2005.63) | [DBLP](https://dblp.org/rec/conf/dsn/SaggeseVKI05.bib)

### An Architectural Framework for Detecting Process Hangs/Crashes
**Authors:** N. Nakka, G.P. Saggese, Z. Kalbarczyk, R.K. Iyer
**Publication:** Dependable Computing - EDCC-5, Budapest, Hungary, Lecture Notes in Computer Science, vol. 3463, pp. 103-121, Springer, 2005
**Links:** [DOI](https://doi.org/10.1007/11408901_8) | [PDF](http://www.crhc.uiuc.edu/~nakka/HCDetect.pdf) | [DBLP](https://dblp.org/rec/conf/edcc/NakkaSKI05.bib)

### Architecture and FPGA Implementation of a Digit-serial RSA Processor
**Authors:** A. Cilardo, A. Mazzeo, L. Romano, G.P. Saggese
**Publication:** New Algorithms, Architectures and Applications for Reconfigurable Computing, pp. 209-218, Springer, 2005

---

## 2004

### A Tamper Resistant Hardware Accelerator for RSA Cryptographic Applications
**Authors:** G.P. Saggese, L. Romano, N. Mazzocca, A. Mazzeo
**Publication:** Journal of Systems Architecture, vol. 50, no. 12, pp. 711-727, 2004
**Links:** [DOI](https://doi.org/10.1016/j.sysarc.2004.04.002) | [PDF](https://www.researchgate.net/publication/223251907_A_tamper_resistant_hardware_accelerator_for_RSA_cryptographic_applications) | [DBLP](https://dblp.org/rec/journals/jsa/SaggeseRMM04.bib)

### A Web Services Based Architecture for Digital Time Stamping
**Authors:** A. Cilardo, A. Mazzeo, L. Romano, G.P. Saggese, G. Cattaneo
**Publication:** Journal of Web Engineering, vol. 2, no. 3, pp. 148-175, 2004
**Links:** [URL](http://www.rintonpress.com/xjwe1/jwe-2-3/148-175.pdf) | [PDF](https://static.aminer.org/pdf/PDF/001/003/193/a_web_services_based_architecture_for_digital_time_stamping.pdf) | [DBLP](https://dblp.org/rec/journals/jwe/CilardoMRSC04.bib)

### Exploring the Design-Space for FPGA-Based Implementation of RSA
**Authors:** A. Cilardo, A. Mazzeo, L. Romano, G.P. Saggese
**Publication:** Microprocessors and Microsystems, vol. 28, no. 4, pp. 183-191, 2004
**Links:** [DOI](https://doi.org/10.1016/j.micpro.2004.03.009) | [DBLP](https://dblp.org/rec/journals/mam/CilardoMRS04.bib)
**Note:** ScienceDirect (paywalled)

### Carry-Save Montgomery Modular Exponentiation on Reconfigurable Hardware
**Authors:** A. Cilardo, A. Mazzeo, L. Romano, G.P. Saggese
**Publication:** 2004 Design, Automation and Test in Europe Conference and Exposition (DATE 2004), Paris, France, pp. 206-211, 2004
**Links:** [DOI](https://doi.org/10.1109/DATE.2004.1269231) | [PDF](https://static.aminer.org/pdf/PDF/000/141/620/carry_save_montgomery_modular_exponentiation_on_reconfigurable_hardware.pdf) | [DBLP](https://dblp.org/rec/conf/date/CilardoMRS04.bib)

### Hardware Support for High Performance, Intrusion- and Fault-Tolerant Systems
**Authors:** G.P. Saggese, C. Basile, L. Romano, Z. Kalbarczyk, R.K. Iyer
**Publication:** 23rd International Symposium on Reliable Distributed Systems (SRDS 2004), Florianópolis, Brazil, pp. 195-204, 2004
**Links:** [DOI](https://doi.org/10.1109/RELDIS.2004.1353020) | [DBLP](https://dblp.org/rec/conf/srds/SaggeseBRKI04.bib)

---

## 2003

### Using Programmable Hardware to Improve the Dependability of Cryptographic Applications (PhD Thesis)
**Author:** G.P. Saggese
**Publication:** PhD Thesis, University of Naples Federico II, Italy, 2003
**Links:** [URL](https://opac.bncf.firenze.sbn.it/bncf-prod/resource?uri=BNI0011112) | [DBLP](https://dblp.org/rec/phd/it/Saggese03.bib)

### FPGA-Based Implementation of a Serial RSA Processor
**Authors:** A. Mazzeo, L. Romano, G.P. Saggese, N. Mazzocca
**Publication:** 2003 Design, Automation and Test in Europe Conference and Exposition (DATE 2003), Munich, Germany, pp. 10582-10589, 2003
**Links:** [DOI](https://doi.org/10.1109/DATE.2003.10188) | [DBLP](https://dblp.org/rec/conf/date/MazzeoRSM03.bib)

### An FPGA-Based Performance Analysis of the Unrolling, Tiling, and Pipelining of the AES Algorithm
**Authors:** G.P. Saggese, A. Mazzeo, N. Mazzocca, A.G.M. Strollo
**Publication:** Field Programmable Logic and Application (FPL 2003), Lisbon, Portugal, Lecture Notes in Computer Science, vol. 2778, pp. 292-302, Springer, 2003
**Links:** [DOI](https://doi.org/10.1007/978-3-540-45234-8_29) | [DBLP](https://dblp.org/rec/conf/fpl/SaggeseMMS03.bib)
**Note:** Springer (paywalled). See also Semantic Scholar

### Using Web Services Technology for Inter-enterprise Integration of Digital Time Stamping
**Authors:** A. Cilardo, A. Mazzeo, L. Romano, G.P. Saggese, G. Cattaneo
**Publication:** On The Move to Meaningful Internet Systems 2003: OTM 2003 Workshops, Catania, Sicily, Italy, Lecture Notes in Computer Science, vol. 2889, pp. 960-974, Springer, 2003
**Links:** [DOI](https://doi.org/10.1007/978-3-540-39962-9_93) | [DBLP](https://dblp.org/rec/conf/otm/CilardoMRSC03.bib)

### Providing Digital Time Stamping Services to Mobile Devices
**Authors:** D. Cotroneo, C. di Flora, A. Mazzeo, L. Romano, S. Russo, G.P. Saggese
**Publication:** 9th IEEE International Workshop on Object-Oriented Real-Time Dependable Systems (WORDS Fall 2003), Anacapri (Capri Island), Italy, pp. 94-100, 2003
**Links:** [DOI](https://doi.org/10.1109/WORDS.2003.1267495) | [DBLP](https://dblp.org/rec/conf/words/CotroneoFMRRS03.bib)

### Providing Interoperable Time Stamping Services
**Authors:** A. Mazzeo, L. Romano, G.P. Saggese
**Publication:** Scuola Superiore, 2003

---

## 2002

### A Technique for FPGA Synthesis Driven by Automatic Source Code Analysis and Transformations
**Authors:** B. Di Martino, N. Mazzocca, G.P. Saggese, A.G.M. Strollo
**Publication:** Field-Programmable Logic and Applications (FPL 2002), Montpellier, France, Lecture Notes in Computer Science, vol. 2438, pp. 47-58, Springer, 2002
**Links:** [DOI](https://doi.org/10.1007/3-540-46117-5_7) | [DBLP](https://dblp.org/rec/conf/fpl/MartinoMSS02.bib)

### Shuffled Serial Adder: An Area-Latency Effective Serial Adder
**Authors:** G.P. Saggese, A.G.M. Strollo, N. Mazzocca, D. De Caro
**Publication:** Proceedings of the 2002 9th IEEE International Conference on Electronics, Circuits and Systems (ICECS 2002), Dubrovnik, Croatia, pp. 607-610, 2002
**Links:** [DOI](https://doi.org/10.1109/ICECS.2002.1046242) | [DBLP](https://dblp.org/rec/conf/icecsys/SaggeseSMC02.bib)

---

## 2001

### A Reconfigurable 2D Convolver for Real-Time SAR Imaging
**Authors:** A.G.M. Strollo, E. Napoli, D. De Caro, G.P. Saggese
**Publication:** Proceedings of the 2001 8th IEEE International Conference on Electronics, Circuits and Systems (ICECS 2001), Malta, pp. 741-744, 2001
**Links:** [DOI](https://doi.org/10.1109/ICECS.2001.957581) | [DBLP](https://dblp.org/rec/conf/icecsys/StrolloNCS01.bib)

### Test Pattern Generator for Hybrid Testing of Combinational Circuits
**Authors:** D. De Caro, N. Mazzocca, E. Napoli, G.P. Saggese, A.G.M. Strollo
**Publication:** Proceedings of the 2001 8th IEEE International Conference on Electronics, Circuits and Systems (ICECS 2001), Malta, pp. 745-748, 2001
**Links:** [DOI](https://doi.org/10.1109/ICECS.2001.957582) | [DBLP](https://dblp.org/rec/conf/icecsys/CaroMNSS01.bib)

---

## Additional Resources

- **DBLP Profile:** [https://dblp.org/pid/35/4554.html](https://dblp.org/pid/35/4554.html)
- **Research Areas:** Causal AI, Cryptography, Computer Arithmetic, Fault Tolerance, FPGA Design, Soft Error Analysis
- **BibTeX File:** All publications are available in BibTeX format in [gp_publications.bib](../gp_publications.bib)
