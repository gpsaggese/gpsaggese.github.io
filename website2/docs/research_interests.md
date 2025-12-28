# Research Interests & Publications

## Research Areas

### [Causal AI](#causal-ai-publications)

2015-present: Developing production-grade Causal AI systems at Causify.AI,
combining Bayesian inference, knowledge graphs, and temporal machine learning for
decision-making under uncertainty. Applications include predictive maintenance,
failure prediction, demand forecasting, and energy optimization.

### [Cryptography](#cryptography-publications)

2000-2005 (PhD work): hardware implementations of cryptographic algorithms,
focusing on RSA and AES acceleration, digital time stamping systems, and
tamper-resistant FPGA designs for secure cryptographic applications.

### [Computer Design](#computer-design-publications)

2000-present: Research on dependable computing systems, from soft error analysis
in microprocessors to low-power circuit design. Work includes fault tolerance
mechanisms, application-specific error detection, approximate computing, and
FPGA-based architectures for high-performance applications.

## Causal AI Publications

### Causify Benchmark: A Benchmark of Causal AI for Predictive Maintenance (2025)
**Authors:** K. Taduri, S. Dhande, G.P. Saggese, P. Smith
**Publication:** arXiv preprint arXiv:2512.01149, 2025
**Links:** [arXiv](https://arxiv.org/abs/2512.01149)

Comprehensive benchmark comparing Bayesian structural causal models against correlation-based ML for predictive maintenance on 10,000 CNC machines. Causal AI achieved $49,500 annual advantage over best ML baseline, 93.9% recall through explicit failure mechanism modeling, and superior interpretability with root-cause explanations.

### Causify DataMap: Causal Probabilistic Reasoning (2025)
**Authors:** G.P. Saggese, P. Smith
**Publication:** Manuscript to be submitted to arXiv, 2025

Automated system for generating causal probabilistic models by integrating knowledge graphs, LLMs, and Bayesian inference. Converts unstructured knowledge into executable models for reasoning and decision-making under uncertainty, with applications to predictive maintenance.

### Causify DataFlow: A Causal Model Simulator (2025)
**Authors:** G.P. Saggese, P. Smith
**Publication:** Manuscript to be submitted to arXiv, 2025

Computational framework for simulating causal models with time-series data using DAG-based architecture with knowledge-time semantics. Guarantees temporal correctness through "tileability" property, enabling unified batch/streaming execution from research through production deployment.

### Causify DataPull: A Causal Data Layer (2025)
**Authors:** G.P. Saggese, P. Smith
**Publication:** Manuscript to be submitted to arXiv, 2025

Bitemporal data layer preserving both event time and knowledge time for time-series data. Ensures causal models only use information available at decision time, with integrated QA and automatic cleaning for reliable causal analysis.

### Causify Sentinel: A Causal Failure Prediction Framework (2025)
**Authors:** C. Ma, S. Nikiforova, G.P. Saggese, P. Smith, K. Taduri
**Publication:** Manuscript to be submitted to arXiv, 2025

Causal failure prediction framework achieving 100% recall on wind turbine bearing failures with weeks-to-months early warning. Combines physics-informed models with propensity score weighting for confounder adjustment, plus state-space forecasting for remaining useful life prediction.

### Causify Grid: Causal Inference in Energy Demand Prediction (2025)
**Authors:** C. Ma, G.P. Saggese, P. Smith
**Publication:** Manuscript to be submitted to arXiv, 2025

Causal inference for energy demand prediction achieving 12.5% MAPE improvement over non-causal baseline. Demonstrates 47.8% coefficient bias reduction through backdoor criterion adjustment, with full Bayesian treatment using Pyro for calibrated uncertainty quantification.

### Causify Horizon: A Causal Demand Forecasting Framework (2025)
**Authors:** C. Ma, G. Pomazkin, G.P. Saggese, P. Smith, D. Tikhomirov, N. Trubacheva
**Publication:** Manuscript to be submitted to arXiv, 2025

Causal demand forecasting framework integrating knowledge graphs with state-space models for supply chain optimization. Combines LLM-assisted knowledge graph construction with query-specific Bayesian networks, applied to spare parts demand with calibrated uncertainty quantification for risk-aware inventory planning.

## Cryptography Publications

### A Tamper Resistant Hardware Accelerator for RSA Cryptographic Applications (2004)
**Authors:** G.P. Saggese, L. Romano, N. Mazzocca, A. Mazzeo
**Publication:** Journal of Systems Architecture, vol. 50, no. 12, pp. 711-727, 2004
**Links:** [DOI](https://doi.org/10.1016/j.sysarc.2004.04.002) | [PDF](https://www.researchgate.net/publication/223251907_A_tamper_resistant_hardware_accelerator_for_RSA_cryptographic_applications) | [DBLP](https://dblp.org/rec/journals/jsa/SaggeseRMM04.bib)

### A Web Services Based Architecture for Digital Time Stamping (2004)
**Authors:** A. Cilardo, A. Mazzeo, L. Romano, G.P. Saggese, G. Cattaneo
**Publication:** Journal of Web Engineering, vol. 2, no. 3, pp. 148-175, 2004
**Links:** [URL](http://www.rintonpress.com/xjwe1/jwe-2-3/148-175.pdf) | [PDF](https://static.aminer.org/pdf/PDF/001/003/193/a_web_services_based_architecture_for_digital_time_stamping.pdf) | [DBLP](https://dblp.org/rec/journals/jwe/CilardoMRSC04.bib)

### Exploring the Design-Space for FPGA-Based Implementation of RSA (2004)
**Authors:** A. Cilardo, A. Mazzeo, L. Romano, G.P. Saggese
**Publication:** Microprocessors and Microsystems, vol. 28, no. 4, pp. 183-191, 2004
**Links:** [DOI](https://doi.org/10.1016/j.micpro.2004.03.009) | [DBLP](https://dblp.org/rec/journals/mam/CilardoMRS04.bib)
**Note:** ScienceDirect (paywalled)

### Carry-Save Montgomery Modular Exponentiation on Reconfigurable Hardware (2004)
**Authors:** A. Cilardo, A. Mazzeo, L. Romano, G.P. Saggese
**Publication:** 2004 Design, Automation and Test in Europe Conference and Exposition (DATE 2004), Paris, France, pp. 206-211, 2004
**Links:** [DOI](https://doi.org/10.1109/DATE.2004.1269231) | [PDF](https://static.aminer.org/pdf/PDF/000/141/620/carry_save_montgomery_modular_exponentiation_on_reconfigurable_hardware.pdf) | [DBLP](https://dblp.org/rec/conf/date/CilardoMRS04.bib)

### Using Programmable Hardware to Improve the Dependability of Cryptographic Applications (2003)
**Author:** G.P. Saggese
**Publication:** PhD Thesis, University of Naples Federico II, Italy, 2003
**Links:** [URL](https://opac.bncf.firenze.sbn.it/bncf-prod/resource?uri=BNI0011112) | [DBLP](https://dblp.org/rec/phd/it/Saggese03.bib)

### FPGA-Based Implementation of a Serial RSA Processor (2003)
**Authors:** A. Mazzeo, L. Romano, G.P. Saggese, N. Mazzocca
**Publication:** 2003 Design, Automation and Test in Europe Conference and Exposition (DATE 2003), Munich, Germany, pp. 10582-10589, 2003
**Links:** [DOI](https://doi.org/10.1109/DATE.2003.10188) | [DBLP](https://dblp.org/rec/conf/date/MazzeoRSM03.bib)

### An FPGA-Based Performance Analysis of the Unrolling, Tiling, and Pipelining of the AES Algorithm (2003)
**Authors:** G.P. Saggese, A. Mazzeo, N. Mazzocca, A.G.M. Strollo
**Publication:** Field Programmable Logic and Application (FPL 2003), Lisbon, Portugal, Lecture Notes in Computer Science, vol. 2778, pp. 292-302, Springer, 2003
**Links:** [DOI](https://doi.org/10.1007/978-3-540-45234-8_29) | [DBLP](https://dblp.org/rec/conf/fpl/SaggeseMMS03.bib)
**Note:** Springer (paywalled). See also Semantic Scholar

### Using Web Services Technology for Inter-enterprise Integration of Digital Time Stamping (2003)
**Authors:** A. Cilardo, A. Mazzeo, L. Romano, G.P. Saggese, G. Cattaneo
**Publication:** On The Move to Meaningful Internet Systems 2003: OTM 2003 Workshops, Catania, Sicily, Italy, Lecture Notes in Computer Science, vol. 2889, pp. 960-974, Springer, 2003
**Links:** [DOI](https://doi.org/10.1007/978-3-540-39962-9_93) | [DBLP](https://dblp.org/rec/conf/otm/CilardoMRSC03.bib)

### Providing Digital Time Stamping Services to Mobile Devices (2003)
**Authors:** D. Cotroneo, C. di Flora, A. Mazzeo, L. Romano, S. Russo, G.P. Saggese
**Publication:** 9th IEEE International Workshop on Object-Oriented Real-Time Dependable Systems (WORDS Fall 2003), Anacapri (Capri Island), Italy, pp. 94-100, 2003
**Links:** [DOI](https://doi.org/10.1109/WORDS.2003.1267495) | [DBLP](https://dblp.org/rec/conf/words/CotroneoFMRRS03.bib)

### Providing Interoperable Time Stamping Services (2003)
**Authors:** A. Mazzeo, L. Romano, G.P. Saggese
**Publication:** Scuola Superiore, 2003

## Computer Design Publications
*Fault Tolerance, Computer Arithmetic, FPGA Design, Microprocessor Architecture*

### A Novel Module-Sign Low-Power Implementation for the DLMS Adaptive Filter With Low Steady-State Error (2022)
**Authors:** G. Di Meo, D. De Caro, G.P. Saggese, E. Napoli, N. Petra, A.G.M. Strollo
**Publication:** IEEE Transactions on Circuits and Systems I: Regular Papers, vol. 69, no. 1, pp. 297-308, 2022
**Links:** [DOI](https://doi.org/10.1109/TCSI.2021.3088913) | [DBLP](https://dblp.org/rec/journals/tcasI/MeoCSNPS22.bib)
**Note:** PDF not freely available

### Approximate Multipliers Using Static Segmentation: Error Analysis and Improvements (2022)
**Authors:** A.G.M. Strollo, E. Napoli, D. De Caro, N. Petra, G.P. Saggese, G. Di Meo
**Publication:** IEEE Transactions on Circuits and Systems I: Regular Papers, vol. 69, no. 6, pp. 2449-2462, 2022
**Links:** [DOI](https://doi.org/10.1109/TCSI.2022.3152921) | [PDF](https://ieeexplore.ieee.org/iel7/8919/9782465/09726786.pdf) | [DBLP](https://dblp.org/rec/journals/tcasI/StrolloNCPSM22.bib)

### Automated Derivation of Application-Specific Error Detectors Using Dynamic Analysis (2011)
**Authors:** K. Pattabiraman, G.P. Saggese, D. Chen, Z. Kalbarczyk, R.K. Iyer
**Publication:** IEEE Transactions on Dependable and Secure Computing, vol. 8, no. 5, pp. 640-655, 2011
**Links:** [DOI](https://doi.org/10.1109/TDSC.2010.19) | [PDF](https://www.researchgate.net/publication/220068507_Automated_Derivation_of_Application-Specific_Error_Detectors_Using_Dynamic_Analysis) | [DBLP](https://dblp.org/rec/journals/tdsc/PattabiramanSCKI11.bib)

### Dynamic Derivation of Application-Specific Error Detectors and their Implementation in Hardware (2006)
**Authors:** K. Pattabiraman, G.P. Saggese, D. Chen, Z. Kalbarczyk, R.K. Iyer
**Publication:** Sixth European Dependable Computing Conference (EDCC 2006), Coimbra, Portugal, pp. 97-108, 2006
**Links:** [DOI](https://doi.org/10.1109/EDCC.2006.9) | [PDF](https://tcipg.org/sites/default/files/papers/2006_Pattabiraman_et_al.pdf) | [DBLP](https://dblp.org/rec/conf/edcc/PattabiramanSCKI06.bib)

### An Experimental Study of Soft Errors in Microprocessors (2005)
**Authors:** G.P. Saggese, N.J. Wang, Z. Kalbarczyk, S.J. Patel, R.K. Iyer
**Publication:** IEEE Micro, vol. 25, no. 6, pp. 30-39, 2005
**Links:** [DOI](https://doi.org/10.1109/MM.2005.104) | [PDF](https://www.researchgate.net/publication/220291001_An_Experimental_Study_of_Soft_Errors_in_Microprocessors) | [DBLP](https://dblp.org/rec/journals/micro/SaggeseWKPI05.bib)

### Microprocessor Sensitivity to Failures: Control vs Execution and Combinational vs Sequential Logic (2005)
**Authors:** G.P. Saggese, A. Vetteth, Z. Kalbarczyk, R.K. Iyer
**Publication:** 2005 International Conference on Dependable Systems and Networks (DSN 2005), Yokohama, Japan, pp. 760-769, 2005
**Links:** [DOI](https://doi.org/10.1109/DSN.2005.63) | [DBLP](https://dblp.org/rec/conf/dsn/SaggeseVKI05.bib)

### An Architectural Framework for Detecting Process Hangs/Crashes (2005)
**Authors:** N. Nakka, G.P. Saggese, Z. Kalbarczyk, R.K. Iyer
**Publication:** Dependable Computing - EDCC-5, Budapest, Hungary, Lecture Notes in Computer Science, vol. 3463, pp. 103-121, Springer, 2005
**Links:** [DOI](https://doi.org/10.1007/11408901_8) | [PDF](http://www.crhc.uiuc.edu/~nakka/HCDetect.pdf) | [DBLP](https://dblp.org/rec/conf/edcc/NakkaSKI05.bib)

### Architecture and FPGA Implementation of a Digit-serial RSA Processor (2005)
**Authors:** A. Cilardo, A. Mazzeo, L. Romano, G.P. Saggese
**Publication:** New Algorithms, Architectures and Applications for Reconfigurable Computing, pp. 209-218, Springer, 2005

### Hardware Support for High Performance, Intrusion- and Fault-Tolerant Systems (2004)
**Authors:** G.P. Saggese, C. Basile, L. Romano, Z. Kalbarczyk, R.K. Iyer
**Publication:** 23rd International Symposium on Reliable Distributed Systems (SRDS 2004), Florianópolis, Brazil, pp. 195-204, 2004
**Links:** [DOI](https://doi.org/10.1109/RELDIS.2004.1353020) | [DBLP](https://dblp.org/rec/conf/srds/SaggeseBRKI04.bib)

### A Technique for FPGA Synthesis Driven by Automatic Source Code Analysis and Transformations (2002)
**Authors:** B. Di Martino, N. Mazzocca, G.P. Saggese, A.G.M. Strollo
**Publication:** Field-Programmable Logic and Applications (FPL 2002), Montpellier, France, Lecture Notes in Computer Science, vol. 2438, pp. 47-58, Springer, 2002
**Links:** [DOI](https://doi.org/10.1007/3-540-46117-5_7) | [DBLP](https://dblp.org/rec/conf/fpl/MartinoMSS02.bib)

### Shuffled Serial Adder: An Area-Latency Effective Serial Adder (2002)
**Authors:** G.P. Saggese, A.G.M. Strollo, N. Mazzocca, D. De Caro
**Publication:** Proceedings of the 2002 9th IEEE International Conference on Electronics, Circuits and Systems (ICECS 2002), Dubrovnik, Croatia, pp. 607-610, 2002
**Links:** [DOI](https://doi.org/10.1109/ICECS.2002.1046242) | [DBLP](https://dblp.org/rec/conf/icecsys/SaggeseSMC02.bib)

### A Reconfigurable 2D Convolver for Real-Time SAR Imaging (2001)
**Authors:** A.G.M. Strollo, E. Napoli, D. De Caro, G.P. Saggese
**Publication:** Proceedings of the 2001 8th IEEE International Conference on Electronics, Circuits and Systems (ICECS 2001), Malta, pp. 741-744, 2001
**Links:** [DOI](https://doi.org/10.1109/ICECS.2001.957581) | [DBLP](https://dblp.org/rec/conf/icecsys/StrolloNCS01.bib)

### Test Pattern Generator for Hybrid Testing of Combinational Circuits (2001)
**Authors:** D. De Caro, N. Mazzocca, E. Napoli, G.P. Saggese, A.G.M. Strollo
**Publication:** Proceedings of the 2001 8th IEEE International Conference on Electronics, Circuits and Systems (ICECS 2001), Malta, pp. 745-748, 2001
**Links:** [DOI](https://doi.org/10.1109/ICECS.2001.957582) | [DBLP](https://dblp.org/rec/conf/icecsys/CaroMNSS01.bib)
