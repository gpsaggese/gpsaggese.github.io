# Research Interests

## Publications List
- [DBLP](https://dblp.uni-trier.de/pers/hd/s/Saggese:Giacinto_Paolo)
- [Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C21&q=gp+saggese&btnG=)
- [ResearchGate](https://www.researchgate.net/scientific-contributions/70377865_GP_Saggese)

## Research Areas

### [Causal AI](#causal-ai-publications)

**2015-present**: Developing production-grade Causal AI systems at Causify.AI,
combining Bayesian inference, knowledge graphs, and temporal machine learning for
decision-making under uncertainty. Applications include predictive maintenance,
failure prediction, demand forecasting, and energy optimization.

### [Cryptography](#cryptography-publications)

**2000-2005 (PhD work)**: hardware implementations of cryptographic algorithms,
focusing on RSA and AES acceleration, digital time stamping systems, and
tamper-resistant FPGA designs for secure cryptographic applications.

### [Computer Design](#computer-design-publications)

**2000-present**: Research on dependable computing systems, from soft error analysis
in microprocessors to low-power circuit design. Work includes fault tolerance
mechanisms, application-specific error detection, approximate computing, and
FPGA-based architectures for high-performance applications.

## Causal AI Publications

### Causify Benchmark: A Benchmark of Causal AI for Predictive Maintenance (2025)

**Authors:** K. Taduri, S. Dhande, G.P. Saggese, P. Smith

**Publication:** arXiv preprint arXiv:2512.01149, 2025

**Links:** [arXiv](https://arxiv.org/abs/2512.01149)

Comprehensive benchmark comparing Bayesian structural causal models against
correlation-based ML for predictive maintenance on 10,000 CNC machines. Causal AI
achieved $49,500 annual advantage over best ML baseline, 93.9% recall through
explicit failure mechanism modeling, and superior interpretability with
root-cause explanations.

### Causify DataMap: A Causal Probabilistic Reasoning (2025)

**Authors:** G.P. Saggese, P. Smith

**Publication:** Preprint, 2025

**Links:** [Preprint](https://drive.google.com/open?id=1A5PZy9TU_Ok4tCMNgt97zDKAh8LKSrpI&usp=drive_copy)

Automated system for generating causal probabilistic models by integrating
knowledge graphs, LLMs, and Bayesian inference. Converts unstructured knowledge
into executable models for reasoning and decision-making under uncertainty, with
applications to predictive maintenance.

### Causify DataFlow: A Causal Simulator for Stream Computing AI (2025)

**Authors:** G.P. Saggese, P. Smith

**Publication:** arXiv preprint arXiv:2512.23977, 2025

**Links:** [arXiv](https://arxiv.org/abs/2512.23977)

Computational framework for simulating causal models with time-series data using
DAG-based architecture with knowledge-time semantics. Guarantees temporal
correctness through "tileability" property, enabling unified batch/streaming
execution from research through production deployment.

### Causify DataPull: A Causal Data Layer for Time-series (2025)

**Authors:** G.P. Saggese, P. Smith

**Publication:** Preprint, 2025

**Links:** [Preprint](https://drive.google.com/open?id=1IwkjHdulKtsAusb3c4CunW3VxN5qndZm&usp=drive_copy)

Bitemporal data layer preserving both event time and knowledge time for
time-series data. Ensures causal models only use information available at
decision time, with integrated QA and automatic cleaning for reliable causal
analysis.

### Causify Sentinel: A Causal Failure Prediction Framework (2025)

**Authors:** C. Ma, S. Nikiforova, G.P. Saggese, P. Smith, K. Taduri

**Publication:** Preprint, 2025

**Links:** [Preprint](https://drive.google.com/open?id=1taqsgmdcyXO-gXS7Eog2gXtSDXg-8w0w&usp=drive_copy)

Causal failure prediction framework achieving 100% recall on wind turbine bearing
failures with weeks-to-months early warning. Combines physics-informed models
with propensity score weighting for confounder adjustment, plus state-space
forecasting for remaining useful life prediction.

### Causify Grid: A Causal Inference in Energy Demand Prediction (2025)

**Authors:** C. Ma, G.P. Saggese, P. Smith

**Publication:** arXiv preprint arXiv:2512.11653, 2025

**Links:** [arXiv](https://arxiv.org/abs/2512.11653)

Causal inference for energy demand prediction achieving 12.5% MAPE improvement
over non-causal baseline. Demonstrates 47.8% coefficient bias reduction through
backdoor criterion adjustment, with full Bayesian treatment using Pyro for
calibrated uncertainty quantification.

### Causify Horizon: A Causal Demand Forecasting Framework (2025)

**Authors:** C. Ma, G. Pomazkin, G.P. Saggese, P. Smith, D. Tikhomirov, N. Trubacheva

**Publication:** Manuscript to be submitted to arXiv, 2025

Causal demand forecasting framework integrating knowledge graphs with state-space
models for supply chain optimization. Combines LLM-assisted knowledge graph
construction with query-specific Bayesian networks, applied to spare parts demand
with calibrated uncertainty quantification for risk-aware inventory planning.

### Causify Optima: The Effect of Latency on Optimal Order Execution Policy (2025)

**Authors:** G.P. Saggese, P. Smith

**Publication:** arXiv preprint arXiv:2504.00846, 2025

**Links:** [arXiv](https://arxiv.org/abs/2504.00846)

Analysis of latency effects on optimal order execution policies in financial markets,
examining how execution delays impact trading strategies and market microstructure
considerations for algorithmic trading systems.

### Causify: AIgentic Development System (2025)

**Authors:** G.P. Saggese, P. Smith

**Publication:** Preprint, 2025

**Links:** [Preprint](https://drive.google.com/open?id=1IS8SaI1HMf2CX0AGKe9FmxQ_hghkKKMK&usp=drive_copy)

Development system leveraging AI agents for automated software engineering tasks,
integrating large language models with code generation, testing, and deployment
workflows for enhanced developer productivity.

### Causify: A Full-Stack Causal AI Framework (2025)

**Authors:** G.P. Saggese, P. Smith

**Publication:** Preprint, 2025

**Links:** [Preprint](https://drive.google.com/open?id=17-rS5RasQ6LdqcJkKkQAc_fRRIh11NAV&usp=drive_copy)

Comprehensive framework for building production-grade Causal AI systems, providing
end-to-end tools for causal model development, simulation, deployment, and
monitoring across diverse industrial applications.

### Runnable Directories: The Solution to the Monorepo vs. Multi-repo Debate (2025)

**Authors:** G.P. Saggese, P. Smith

**Publication:** arXiv preprint arXiv:2512.03815, 2025

**Links:** [arXiv](https://arxiv.org/abs/2512.03815)

Novel approach to code organization that resolves the monorepo versus multi-repo
trade-offs through self-contained, executable directory structures with explicit
dependencies and isolated runtime environments.

## Cryptography Publications

### A Tamper Resistant Hardware Accelerator for RSA Cryptographic Applications (2004)

**Authors:** G.P. Saggese, L. Romano, N. Mazzocca, A. Mazzeo

**Publication:** Journal of Systems Architecture, vol. 50, no. 12, pp. 711-727, 2004

**Links:** [PDF](papers/A_tamper_resistant_hardware_accelerator_for_RSA_cryptographic_applications_q.pdf) | [DOI](https://doi.org/10.1016/j.sysarc.2004.04.002) | [DBLP](https://dblp.org/rec/journals/jsa/SaggeseRMM04.bib)

### A Web Services Based Architecture for Digital Time Stamping (2004)

**Authors:** A. Cilardo, A. Mazzeo, L. Romano, G.P. Saggese, G. Cattaneo

**Publication:** Journal of Web Engineering, vol. 2, no. 3, pp. 148-175, 2004

**Links:** [DBLP](https://dblp.org/rec/journals/jwe/CilardoMRSC04.bib)

### Exploring the Design-Space for FPGA-Based Implementation of RSA (2004)

**Authors:** A. Cilardo, A. Mazzeo, L. Romano, G.P. Saggese

**Publication:** Microprocessors and Microsystems, vol. 28, no. 4, pp. 183-191, 2004

**Links:** [PDF](papers/Exploring_the_design-space_for_FPGA-based_implementation_of_RSA.pdf) | [DOI](https://doi.org/10.1016/j.micpro.2004.03.009) | [DBLP](https://dblp.org/rec/journals/mam/CilardoMRS04.bib)

### Carry-Save Montgomery Modular Exponentiation on Reconfigurable Hardware (2004)

**Authors:** A. Cilardo, A. Mazzeo, L. Romano, G.P. Saggese

**Publication:** 2004 Design, Automation and Test in Europe Conference and Exposition (DATE 2004), Paris, France, pp. 206-211, 2004

**Links:** [PDF](papers/Carry-save_Montgomery_modular_exponentiation_on_reconfigurable_hardware.pdf) | [DOI](https://doi.org/10.1109/DATE.2004.1269231) | [DBLP](https://dblp.org/rec/conf/date/CilardoMRS04.bib)

### Using Programmable Hardware to Improve the Dependability of Cryptographic Applications (2003)

**Author:** G.P. Saggese

**Publication:** PhD Thesis, University of Naples Federico II, Italy, 2003

**Links:** [DBLP](https://dblp.org/rec/phd/it/Saggese03.bib)

### FPGA-Based Implementation of a Serial RSA Processor (2003)

**Authors:** A. Mazzeo, L. Romano, G.P. Saggese, N. Mazzocca

**Publication:** 2003 Design, Automation and Test in Europe Conference and Exposition (DATE 2003), Munich, Germany, pp. 10582-10589, 2003

**Links:** [PDF](papers/FPGA-based_implementation_of_a_serial_RSA_processor.pdf) | [DOI](https://doi.org/10.1109/DATE.2003.10188) | [DBLP](https://dblp.org/rec/conf/date/MazzeoRSM03.bib)

### An FPGA-Based Performance Analysis of the Unrolling, Tiling, and Pipelining of the AES Algorithm (2003)

**Authors:** G.P. Saggese, A. Mazzeo, N. Mazzocca, A.G.M. Strollo

**Publication:** Field Programmable Logic and Application (FPL 2003), Lisbon, Portugal, Lecture Notes in Computer Science, vol. 2778, pp. 292-302, Springer, 2003

**Links:** [PDF](papers/An_FPGA-Based_Performance_Analysis_of_the_Unrolling,_Tiling,_and_Pipelining_of_the_AES_Algorithm.pdf) | [DOI](https://doi.org/10.1007/978-3-540-45234-8_29) | [DBLP](https://dblp.org/rec/conf/fpl/SaggeseMMS03.bib)

### Using Web Services Technology for Inter-enterprise Integration of Digital Time Stamping (2003)

**Authors:** A. Cilardo, A. Mazzeo, L. Romano, G.P. Saggese, G. Cattaneo

**Publication:** On The Move to Meaningful Internet Systems 2003: OTM 2003 Workshops, Catania, Sicily, Italy, Lecture Notes in Computer Science, vol. 2889, pp. 960-974, Springer, 2003

**Links:** [PDF](papers/Using_Web_Services_Technology_for_Inter-enterprise_Integration_of_Digital_Time_Stamping.pdf) | [DOI](https://doi.org/10.1007/978-3-540-39962-9_93) | [DBLP](https://dblp.org/rec/conf/otm/CilardoMRSC03.bib)

### Providing Digital Time Stamping Services to Mobile Devices (2003)

**Authors:** D. Cotroneo, C. di Flora, A. Mazzeo, L. Romano, S. Russo, G.P. Saggese

**Publication:** 9th IEEE International Workshop on Object-Oriented Real-Time Dependable Systems (WORDS Fall 2003), Anacapri (Capri Island), Italy, pp. 94-100, 2003

**Links:** [DOI](https://doi.org/10.1109/WORDS.2003.1267495) | [DBLP](https://dblp.org/rec/conf/words/CotroneoFMRRS03.bib)

### Providing Interoperable Time Stamping Services (2003)

**Authors:** A. Mazzeo, L. Romano, G.P. Saggese

**Publication:** Scuola Superiore, 2003

**Note:** PDF not available online

## Computer Design Publications
*Fault Tolerance, Computer Arithmetic, FPGA Design, Microprocessor Architecture*

### Automated Derivation of Application-Specific Error Detectors Using Dynamic Analysis (2011)

**Authors:** K. Pattabiraman, G.P. Saggese, D. Chen, Z. Kalbarczyk, R.K. Iyer

**Publication:** IEEE Transactions on Dependable and Secure Computing, vol. 8, no. 5, pp. 640-655, 2011

**Links:** [PDF](papers/Automated_Derivation_of_Application-Specific_Error_Detectors_Using_Dynamic_Analysis.pdf) | [DOI](https://doi.org/10.1109/TDSC.2010.19) | [DBLP](https://dblp.org/rec/journals/tdsc/PattabiramanSCKI11.bib)

### Dynamic Derivation of Application-Specific Error Detectors and their Implementation in Hardware (2006)

**Authors:** K. Pattabiraman, G.P. Saggese, D. Chen, Z. Kalbarczyk, R.K. Iyer

**Publication:** Sixth European Dependable Computing Conference (EDCC 2006), Coimbra, Portugal, pp. 97-108, 2006

**Links:** [PDF](papers/Dynamic_Derivation_of_Application-Specific_Error_Detectors_and_their_Implementation_in_Hardware.pdf) | [DOI](https://doi.org/10.1109/EDCC.2006.9) | [DBLP](https://dblp.org/rec/conf/edcc/PattabiramanSCKI06.bib)

### An Experimental Study of Soft Errors in Microprocessors (2005)

**Authors:** G.P. Saggese, N.J. Wang, Z. Kalbarczyk, S.J. Patel, R.K. Iyer

**Publication:** IEEE Micro, vol. 25, no. 6, pp. 30-39, 2005

**Links:** [PDF](papers/An_experimental_study_of_soft_errors_in_microprocessors.pdf) | [DOI](https://doi.org/10.1109/MM.2005.104) | [DBLP](https://dblp.org/rec/journals/micro/SaggeseWKPI05.bib)

### Microprocessor Sensitivity to Failures: Control vs Execution and Combinational vs Sequential Logic (2005)

**Authors:** G.P. Saggese, A. Vetteth, Z. Kalbarczyk, R.K. Iyer

**Publication:** 2005 International Conference on Dependable Systems and Networks (DSN 2005), Yokohama, Japan, pp. 760-769, 2005

**Links:** [PDF](papers/Microprocessor_sensitivity_to_failures_control_vs._execution_and_combinational_vs._sequential_logic.pdf) | [DOI](https://doi.org/10.1109/DSN.2005.63) | [DBLP](https://dblp.org/rec/conf/dsn/SaggeseVKI05.bib)

### An Architectural Framework for Detecting Process Hangs/Crashes (2005)

**Authors:** N. Nakka, G.P. Saggese, Z. Kalbarczyk, R.K. Iyer

**Publication:** Dependable Computing - EDCC-5, Budapest, Hungary, Lecture Notes in Computer Science, vol. 3463, pp. 103-121, Springer, 2005

**Links:** [PDF](papers/An_Architectural_Framework_for_Detecting_Process_Hangs-Crashes.pdf) | [DOI](https://doi.org/10.1007/11408901_8) | [DBLP](https://dblp.org/rec/conf/edcc/NakkaSKI05.bib)

### Architecture and FPGA Implementation of a Digit-serial RSA Processor (2005)

**Authors:** A. Cilardo, A. Mazzeo, L. Romano, G.P. Saggese

**Publication:** New Algorithms, Architectures and Applications for Reconfigurable Computing, pp. 209-218, Springer, 2005

**Links:** [PDF](papers/Architecture_and_FPGA_Implementation_of_a_Digit-serial_RSA_Processor.pdf)

### Hardware Support for High Performance, Intrusion- and Fault-Tolerant Systems (2004)

**Authors:** G.P. Saggese, C. Basile, L. Romano, Z. Kalbarczyk, R.K. Iyer

**Publication:** 23rd International Symposium on Reliable Distributed Systems (SRDS 2004), Florianópolis, Brazil, pp. 195-204, 2004

**Links:** [PDF](papers/Hardware_support_for_high_performance_intrusion-_and_fault-tolerant_systems.pdf) | [DOI](https://doi.org/10.1109/RELDIS.2004.1353020) | [DBLP](https://dblp.org/rec/conf/srds/SaggeseBRKI04.bib)

### A Technique for FPGA Synthesis Driven by Automatic Source Code Analysis and Transformations (2002)

**Authors:** B. Di Martino, N. Mazzocca, G.P. Saggese, A.G.M. Strollo

**Publication:** Field-Programmable Logic and Applications (FPL 2002), Montpellier, France, Lecture Notes in Computer Science, vol. 2438, pp. 47-58, Springer, 2002

**Links:** [PDF](papers/A_Technique_for_FPGA_Synthesis_Driven_by_Automatic_Source_Code_Analysis_and_Transformations.pdf) | [DOI](https://doi.org/10.1007/3-540-46117-5_7) | [DBLP](https://dblp.org/rec/conf/fpl/MartinoMSS02.bib)

### Shuffled Serial Adder: An Area-Latency Effective Serial Adder (2002)

**Authors:** G.P. Saggese, A.G.M. Strollo, N. Mazzocca, D. De Caro

**Publication:** Proceedings of the 2002 9th IEEE International Conference on Electronics, Circuits and Systems (ICECS 2002), Dubrovnik, Croatia, pp. 607-610, 2002

**Links:** [PDF](papers/Shuffled_serial_adder_an_area-latency_effective_serial_adder.pdf) | [DOI](https://doi.org/10.1109/ICECS.2002.1046242) | [DBLP](https://dblp.org/rec/conf/icecsys/SaggeseSMC02.bib)

### A Reconfigurable 2D Convolver for Real-Time SAR Imaging (2001)

**Authors:** A.G.M. Strollo, E. Napoli, D. De Caro, G.P. Saggese

**Publication:** Proceedings of the 2001 8th IEEE International Conference on Electronics, Circuits and Systems (ICECS 2001), Malta, pp. 741-744, 2001

**Links:** [PDF](papers/A_reconfigurable_2D_convolver_for_real-time_SAR_imaging.pdf) | [DOI](https://doi.org/10.1109/ICECS.2001.957581) | [DBLP](https://dblp.org/rec/conf/icecsys/StrolloNCS01.bib)

### Test Pattern Generator for Hybrid Testing of Combinational Circuits (2001)

**Authors:** D. De Caro, N. Mazzocca, E. Napoli, G.P. Saggese, A.G.M. Strollo

**Publication:** Proceedings of the 2001 8th IEEE International Conference on Electronics, Circuits and Systems (ICECS 2001), Malta, pp. 745-748, 2001

**Links:** [PDF](papers/Test_pattern_generator_for_hybrid_testing_of_combinational_circuits.pdf) | [DOI](https://doi.org/10.1109/ICECS.2001.957582) | [DBLP](https://dblp.org/rec/conf/icecsys/CaroMNSS01.bib)
