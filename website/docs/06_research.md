# Research Interests

## Publications List

- [DBLP](https://dblp.uni-trier.de/pers/hd/s/Saggese:Giacinto_Paolo)
- [Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C21&q=gp+saggese&btnG=)
- [ResearchGate](https://www.researchgate.net/scientific-contributions/70377865_GP_Saggese)

## Research Areas

### [Causal AI](#causal-ai-publications)

**2015-present**: Developing production-grade Causal AI systems at Causify.AI,
combining Bayesian inference, knowledge graphs, and temporal machine learning
for decision-making under uncertainty. Applications include predictive
maintenance, failure prediction, demand forecasting, and energy optimization.

### [Cryptography](#cryptography-publications)

**2000-2005 (PhD work)**: hardware implementations of cryptographic algorithms,
focusing on RSA and AES acceleration, digital time stamping systems, and
tamper-resistant FPGA designs for secure cryptographic applications.

### [Computer Design](#computer-design-publications)

**2000-present**: Research on dependable computing systems, from soft error
analysis in microprocessors to low-power circuit design. Work includes fault
tolerance mechanisms, application-specific error detection, approximate
computing, and FPGA-based architectures for high-performance applications.

## Causal AI Publications

### Causify Causal Technology Stack (2025)

**Authors:** G.P. Saggese, P. Smith

**Publication:** Preprint, 2025

**Links:**
[Preprint](https://drive.google.com/file/d/1GRIqZSvd6k1iC8SfK2xpT7bKhd5B-8i2/view?usp=drive_link)

- Technology stack overview: Comprehensive documentation of Causify's
  intellectual property covering core technologies (DataFlow, DataMap,
  DataPull), vertical applications (Grid, Horizon, Sentinel, Optima, Benchmark),
  and AI/ML systems developed over two years
- System architecture: Presents complete framework for production-grade Causal
  AI, integrating Bayesian inference, knowledge graphs, and temporal machine
  learning for decision-making under uncertainty
- Development infrastructure: Includes AIgentic Development system, Full-Stack
  Framework, and Runnable Directories enabling end-to-end causal AI application
  development and deployment

### Beyond Accuracy: A Stability-Aware Metric for Multi-Horizon Forecasting (2026)

**Authors:** C. Ma, G. Pomazkin, G.P. Saggese, P. Smith

**Publication:** arXiv preprint arXiv:2601.10863, 2026

**Links:** [arXiv](https://arxiv.org/abs/2601.10863)

- Novel evaluation metric: Introduces "forecast accuracy and coherence score"
  (AC score) for assessing probabilistic forecasts across multiple time
  horizons, combining both prediction accuracy and temporal consistency in a
  single measure
- Stability-aware approach: Ensures forecasts remain stable as the starting
  point shifts, addressing the critical problem of forecast volatility that
  affects decision-making in production systems
- Empirical validation: Demonstrates 75% reduction in forecast volatility on M4
  Hourly benchmark while maintaining comparable or improved point forecast
  accuracy through differentiable optimization of seasonal ARIMA models

### Causify Benchmark: A Benchmark of Causal AI for Predictive Maintenance (2025)

**Authors:** K. Taduri, S. Dhande, G.P. Saggese, P. Smith

**Publication:** arXiv preprint arXiv:2512.01149, 2025

**Links:** [arXiv](https://arxiv.org/abs/2512.01149)

- Benchmark design: Comprehensive comparison of Bayesian structural causal
  models against correlation-based machine learning for predictive maintenance
  using real-world dataset of 10,000 CNC machines
- Performance advantages: Causal AI achieved $49,500 annual economic advantage
  over best ML baseline with 93.9% recall through explicit modeling of failure
  mechanisms rather than relying on statistical correlations
- Interpretability focus: Demonstrates superior explainability through
  root-cause analysis and causal explanations, enabling actionable maintenance
  decisions compared to black-box ML approaches

### Causify DataMap: A Causal Probabilistic Reasoning (2025)

**Authors:** G.P. Saggese, P. Smith

**Publication:** Preprint, 2025

**Links:**
[Preprint](https://drive.google.com/open?id=1A5PZy9TU_Ok4tCMNgt97zDKAh8LKSrpI&usp=drive_copy)

- Automated model generation: System that automatically generates causal
  probabilistic models by integrating knowledge graphs, large language models,
  and Bayesian inference for domain-specific reasoning
- Knowledge transformation: Converts unstructured domain knowledge into
  executable causal models with probabilistic semantics, enabling formal
  reasoning and decision-making under uncertainty
- Application focus: Demonstrates practical deployment in predictive maintenance
  scenarios where expert knowledge must be combined with data-driven inference
  for reliable failure prediction

### Causify DataFlow: A Causal Simulator for Stream Computing AI (2025)

**Authors:** G.P. Saggese, P. Smith

**Publication:** arXiv preprint arXiv:2512.23977, 2025

**Links:** [arXiv](https://arxiv.org/abs/2512.23977)

- DAG-based architecture: Computational framework for simulating causal models
  with time-series data using directed acyclic graph architecture enhanced with
  knowledge-time semantics for temporal causal reasoning
- Temporal correctness: Guarantees proper temporal ordering through
  "tileability" property, ensuring causal models respect information
  availability constraints and avoid look-ahead bias in streaming contexts
- Unified execution: Enables seamless transition from research to production
  through unified batch and streaming execution model, eliminating need for
  separate implementation paths

### Causify DataPull: A Causal Data Layer for Time-series (2025)

**Authors:** G.P. Saggese, P. Smith

**Publication:** Preprint, 2025

**Links:**
[Preprint](https://drive.google.com/open?id=1IwkjHdulKtsAusb3c4CunW3VxN5qndZm&usp=drive_copy)

- Bitemporal architecture: Data layer preserving both event time (when events
  occurred) and knowledge time (when information became available) to maintain
  proper causal semantics in time-series analysis
- Causal correctness: Ensures causal models only use information that was
  actually available at decision time, preventing look-ahead bias and enabling
  faithful backtesting of predictive systems
- Data quality integration: Incorporates integrated quality assurance and
  automatic data cleaning mechanisms to ensure reliability and consistency for
  downstream causal analysis applications

### Causify Sentinel: A Causal Failure Prediction Framework (2025)

**Authors:** C. Ma, S. Nikiforova, G.P. Saggese, P. Smith, K. Taduri

**Publication:** Preprint, 2025

**Links:**
[Preprint](https://drive.google.com/open?id=1taqsgmdcyXO-gXS7Eog2gXtSDXg-8w0w&usp=drive_copy)

- Causal framework: Failure prediction system combining physics-informed models
  with propensity score weighting for confounder adjustment, enabling accurate
  identification of impending equipment failures
- Early warning performance: Achieves 100% recall on wind turbine bearing
  failures with weeks-to-months advance warning, significantly outperforming
  correlation-based approaches through explicit causal modeling
- Remaining useful life: Integrates state-space forecasting models to predict
  remaining useful life after failure detection, enabling optimized maintenance
  scheduling and resource allocation

### Causify Grid: A Causal Inference in Energy Demand Prediction (2025)

**Authors:** C. Ma, G.P. Saggese, P. Smith

**Publication:** arXiv preprint arXiv:2512.11653, 2025

**Links:** [arXiv](https://arxiv.org/abs/2512.11653)

- Causal inference approach: Application of causal inference methods to energy
  demand prediction, addressing confounding through backdoor criterion
  adjustment to identify true causal relationships between weather and
  consumption
- Bias reduction: Demonstrates 47.8% coefficient bias reduction compared to
  naive correlation-based models, revealing that traditional approaches
  systematically overestimate weather effects due to uncontrolled confounders
- Bayesian uncertainty: Implements full Bayesian treatment using Pyro
  probabilistic programming for calibrated uncertainty quantification, achieving
  12.5% MAPE improvement over non-causal baselines

### Causify Horizon: A Causal Demand Forecasting Framework (2025)

**Authors:** C. Ma, G. Pomazkin, G.P. Saggese, P. Smith, D. Tikhomirov, N.
Trubacheva

**Publication:** Preprint, 2025

**Links:**
[Preprint](https://drive.google.com/file/d/1FlxErQG4nCDCDOppBmyelHQ-vwe0JN_c/view?usp=drive_link)

- Knowledge-informed forecasting: Causal framework integrating domain knowledge
  graphs with state-space models for demand forecasting, combining LLM-assisted
  knowledge extraction with query-specific Bayesian network construction
- Supply chain application: Applied to spare parts demand prediction where
  domain expertise about equipment relationships, failure modes, and operational
  patterns significantly improves forecast accuracy over pure data-driven
  approaches
- Risk-aware planning: Provides calibrated probabilistic forecasts with explicit
  uncertainty quantification, enabling risk-aware inventory optimization and
  resource allocation in supply chain management

### Causify Optima: The Effect of Latency on Optimal Order Execution Policy (2025)

**Authors:** G.P. Saggese, P. Smith

**Publication:** arXiv preprint arXiv:2504.00846, 2025

**Links:** [arXiv](https://arxiv.org/abs/2504.00846)

- Latency analysis: Examines how execution delays in financial markets affect
  optimal order execution policies, quantifying the impact of latency on trading
  performance and strategy effectiveness
- Market microstructure: Analyzes market microstructure considerations including
  price impact, adverse selection, and timing risk in the presence of execution
  delays for algorithmic trading systems
- Policy optimization: Develops optimal execution strategies that explicitly
  account for latency constraints, providing practical guidance for designing
  robust algorithmic trading systems in realistic market conditions

### Causify: AIgentic Development System (2025)

**Authors:** G.P. Saggese, P. Smith

**Publication:** Preprint, 2025

**Links:**
[Preprint](https://drive.google.com/open?id=1IS8SaI1HMf2CX0AGKe9FmxQ_hghkKKMK&usp=drive_copy)

- AI-powered development: System leveraging AI agents and large language models
  to automate software engineering tasks including code generation, refactoring,
  testing, and documentation across the development lifecycle
- Workflow integration: Seamlessly integrates with existing development
  workflows and tools, providing automated assistance for routine programming
  tasks while maintaining human oversight for critical decisions
- Productivity enhancement: Demonstrates significant improvements in developer
  productivity through intelligent automation of repetitive tasks, enabling
  developers to focus on high-level design and problem-solving

### Causify: A Full-Stack Causal AI Framework (2025)

**Authors:** G.P. Saggese, P. Smith

**Publication:** Preprint, 2025

**Links:**
[Preprint](https://drive.google.com/open?id=17-rS5RasQ6LdqcJkKkQAc_fRRIh11NAV&usp=drive_copy)

- Full-stack architecture: Comprehensive framework providing end-to-end
  infrastructure for building production-grade Causal AI systems, from research
  and development through deployment and monitoring
- Development tools: Integrated toolchain for causal model development,
  simulation, validation, and deployment, enabling rapid prototyping and
  seamless transition from research to production
- Industrial applications: Designed for diverse industrial use cases including
  predictive maintenance, demand forecasting, failure prediction, and
  optimization problems requiring causal reasoning under uncertainty

### Runnable Directories: The Solution to the Monorepo vs. Multi-repo Debate (2025)

**Authors:** G.P. Saggese, P. Smith

**Publication:** arXiv preprint arXiv:2512.03815, 2025

**Links:** [arXiv](https://arxiv.org/abs/2512.03815)

- Code organization paradigm: Novel approach resolving monorepo vs multi-repo
  trade-offs through self-contained, executable directory structures that
  combine benefits of both approaches
- Dependency management: Each directory explicitly declares dependencies and
  maintains isolated runtime environment, enabling independent development while
  preserving code sharing and atomic refactoring capabilities
- Practical deployment: Demonstrates how runnable directories simplify build
  systems, testing, deployment, and version management while maintaining
  reproducibility and modularity in large-scale software projects

## Cryptography Publications

### A Tamper Resistant Hardware Accelerator for RSA Cryptographic Applications (2004)

**Authors:** G.P. Saggese, L. Romano, N. Mazzocca, A. Mazzeo

**Publication:** Journal of Systems Architecture, vol. 50, no. 12, pp. 711-727,
2004

**Links:**
[PDF](https://gpsaggese.github.io/umd_classes/papers/A_tamper_resistant_hardware_accelerator_for_RSA_cryptographic_applications_q.pdf)
| [DOI](https://doi.org/10.1016/j.sysarc.2004.04.002) |
[DBLP](https://dblp.org/rec/journals/jsa/SaggeseRMM04.bib)

- Hardware accelerator design: Presents a hardware accelerator integrating an
  RSA processor and RSA key-store to improve both security and performance of
  RSA cryptographic applications
- Implementation approach: FPGA-based implementation using Commercial
  Off-The-Shelf (COTS) programmable hardware, providing architectural solutions
  that maximize security/performance while minimizing hardware resource costs
- Key contributions: Describes functional blocks and interactions, evaluates
  performance and chip area occupation, and analyzes design trade-offs across
  different parallelism levels

### A Web Services Based Architecture for Digital Time Stamping (2004)

**Authors:** A. Cilardo, A. Mazzeo, L. Romano, G.P. Saggese, G. Cattaneo

**Publication:** Journal of Web Engineering, vol. 2, no. 3, pp. 148-175, 2004

**Links:** [DBLP](https://dblp.org/rec/journals/jwe/CilardoMRSC04.bib)

- Web services framework: Architecture for digital time stamping services
  leveraging SOAP-based web services to enable distributed, interoperable time
  stamp authority infrastructure
- Standards integration: Implements RFC 3161 time stamp protocol with PKI
  infrastructure, providing secure and verifiable digital timestamps through
  standardized interfaces
- Enterprise deployment: Enables integration of time stamping services into
  business workflows and document management systems through web service
  interfaces and XML-based protocols

### Exploring the Design-Space for FPGA-Based Implementation of RSA (2004)

**Authors:** A. Cilardo, A. Mazzeo, L. Romano, G.P. Saggese

**Publication:** Microprocessors and Microsystems, vol. 28, no. 4, pp. 183-191,
2004

**Links:**
[PDF](https://gpsaggese.github.io/umd_classes/papers/Exploring_the_design-space_for_FPGA-based_implementation_of_RSA.pdf)
| [DOI](https://doi.org/10.1016/j.micpro.2004.03.009) |
[DBLP](https://dblp.org/rec/journals/mam/CilardoMRS04.bib)

- Design space exploration: Comprehensive methodology for evaluating RSA
  implementation alternatives on FPGAs, analyzing trade-offs between
  performance, area, and power consumption across different architectural
  choices
- Implementation strategies: Examines various approaches including digit-serial
  vs fully parallel architectures, different word sizes, and Montgomery
  multiplication optimizations for modular exponentiation
- Performance analysis: Provides quantitative comparison of different design
  points, demonstrating how architectural parameters affect throughput, latency,
  and resource utilization on reconfigurable hardware

### Carry-Save Montgomery Modular Exponentiation on Reconfigurable Hardware (2004)

**Authors:** A. Cilardo, A. Mazzeo, L. Romano, G.P. Saggese

**Publication:** 2004 Design, Automation and Test in Europe Conference and
Exposition (DATE 2004), Paris, France, pp. 206-211, 2004

**Links:**
[PDF](https://gpsaggese.github.io/umd_classes/papers/Carry-save_Montgomery_modular_exponentiation_on_reconfigurable_hardware.pdf)
| [DOI](https://doi.org/10.1109/DATE.2004.1269231) |
[DBLP](https://dblp.org/rec/conf/date/CilardoMRS04.bib)

- Novel algorithm: Introduces a carry-save implementation of Montgomery modular
  exponentiation that eliminates carry propagation delays in the critical
  computation path, enabling higher clock frequencies
- Architecture design: Presents a systolic array architecture for FPGAs that
  implements the carry-save Montgomery algorithm with optimized datapath and
  control logic for cryptographic operations
- Performance improvements: Achieves significant speedup compared to traditional
  implementations by reducing the critical path delay through carry-save
  representation, making it suitable for high-performance cryptographic
  applications

### Using Programmable Hardware to Improve the Dependability of Cryptographic Applications (2003)

**Author:** G.P. Saggese

**Publication:** PhD Thesis, University of Naples Federico II, Italy, 2003

**Links:** [DBLP](https://dblp.org/rec/phd/it/Saggese03.bib)

- FPGA-based cryptography: Explores use of field-programmable gate arrays to
  implement high-performance, tamper-resistant cryptographic processors with
  focus on RSA and AES algorithms
- Dependability analysis: Investigates how reconfigurable hardware can enhance
  security and reliability of cryptographic applications through hardware-based
  protection mechanisms and fault tolerance
- Implementation strategies: Develops methodologies for designing secure
  cryptographic systems on FPGAs, balancing performance, area efficiency, and
  resistance to physical and side-channel attacks

### FPGA-Based Implementation of a Serial RSA Processor (2003)

**Authors:** A. Mazzeo, L. Romano, G.P. Saggese, N. Mazzocca

**Publication:** 2003 Design, Automation and Test in Europe Conference and
Exposition (DATE 2003), Munich, Germany, pp. 10582-10589, 2003

**Links:**
[PDF](https://gpsaggese.github.io/umd_classes/papers/FPGA-based_implementation_of_a_serial_RSA_processor.pdf)
| [DOI](https://doi.org/10.1109/DATE.2003.10188) |
[DBLP](https://dblp.org/rec/conf/date/MazzeoRSM03.bib)

- Serial architecture: Presents a resource-efficient serial RSA processor design
  for FPGAs that minimizes hardware complexity while maintaining reasonable
  performance through bit-serial processing
- Montgomery algorithm: Implements RSA encryption/decryption using Montgomery
  modular multiplication in a serial fashion, trading speed for reduced area
  occupation on reconfigurable hardware
- Implementation results: Demonstrates practical FPGA implementation with
  detailed analysis of hardware utilization, throughput, and scalability for
  different RSA key sizes (1024, 2048 bits)

### An FPGA-Based Performance Analysis of the Unrolling, Tiling, and Pipelining of the AES Algorithm (2003)

**Authors:** G.P. Saggese, A. Mazzeo, N. Mazzocca, A.G.M. Strollo

**Publication:** Field Programmable Logic and Application (FPL 2003), Lisbon,
Portugal, Lecture Notes in Computer Science, vol. 2778, pp. 292-302, Springer,
2003

**Links:**
[PDF](https://gpsaggese.github.io/umd_classes/papers/An_FPGA-Based_Performance_Analysis_of_the_Unrolling,_Tiling,_and_Pipelining_of_the_AES_Algorithm.pdf)
| [DOI](https://doi.org/10.1007/978-3-540-45234-8_29) |
[DBLP](https://dblp.org/rec/conf/fpl/SaggeseMMS03.bib)

- Optimization techniques: Systematically explores three fundamental
  optimization strategies (loop unrolling, loop tiling, and pipelining) for
  accelerating AES encryption on FPGAs
- Performance evaluation: Provides comprehensive empirical analysis of how each
  transformation affects throughput, latency, and hardware resource utilization
  across different implementation configurations
- Design trade-offs: Demonstrates that combining transformations yields varying
  performance/area trade-offs, enabling designers to select optimal
  configurations based on specific application requirements

### Using Web Services Technology for Inter-enterprise Integration of Digital Time Stamping (2003)

**Authors:** A. Cilardo, A. Mazzeo, L. Romano, G.P. Saggese, G. Cattaneo

**Publication:** On The Move to Meaningful Internet Systems 2003: OTM 2003
Workshops, Catania, Sicily, Italy, Lecture Notes in Computer Science, vol. 2889,
pp. 960-974, Springer, 2003

**Links:**
[PDF](https://gpsaggese.github.io/umd_classes/papers/Using_Web_Services_Technology_for_Inter-enterprise_Integration_of_Digital_Time_Stamping.pdf)
| [DOI](https://doi.org/10.1007/978-3-540-39962-9_93) |
[DBLP](https://dblp.org/rec/conf/otm/CilardoMRSC03.bib)

- Web services architecture: Proposes a distributed architecture using
  SOAP-based web services to enable interoperability between different digital
  time stamping authorities and clients across organizational boundaries
- Standards compliance: Implements RFC 3161 time stamp protocol and integrates
  with PKI infrastructure, providing secure and verifiable time stamping
  services through standardized web service interfaces
- Inter-enterprise integration: Enables seamless integration of time stamping
  services into business workflows by leveraging WSDL, UDDI, and XML
  technologies for service discovery and invocation across enterprise boundaries

### Providing Digital Time Stamping Services to Mobile Devices (2003)

**Authors:** D. Cotroneo, C. di Flora, A. Mazzeo, L. Romano, S. Russo, G.P.
Saggese

**Publication:** 9th IEEE International Workshop on Object-Oriented Real-Time
Dependable Systems (WORDS Fall 2003), Anacapri (Capri Island), Italy, pp.
94-100, 2003

**Links:** [DOI](https://doi.org/10.1109/WORDS.2003.1267495) |
[DBLP](https://dblp.org/rec/conf/words/CotroneoFMRRS03.bib)

- Mobile architecture: Extends digital time stamping services to mobile devices,
  addressing challenges of limited computational resources, intermittent
  connectivity, and security constraints in mobile environments
- Protocol adaptation: Adapts RFC 3161 time stamp protocol for mobile devices
  with optimizations for bandwidth efficiency, battery consumption, and handling
  of network disconnections
- Real-time requirements: Addresses dependability and real-time requirements for
  mobile time stamping applications including secure document signing and
  authentication on resource-constrained devices

### Providing Interoperable Time Stamping Services (2003)

**Authors:** A. Mazzeo, L. Romano, G.P. Saggese

**Publication:** Scuola Superiore, 2003

**Note:** PDF not available online

- Interoperability framework: Addresses challenges of ensuring interoperability
  between different time stamping authority implementations through standardized
  protocols and data formats
- Cross-platform integration: Enables seamless integration of time stamping
  services across heterogeneous platforms and organizational boundaries using
  open standards
- Service architecture: Proposes architecture for time stamping service
  infrastructure that supports multiple time stamp authority providers while
  maintaining consistent verification and validation procedures

## Computer Design Publications

_Fault Tolerance, Computer Arithmetic, FPGA Design, Microprocessor Architecture_

### Automated Derivation of Application-Specific Error Detectors Using Dynamic Analysis (2011)

**Authors:** K. Pattabiraman, G.P. Saggese, D. Chen, Z. Kalbarczyk, R.K. Iyer

**Publication:** IEEE Transactions on Dependable and Secure Computing, vol. 8,
no. 5, pp. 640-655, 2011

**Links:**
[PDF](https://gpsaggese.github.io/umd_classes/papers/Automated_Derivation_of_Application-Specific_Error_Detectors_Using_Dynamic_Analysis.pdf)
| [DOI](https://doi.org/10.1109/TDSC.2010.19) |
[DBLP](https://dblp.org/rec/journals/tdsc/PattabiramanSCKI11.bib)

- Automated methodology: Presents a technique for automatically generating
  application-specific error detectors through dynamic program execution
  analysis, identifying critical variables and deriving program invariants
  during runtime
- Critical variable targeting: Focuses on detecting data errors by learning
  application-specific behavioral signatures that target variables most prone to
  corruption and with highest impact on program correctness
- Hardware implementation: Demonstrates FPGA-based implementation showing the
  approach scales to real-world deployment, successfully catching data
  corruption errors while minimizing false positives through
  application-specific tuning

### Dynamic Derivation of Application-Specific Error Detectors and their Implementation in Hardware (2006)

**Authors:** K. Pattabiraman, G.P. Saggese, D. Chen, Z. Kalbarczyk, R.K. Iyer

**Publication:** Sixth European Dependable Computing Conference (EDCC 2006),
Coimbra, Portugal, pp. 97-108, 2006

**Links:**
[PDF](https://gpsaggese.github.io/umd_classes/papers/Dynamic_Derivation_of_Application-Specific_Error_Detectors_and_their_Implementation_in_Hardware.pdf)
| [DOI](https://doi.org/10.1109/EDCC.2006.9) |
[DBLP](https://dblp.org/rec/conf/edcc/PattabiramanSCKI06.bib)

- Automated synthesis: Algorithm that analyzes dynamic execution traces to
  automatically extract error detector specifications, identifying detector
  classes, parameters, and optimal placement within application code
- Error detection optimization: Systematic process for maximizing error
  detection coverage through application-specific optimization, validated
  through comprehensive fault injection testing
- Hardware implementation: Efficient runtime verification through hardware
  implementation with detailed evaluation of detection coverage, resource
  requirements, and performance overhead across multiple benchmark programs

### An Experimental Study of Soft Errors in Microprocessors (2005)

**Authors:** G.P. Saggese, N.J. Wang, Z. Kalbarczyk, S.J. Patel, R.K. Iyer

**Publication:** IEEE Micro, vol. 25, no. 6, pp. 30-39, 2005

**Links:**
[PDF](https://gpsaggese.github.io/umd_classes/papers/An_experimental_study_of_soft_errors_in_microprocessors.pdf)
| [DOI](https://doi.org/10.1109/MM.2005.104) |
[DBLP](https://dblp.org/rec/journals/micro/SaggeseWKPI05.bib)

- Experimental analysis: Investigates soft errors in microprocessors through
  systematic fault injection techniques to assess how transient faults affect
  processor operation and reliability across different architectural components
- Vulnerability assessment: Provides empirical data on soft error sensitivity
  across microprocessor elements, identifying which architectural structures are
  most susceptible to failure and how faults propagate through the system
- Design implications: Develops assessment methodologies and protection
  strategies based on experimental findings, advancing understanding of
  microprocessor fault tolerance and offering practical insights for designing
  resilient computing systems

### Microprocessor Sensitivity to Failures: Control vs Execution and Combinational vs Sequential Logic (2005)

**Authors:** G.P. Saggese, A. Vetteth, Z. Kalbarczyk, R.K. Iyer

**Publication:** 2005 International Conference on Dependable Systems and
Networks (DSN 2005), Yokohama, Japan, pp. 760-769, 2005

**Links:**
[PDF](https://gpsaggese.github.io/umd_classes/papers/Microprocessor_sensitivity_to_failures_control_vs._execution_and_combinational_vs._sequential_logic.pdf)
| [DOI](https://doi.org/10.1109/DSN.2005.63) |
[DBLP](https://dblp.org/rec/conf/dsn/SaggeseVKI05.bib)

- Comparative fault analysis: Examines microprocessor sensitivity to hardware
  failures through systematic fault injection experiments, comparing
  vulnerabilities between control logic vs execution logic and combinational vs
  sequential circuits
- Empirical findings: Reveals that execution logic exhibits greater sensitivity
  to failures than control logic, and sequential circuits demonstrate higher
  vulnerability compared to combinational circuits
- Design priorities: Provides quantitative data on fault propagation across
  different architectural regions, enabling designers to prioritize protection
  mechanisms for the most critical components in fault-tolerant processors

### An Architectural Framework for Detecting Process Hangs/Crashes (2005)

**Authors:** N. Nakka, G.P. Saggese, Z. Kalbarczyk, R.K. Iyer

**Publication:** Dependable Computing - EDCC-5, Budapest, Hungary, Lecture Notes
in Computer Science, vol. 3463, pp. 103-121, Springer, 2005

**Links:**
[PDF](https://gpsaggese.github.io/umd_classes/papers/An_Architectural_Framework_for_Detecting_Process_Hangs-Crashes.pdf)
| [DOI](https://doi.org/10.1007/11408901_8) |
[DBLP](https://dblp.org/rec/conf/edcc/NakkaSKI05.bib)

- In-processor hardware module: Proposes three hardware detection techniques
  integrated into the superscalar processor pipeline to reduce error detection
  latency and instrumentation overhead for heartbeat-based process crash and
  hang detection
- Multi-level hang detection: Implements Instruction Count Heartbeat (ICH) for
  detecting crashes and instruction-level hangs, Infinite Loop Hang Detector
  (ILHD) for legitimate loop hangs, and Sequential Code Hang Detector (SCHD) for
  illegal loop detection
- Practical implementation: Addresses challenges in practical deployment of
  heartbeat-based detection by providing hardware-accelerated monitoring that
  operates within the processor's main pipeline without significant performance
  impact

### Architecture and FPGA Implementation of a Digit-serial RSA Processor (2005)

**Authors:** A. Cilardo, A. Mazzeo, L. Romano, G.P. Saggese

**Publication:** New Algorithms, Architectures and Applications for
Reconfigurable Computing, pp. 209-218, Springer, 2005

**Links:**
[PDF](https://gpsaggese.github.io/umd_classes/papers/Architecture_and_FPGA_Implementation_of_a_Digit-serial_RSA_Processor.pdf)

- Digit-serial architecture: Extends serial RSA implementation to process data
  in higher-radix digits rather than single bits, providing a middle ground
  between fully parallel and bit-serial approaches with improved area-time
  tradeoffs
- Montgomery multiplication: Implements Montgomery modular multiplication
  algorithm in digit-serial fashion, processing w-bit digits per clock cycle to
  balance throughput requirements with hardware resource constraints
- FPGA implementation: Presents detailed FPGA implementation results
  demonstrating how digit-serial processing enables scalable RSA cryptographic
  operations with configurable performance and area characteristics for
  different key sizes

### Hardware Support for High Performance, Intrusion- and Fault-Tolerant Systems (2004)

**Authors:** G.P. Saggese, C. Basile, L. Romano, Z. Kalbarczyk, R.K. Iyer

**Publication:** 23rd International Symposium on Reliable Distributed Systems
(SRDS 2004), Florianópolis, Brazil, pp. 195-204, 2004

**Links:**
[PDF](https://gpsaggese.github.io/umd_classes/papers/Hardware_support_for_high_performance_intrusion-_and_fault-tolerant_systems.pdf)
| [DOI](https://doi.org/10.1109/RELDIS.2004.1353020) |
[DBLP](https://dblp.org/rec/conf/srds/SaggeseBRKI04.bib)

- Integrated architecture: FPGA-based crypto-engine combining optimized RSA
  processors for computationally intensive cryptographic operations with
  tamper-resistant key storage, using multiple clock domains and balanced
  parallelism for linear speed-up
- Formal verification: Architecture formally modeled and verified using Spin
  model checker, implementing preemptive deterministic scheduling algorithm to
  handle nondeterministic behavior and guarantee strong replica consistency in
  replicated systems
- Security and performance: Combines active replication with threshold
  cryptography and multithreaded replica code to efficiently leverage parallel
  crypto-engine, demonstrated on attribute authority servers requiring both high
  performance and robust security

### A Technique for FPGA Synthesis Driven by Automatic Source Code Analysis and Transformations (2002)

**Authors:** B. Di Martino, N. Mazzocca, G.P. Saggese, A.G.M. Strollo

**Publication:** Field-Programmable Logic and Applications (FPL 2002),
Montpellier, France, Lecture Notes in Computer Science, vol. 2438, pp. 47-58,
Springer, 2002

**Links:**
[PDF](https://gpsaggese.github.io/umd_classes/papers/A_Technique_for_FPGA_Synthesis_Driven_by_Automatic_Source_Code_Analysis_and_Transformations.pdf)
| [DOI](https://doi.org/10.1007/3-540-46117-5_7) |
[DBLP](https://dblp.org/rec/conf/fpl/MartinoMSS02.bib)

- Automatic synthesis from C: Presents technique for automatic synthesis of
  high-performance FPGA-based computing machines from C language source code by
  exploiting data-parallelism through hardware application of automatic loop
  transformations
- Design space exploration: Uses transformation-intensive branch-and-bound
  approach to search design space and explore area-performance tradeoffs,
  considering performance aspects early in the design stage before low-level
  synthesis
- Architectural optimization: Applies optimizations at architectural level using
  hardware block library for arithmetic and functional primitives, achieving
  higher benefits compared to gate-level optimizations

### Shuffled Serial Adder: An Area-Latency Effective Serial Adder (2002)

**Authors:** G.P. Saggese, A.G.M. Strollo, N. Mazzocca, D. De Caro

**Publication:** Proceedings of the 2002 9th IEEE International Conference on
Electronics, Circuits and Systems (ICECS 2002), Dubrovnik, Croatia, pp. 607-610,
2002

**Links:**
[PDF](https://gpsaggese.github.io/umd_classes/papers/Shuffled_serial_adder_an_area-latency_effective_serial_adder.pdf)
| [DOI](https://doi.org/10.1109/ICECS.2002.1046242) |
[DBLP](https://dblp.org/rec/conf/icecsys/SaggeseSMC02.bib)

- Novel architecture: Serial adder with logarithmic latency proportional to
  log₂N (where N is operand width), derived from Kogge-Stone adder but
  reorganized into slice-based format for practical serial hardware
  implementation
- Direct parallel I/O: Accepts parallel inputs and produces parallel outputs
  directly, eliminating conversion overhead required by traditional digit-serial
  approaches that need separate data formatting stages
- Performance validation: Circuit simulations demonstrate competitive area-time
  tradeoffs compared to digit-serial adders across multiple operand widths using
  standard-cell VLSI implementations

### A Reconfigurable 2D Convolver for Real-Time SAR Imaging (2001)

**Authors:** A.G.M. Strollo, E. Napoli, D. De Caro, G.P. Saggese

**Publication:** Proceedings of the 2001 8th IEEE International Conference on
Electronics, Circuits and Systems (ICECS 2001), Malta, pp. 741-744, 2001

**Links:**
[PDF](https://gpsaggese.github.io/umd_classes/papers/A_reconfigurable_2D_convolver_for_real-time_SAR_imaging.pdf)
| [DOI](https://doi.org/10.1109/ICECS.2001.957581) |
[DBLP](https://dblp.org/rec/conf/icecsys/StrolloNCS01.bib)

- Reconfigurable parallel architecture: Presents completely parallel structure
  for SAR signal processing that can be dynamically adapted to accommodate
  varying dimensions of data matrices and filter configurations for flexible
  real-time operation
- Time-domain processing: Achieves real-time processing using signum-coded
  algorithm with time-domain processing instead of frequency-domain techniques,
  enabling efficient convolution operations for SAR imaging applications
- VLSI implementation: Validates architecture through standard-cell VLSI
  implementation and comprehensive circuit simulations, demonstrating
  computational efficiency suitable for practical synthetic aperture radar
  systems

### Test Pattern Generator for Hybrid Testing of Combinational Circuits (2001)

**Authors:** D. De Caro, N. Mazzocca, E. Napoli, G.P. Saggese, A.G.M. Strollo

**Publication:** Proceedings of the 2001 8th IEEE International Conference on
Electronics, Circuits and Systems (ICECS 2001), Malta, pp. 745-748, 2001

**Links:**
[PDF](https://gpsaggese.github.io/umd_classes/papers/Test_pattern_generator_for_hybrid_testing_of_combinational_circuits.pdf)
| [DOI](https://doi.org/10.1109/ICECS.2001.957582) |
[DBLP](https://dblp.org/rec/conf/icecsys/CaroMNSS01.bib)

- Hybrid testing approach: Novel test pattern generator combining pseudo-random
  testing (using linear feedback network) and deterministic testing (using
  nonlinear feedback network) through shift register reacted through two
  different networks
- Automated synthesis: Synthesis tool employing state space heuristic search and
  selfish gene genetic algorithm to design generators optimized for specific
  test requirements and circuit characteristics
- Performance validation: Testing on ISCAS'85 benchmark circuits demonstrates
  improvements over existing techniques in both synthesis time and test sequence
  length for built-in self-test implementation
