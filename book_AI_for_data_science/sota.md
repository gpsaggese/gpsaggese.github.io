# AutoML & Auto EDA: Comprehensive Resource Guide
A curated collection of papers, books, GitHub projects, and tools for Automated
Machine Learning (AutoML) and Automated Exploratory Data Analysis (Auto EDA)

## Recent Papers & Surveys (2021-2025)

### AutoML Papers

#### Alcobaca et al., "A Literature Review on Automated Machine Learning", 2025
- **Description**: Traces AutoML evolution from early metalearning and
  hyperparameter optimization to latest advancements in neural architecture
  search and automated pipeline design
- **Link**:
  - https://repositorio.usp.br/bitstreams/866fc2e9-8e21-4460-a0d8-87b40fc88222
  - https://link.springer.com/article/10.1007/s10462-025-11397-2

#### Baratchi, M. et al., "Automated Machine Learning: Past, Present and Future", 2024
- **Description**: Comprehensive overview covering search space, search
  strategy, performance evaluation, hyperparameter optimization, and neural
  architecture search
- **Link**:
  - https://publications.rwth-aachen.de/record/985693/files/985693.pdf
  - https://link.springer.com/article/10.1007/s10462-024-10726-1

#### He, X. et al., "Automated Machine Learning: From Principles to Practices", 2024
- **Description**: Focuses on principle analysis of AutoML
  - Formal definitions
  - Bi-level optimization formulation
  - Theoretical analysis
  - Real-world AutoML practices
- **Link**: https://arxiv.org/html/1810.13306v5

#### Zhang, Y. et al., "A Survey on Recent Advancements in Auto-Machine Learning with a Focus on Feature Engineering", 2023
- **Description**: Detailed study on AutoML pipeline steps including:
  - Feature selection (FS)
  - Feature engineering (FE)
  - Neural architecture search (NAS)
  - Hyperparameter optimization (HPO)
  - Model selection (MS)
- **Link**: https://www.researchgate.net/publication/371726620

#### Jomaa, H. S. et al., "AutoML: A Survey of the State-of-the-Art", 2021
- **Link**: https://arxiv.org/abs/1908.00709

#### Zoeller, M. A. et al., "AutoML to Date and Beyond: Challenges and Opportunities", 2021
- **Link**: https://arxiv.org/abs/2010.10777

### Auto EDA Papers

#### Zhu, J.-P. et al., "Towards Automated Cross-domain Exploratory Data Analysis Through Large Language Models", 2025
- **Description**: Latest LLM-based EDA approach addressing:
  - SQL query generation for data exploration
  - Automated visualization type selection
  - Cross-domain applicability
- **Link**: https://arxiv.org/abs/2412.07214

#### Akella, A. and Narayanam, K., "QUIS: Question-guided Insights Generation for Automated Exploratory Data Analysis", 2024
- **Description**: Recent approach to automated EDA with ISGen module
  - Produces multiple relevant insights in response to questions
  - Requires no prior training
  - Enables adaptation to new datasets
- **Link**: https://arxiv.org/abs/2410.10270

#### Li, X. et al., "Automated Data Exploration and Analysis", 2024
- **Description**: Presents AutoEDA system with:
  - Dynamic field recognition framework
  - Analysis strategy evaluation based on Chart Attribute Length
  - Interactive visual backtracking
- **Link**: https://link.springer.com/chapter/10.1007/978-3-031-82021-2_27

#### Milo, T. and Somech, A., "Automating Exploratory Data Analysis Via Machine Learning: an Overview", (date unknown)
- **Description**: Comprehensive tutorial on automating EDA covering:
  - Single exploratory action recommenders
  - KNN-based classifiers for interestingness prediction
  - Active learning methods
  - Deep reinforcement learning
  - Sequence-to-sequence models
- **Link**:
  https://www.semanticscholar.org/paper/Automating-Exploratory-Data-Analysis-via-Machine-An-Milo-Somech

## AutoML Frameworks

### Major AutoML Tools Comparison
| Framework                             | Organization                 | Year  | Primary Focus              | Key Features                                                         | GitHub                                   |
| ------------------------------------- | ---------------------------- | ----- | -------------------------- | -------------------------------------------------------------------- | ---------------------------------------- |
| **Auto-sklearn**                      | University of Freiburg       | 2015+ | Classical ML               | Bayesian optimization, meta-learning, ensemble construction          | https://github.com/automl/auto-sklearn   |
| **AutoGluon**                         | Amazon Web Services          | 2019  | Deep Learning + Tabular    | Multi-layered ensembling, multimodal learning (text, image, tabular) | https://github.com/autogluon/autogluon   |
| **H2O AutoML**                        | H2O.ai                       | 2017+ | Distributed ML             | Random search, stacked ensembles, scalable, explainability           | https://github.com/h2oai/h2o-3           |
| **TPOT**                              | Epistasis Lab (Cedars-Sinai) | 2015+ | Genetic Programming        | Genetic algorithms, pipeline optimization, biomedical focus          | https://github.com/EpistasisLab/TPOT     |
| **FLAML**                             | Microsoft                    | 2020+ | Fast & Lightweight         | Cost-conscious optimization, multi-fidelity                          | https://github.com/microsoft/FLAML       |
| **LightAutoML**                       | SberAutoML Lab               | 2021+ | Fast AutoML                | Gradient boosting, time-series, fast execution                       | https://github.com/sb-ai-lab/LightAutoML |
| **AutoKeras**                         | DATA Lab                     | 2018+ | Deep Learning              | Neural architecture search for deep learning                         | https://github.com/autokeras/autokeras   |
| **PyCaret**                           | PyCaret Team                 | 2019+ | Low-code ML                | Simple API, comparison of multiple models                            | https://github.com/pycaret/pycaret       |
| **NNI (Neural Network Intelligence)** | Microsoft                    | 2018+ | Neural Architecture Search | Feature engineering, NAS, hyperparameter tuning                      | https://github.com/microsoft/nni         |
| **MLJAR**                             | MLJAR Team                   | 2019+ | User-friendly AutoML       | Open-source, accessible interface                                    | https://github.com/mljar/mljar-studio    |

### AutoML Framework Details

#### Auto-sklearn
- **Approach**: Bayesian optimization with meta-learning
- **Features**:
  - Automatic feature preprocessing
  - Model selection and ensembling
  - Meta-learning from past experiments
  - 15 classifiers, 14 preprocessing methods, 4 data preprocessing methods
  - 110 hyperparameters
- **Best For**: Classical machine learning on tabular data
- **Repository**: https://github.com/automl/auto-sklearn

#### AutoGluon
- **Approach**: Multi-layered stack ensembling
- **Features**:
  - Single-line training on unprocessed tabular CSV files
  - Deep learning integration
  - Multimodal support (tabular, text, images)
  - Efficient and scalable
  - Minimal user intervention required
- **Best For**: Production environments, multimodal learning
- **Repository**: https://github.com/autogluon/autogluon

#### H2O AutoML
- **Approach**: Random search + stacked ensembles
- **Features**:
  - Automatic training and tuning of multiple models
  - Time-limited execution
  - Support for GBM, Random Forests, DNNs, GLMs
  - Leaderboard of ranked models
  - Built-in model explainability
  - Scalable distributed processing
  - Available in R, Python, Java, Scala
- **Best For**: Large-scale data, distributed computing
- **Repository**: https://github.com/h2oai/h2o-3

#### TPOT (Tree Pipeline Optimization Tool)
- **Approach**: Genetic algorithms for pipeline optimization
- **Features**:
  - Optimizes ML pipeline combinations
  - Feature preprocessing techniques
  - Hyperparameter tuning via genetic programming
  - Originally designed for biomedical data
  - Widely used as baseline in research
- **Best For**: Biomedical applications, genetic algorithm research
- **Repository**: https://github.com/EpistasisLab/TPOT
- **Note**: Original project no longer actively developed; TPOT2 is in progress

#### FLAML (Fast and Lightweight AutoML)
- **Approach**: Cost-conscious optimization
- **Features**:
  - Efficient hyperparameter optimization
  - Multi-fidelity evaluation
  - Support for multiple ML libraries
- **Repository**: https://github.com/microsoft/FLAML

#### LightAutoML
- **Approach**: Gradient boosting focused
- **Features**:
  - Fast execution
  - Time-series support
  - Distributed processing
- **Repository**: https://github.com/sb-ai-lab/LightAutoML

### AutoML Resource Collections

#### Awesome AutoML Papers
- **Description**: Curated list of automated machine learning papers, articles,
  tutorials, slides and projects
- **Includes**: Up-to-date overviews of AutoML techniques
- **Key Papers**:
  - Baratchi, M. et al., "Automated machine learning: past, present and future", 2024
  - Li, Y. et al., "On Hyperparameter Optimization of Machine Learning Algorithms", 2020
  - Elshawi, R. et al., "Automated Machine Learning: State-of-The-Art and Open Challenges", 2019
  - Yao, Q. et al., "Taking Human out of Learning Applications: A Survey on Automated Machine Learning", 2018
- **GitHub**: https://github.com/hibayesian/awesome-automl-papers

#### Automl-list
- **Description**: Comprehensive list of open-source and commercial AutoML tools
- **Includes**: Tool comparisons, descriptions, and categorizations
- **GitHub**: https://github.com/askery/automl-list

## Auto EDA Tools

### Python Auto EDA Libraries

#### Pandas-Profiling (YData Profiling)
- **Type**: Library
- **Package Name**: ydata-profiling (formerly pandas-profiling)
- **Key Features**:
  - Quick data summaries
  - Correlation analysis
  - Distribution analysis
  - Missing value detection
  - Profile reports with detailed statistics
- **Installation**: `pip install ydata-profiling`
- **Use Case**: Quick EDA reports for any dataset

#### AutoViz
- **Type**: Library
- **Key Features**:
  - Automated visualization
  - Data quality issue analysis and fixing
  - Relationship visualization
  - Customizable plots with Bokeh
  - Interactive chart formats
- **Installation**: `pip install autoviz`
- **Special**: Integrates pandas-dq for data quality checks
- **Use Case**: Automated visualization and data quality assessment

#### DataPrep.EDA
- **Type**: Library/Framework
- **Key Features**:
  - 10X faster than Pandas-based profiling
  - Dask-based computing for big data support
  - Interactive visualizations in reports
  - Dask DataFrame support
  - Collect and clean data functionality
- **Installation**: `pip install dataprep`
- **Use Case**: Large datasets, big data analysis
- **Note**: Highly optimized for performance

#### D-Tale
- **Type**: Web Application (Flask + React)
- **Key Features**:
  - Interactive Pandas data analysis
  - IPython notebook integration
  - 3D plots and heatmaps
  - Custom column creation
  - "Code Expert" feature (unique)
  - Detailed analysis and filtering
  - Code export functionality
- **Installation**: `pip install dtale`
- **Use Case**: Interactive exploratory analysis with code generation

#### Sweetviz
- **Type**: Library
- **Key Features**:
  - Auto-visualization with high-density plots
  - Dataset comparison capabilities
  - Train/test dataset comparison
  - Quick report generation
  - Interactive HTML output
- **Installation**: `pip install sweetviz`
- **Use Case**: Comparative data analysis and reports

#### ExploriPy
- **Type**: Library
- **Key Features**:
  - Automated EDA
  - Statistical tests (ANOVA, Chi-Square, WOE, Information Value, Tukey HSD)
  - Interactive HTML output
  - Various charts and tables
  - CSV download options
- **Installation**: `pip install exploripy`
- **Input**: Pandas DataFrame + categorical variables list
- **Use Case**: Statistical analysis and EDA for tabular data

#### Lux (Lux-API)
- **Type**: Library
- **Key Features**:
  - Intelligent visual discovery
  - Single print statement visualization
  - Automatic insight generation
  - Pandas DataFrame integration
- **Installation**: `pip install lux-api`
- **Use Case**: Automatic insight discovery

#### DataPrep (General)
- **Type**: Low-code data preparation library
- **Key Features**:
  - Data collection from common sources
  - Data cleaning capabilities
  - EDA module (DataPrep.EDA)
  - Visualization tools
- **Installation**: `pip install dataprep`
- **Use Case**: End-to-end data preparation

#### Additional Libraries
- **speedML**: Large ML library with EDA module
- **edaviz**: Fast data exploration (free version for small datasets)
- **pandas-summary**: Simple extension to pandas.describe()
- **HoloViews**: Automated visualization based on data annotations
- **lens**: Fast calculation of summary statistics and correlations

### R Auto EDA Packages

#### DataExplorer
- **Type**: CRAN Package
- **Key Features**:
  - Automated data exploration
  - Univariate and bivariate plots
  - PCA analysis
  - Data treatment capabilities

#### funModeling
- **Type**: CRAN Package
- **Key Features**:
  - Automated EDA
  - Simple feature engineering
  - Outlier detection
  - Exploration functions

#### SmartEDA
- **Type**: CRAN Package
- **Key Features**:
  - Automated generation of descriptive statistics
  - Univariate and bivariate plots
  - Parallel coordinate plots
- **Reference**: Dedicated research paper available

#### Visdat
- **Type**: CRAN Package
- **Key Features**: 6 exploratory/diagnostic plots for initial data analysis

#### Dlookr
- **Type**: CRAN Package
- **Key Features**:
  - Data quality diagnosis
  - Basic exploration
  - Feature transformations

#### Xray
- **Type**: CRAN Package
- **Key Features**:
  - First look at data distributions
  - Anomaly detection

#### Arsenal
- **Type**: CRAN Package
- **Key Features**:
  - Statistical summaries
  - Quick reporting capabilities

### Auto EDA GitHub Projects

#### autoEDA-resources
- **Description**: Curated list of software and papers related to automated EDA
- **Includes**: Visualization recommendation, tools for data exploration
- **GitHub**: https://github.com/mstaniak/autoEDA-resources
- **Content**: Comprehensive database of AutoEDA tools and papers

#### Exploratory-data-analysis-python
- **Description**: Collection of automated EDA tools with practical examples
- **Tools Covered**:
  - D-Tale
  - AutoViz
  - DataPrep
  - Klib
  - Lux
  - Pandas-Profiling
  - Sweetviz
- **GitHub**:
  https://github.com/aanand-datascience/exploratory-data-analysis-python

#### EDA-with-AutoEDA-libraries
- **Description**: Implementation, analysis, and comparison of different Auto
  EDA libraries
- **Analysis**: Working functionalities and insights of various libraries
- **GitHub**: https://github.com/shivpalSW/EDA-with-AutoEDA-libraries

#### AutoEDA Toolkit
- **Description**: Open-source Python application for automated EDA
- **Features**:
  - Data preprocessing
  - Missing data handling
  - Visualization
  - Insight discovery
  - Pattern finding
- **GitHub**: https://github.com/Devang-C/AutoEDA

#### Automated-EDA (Yogi776)
- **Description**: Comprehensive automated EDA demonstration
- **Tools Compared**:
  - Pandas-profiling
  - Sweetviz
  - AutoViz
  - D-Tale
- **Includes**: Data quality checks, statistical tests, quantitative tests
- **GitHub**: https://github.com/Yogi776/Automated-EDA

#### Automated-EDA Toolkit (Manisjha)
- **Description**: Complete toolkit using multiple Python libraries
- **Libraries Included**:
  - Pandas_profiling
  - Sweetviz
  - Autoviz
  - Dataprep
  - D-Tale
  - Missingno
  - Plotly
  - Pandas-visual-analysis
- **GitHub**: https://github.com/Manisjha/Automated-EDA

#### AutoEDA (kkm24132)
- **Description**: Comparative analysis of AutoEDA tools
- **Features Compared**: Performance, speed, comprehensiveness, visualizations
- **GitHub**: https://github.com/kkm24132/AutoEDA

## Books & Learning Resources

### Primary AutoML Book

#### Automated Machine Learning: Methods, Systems, Challenges
- **Editors**: Frank Hutter, Lars Kotthoff, Joaquin Vanschoren
- **Publisher**: Springer
- **Year**: 2019
- **ISBN**: 978-3-030-05318-5
- **DOI**: https://doi.org/10.1007/978-3-030-05318-5
- **Content**: Comprehensive coverage of AutoML methodologies, systems, and
  research challenges
- **URL**: https://link.springer.com/book/10.1007/978-3-030-05318-5

### Data Science & EDA References

#### Data Wrangling with Pandas
- **Author**: Wes McKinney
- **Content**: Comprehensive guide including:
  - Data loading and cleaning
  - EDA techniques
  - Visualization
  - Statistical analysis
  - Numerous examples and exercises

### Online Resources & Tutorials

#### Automated EDA Guide - Kanaries
- **Content**: Benefits, drawbacks, and step-by-step guides
- **Libraries Covered**: DataPrep, Pandas
- **URL**: https://docs.kanaries.net/articles/auto-eda-guide

#### EDA Tools Documentation - DrShahizan
- **Description**: Detailed documentation on Python EDA tools
- **Content**: Tool comparisons, usage guides, best practices
- **URL**: https://drshizan.gitbook.io/eda1/tools-and-software/python-tools

#### 10 GitHub Repositories for AutoML
- **Description**: Guide on using major AutoML frameworks
- **Frameworks Covered**: Auto-sklearn, TPOT, AutoKeras, H2O AutoML, NNI,
  Hypertunity, Dragonfly
- **URL**:
  https://www.sabrepc.com/blog/Deep-Learning-and-AI/10-github-repositories-for-automl

#### 8 Open-Source AutoML Frameworks
- **Date**: February 10, 2025
- **Content**: Historical evolution and comparison of AutoML tools
- **Coverage**: Auto-WEKA, TPOT, AutoGluon, H2O, Auto-sklearn, MLJAR
- **URL**: https://mljar.com/blog/python-automl/

## Benchmark & Comparison Resources

### OpenML AutoML Benchmark
- **Description**: Official benchmark comparing multiple AutoML systems
- **Tools Benchmarked**: TPOT, auto-sklearn, H2O AutoML, AutoGluon-Tabular
- **Repository**: https://github.com/h2oai/h2o-automl-paper
- **Includes**: Results files, benchmark instructions, framework comparisons
- **Paper**: LeDell, E. et al., "H2O AutoML: Scalable Automatic Machine Learning", 2020 (ICML AutoML Workshop)

### H2O AutoML Paper Repository
- **Description**: Code and experiments for H2O AutoML paper
- **Content**:
  - Baseline experiments
  - Blending vs CV Stacking comparisons
  - OpenML benchmark results
  - Hardware configurations (c5.metal Amazon EC2 instance)
- **Data**: Airlines dataset with experimental setup
- **GitHub**: https://github.com/h2oai/h2o-automl-paper

### A Survey of Evaluating AutoML and Automated Feature Engineering
- **Date**: 2025
- **Focus**: Comprehensive review of AutoML and feature engineering tools
- **Key Insights**:
  - Performance evaluation on structured and time-series data
  - Tool strengths and limitations
  - Integration benefits with AutoML frameworks
  - Benchmarking on industry-related datasets
- **Content**: Addresses gap between separate tool evaluations and real-world
  applicability
- **PDF**: https://www.scitepress.org/Papers/2025/132667/132667.pdf

### Deep Pipeline Embeddings for AutoML
- **Description**: Recent research on pipeline embeddings
- **Focus**: Advanced AutoML techniques and architecture
- **Reference Paper**: arXiv:2305.14009
- **Link**: https://arxiv.org/abs/2305.14009

## Quick Comparison: Popular Tools

### AutoML - Best Use Cases
| Use Case                        | Recommended Tool | Reason                                        |
| ------------------------------- | ---------------- | --------------------------------------------- |
| **Tabular Data (Classical ML)** | Auto-sklearn     | Excellent meta-learning, established baseline |
| **Production Environment**      | AutoGluon        | Scalable, multimodal, minimal setup           |
| **Large-Scale Data**            | H2O AutoML       | Distributed, fast, explainable                |
| **Biomedical Applications**     | TPOT             | Designed for domain, genetic programming      |
| **Deep Learning**               | AutoKeras, NNI   | Neural architecture search focus              |
| **Quick Prototyping**           | PyCaret          | User-friendly, simple API                     |
| **Cost-Sensitive**              | FLAML            | Efficient optimization                        |

### Auto EDA - Best Use Cases
| Scenario                  | Recommended Tool | Key Advantage                             |
| ------------------------- | ---------------- | ----------------------------------------- |
| **Quick Report**          | Pandas-Profiling | Fast, comprehensive, standard             |
| **Large Datasets**        | DataPrep.EDA     | 10X faster with Dask                      |
| **Interactive Analysis**  | D-Tale           | Code export, detailed exploration         |
| **Visual Insights**       | AutoViz          | Automated visualization, quality checks   |
| **Statistical Tests**     | ExploriPy        | ANOVA, Chi-Square, WOE, Information Value |
| **Dataset Comparison**    | Sweetviz         | Train/test comparison, easy reports       |
| **Intelligent Discovery** | Lux              | Automatic insight generation              |

## Timeline of AutoML Evolution

### Early Phase (2014-2017)
- Auto-WEKA (2014)
- TPOT (2015)
- Auto-sklearn (2015)
- Hyperparameter optimization focus

### Growth Phase (2017-2019)
- H2O AutoML (2017)
- AutoKeras (2018)
- Automated feature engineering
- NAS becoming prominent

### Maturation Phase (2019-2021)
- AutoGluon (2019)
- PyCaret (2019)
- FLAML (2020)
- Multi-modal AutoML

### Modern Phase (2021-Present)
- LLM-based AutoML and Auto EDA
- Reinforcement learning approaches
- Pipeline embeddings
- Cross-domain automation
- Production-ready systems

## Additional Resources

### Datasets for Testing
- OpenML datasets (https://www.openml.org)
- Kaggle competitions (https://www.kaggle.com)
- UCI Machine Learning Repository (https://archive.ics.uci.edu)

### Forums & Discussion
- Stack Overflow: [automl], [auto-sklearn], [autogluon] tags
- GitHub Discussions in respective repositories
- Academic conferences: ICML, NeurIPS, ICLR

### Workshops & Seminars
- ICML AutoML Workshop (annual)
- NeurIPS workshops on AutoML
- University courses on machine learning automation

## Glossary of Terms
- **AutoML**: Automated Machine Learning - automating ML model development
- **Auto EDA**: Automated Exploratory Data Analysis - automating initial data
  analysis
- **Hyperparameter Optimization (HPO)**: Tuning ML model parameters
  automatically
- **Neural Architecture Search (NAS)**: Automatically designing neural network
  architectures
- **Meta-Learning**: Learning from past experiments to improve future searches
- **Pipeline**: Complete ML workflow from data to predictions
- **Ensemble**: Combining multiple models for better predictions
- **Bayesian Optimization**: Probabilistic approach to hyperparameter search
- **Genetic Algorithms**: Evolution-based search optimization
- **Feature Engineering**: Creating relevant features from raw data
- **Feature Selection**: Choosing most important features
- **Model Selection**: Choosing best algorithm for the task
