**Description**

LiNGAM (Linear Non-Gaussian Acyclic Model) is a statistical method for causal inference that allows researchers to identify causal relationships from observational data. It is particularly useful for discovering the structure of causal graphs and understanding the underlying mechanisms of data generation. 

Technologies Used
LiNGAM

- Estimates causal relationships between variables using non-Gaussianity.
- Capable of identifying the direction of causal influence.
- Works with continuous and discrete data types.

---

### Project 1: Causal Relationships in Industrial Process Variables
**Difficulty**: 1 (Easy)

**Project Objective**:  
Explore causal relationships among continuous sensor readings in an industrial process to understand dependencies and potential process bottlenecks or failure propagations.

**Dataset Suggestions**:  
- Use the **CIPCaD-Bench** datasets (e.g., Tennessee Eastman process data) for causal discovery benchmarks.  
- Publicly available via [CIPCaD-Bench GitHub Repository](https://github.com/cipcad-bench?utm_source=chatgpt.com) and accompanying [research paper](https://arxiv.org/abs/2208.01529?utm_source=chatgpt.com).

**Tasks**:  
- Clean and prepare the dataset (sensor time-series, remove noise/outliers).  
- Apply LiNGAM to infer a causal DAG among the process variables.  
- Visualize the causal graph using Graphviz or NetworkX.  
- Interpret which process variables are likely causal drivers and discuss implications for process monitoring or control.

**Bonus Ideas (Optional)**:  
- Compare results with correlation-based analysis to highlight differences.  
- Extend analysis by including lagged features to explore delayed causal effects.

---

### Project 2: Mobility, Economic Activity, and Recovery Post-Shock
**Difficulty**: 2 (Medium)

**Project Objective**:  
Investigate how mobility (visits, travel distances, time in places), economic sector activity, and recovery from disruptions (e.g., pandemic lockdowns) causally relate to each other over time.

**Dataset Suggestions**:  
- Use the **High-resolution Mobility Data** dataset tracking mobility and economic activity across the U.S. from 2019–2023.  
- Available via [research dataset](https://arxiv.org/abs/2506.13985?utm_source=chatgpt.com) with public release information.  
- Supplement with county-level open economic datasets (e.g., U.S. Bureau of Economic Analysis).

**Tasks**:  
- Preprocess and align mobility and economic activity data by region and time.  
- Create lagged variables to test temporal causal directions.  
- Apply LiNGAM to estimate causal relationships and identify directionality.  
- Assess stability of causal edges across different periods (pre- vs post-shock).  

**Bonus Ideas (Optional)**:  
- Incorporate additional socioeconomic indicators (e.g., unemployment, poverty rates).  
- Compare LiNGAM findings with alternative methods such as Granger causality.

---

### Project 3: Causal Structure of Online News Popularity
**Difficulty**: 3 (Hard)

**Project Objective**:  
Identify causal relationships between content features (e.g., keywords, sentiment, publishing time) and the popularity of online news articles, focusing on what drives higher engagement.

**Dataset Suggestions**:  
- Use the **Online News Popularity Dataset** from the UCI Machine Learning Repository.  
- Publicly available here: [Online News Popularity](https://archive.ics.uci.edu/ml/datasets/Online+News+Popularity).  
- Contains 39,000 articles from Mashable with features such as number of keywords, sentiment polarity, publishing weekday, and shares.

**Tasks**:  
- **Data Preparation**:  
  - Load and preprocess the dataset (remove irrelevant columns, normalize continuous features).  
  - Select key features such as number of images, sentiment score, and publish day.  

- **Exploratory Data Analysis**:  
  - Analyze patterns between article characteristics and number of shares.  
  - Identify potential causal factors (e.g., does sentiment drive shares, or do shares correlate with posting time?).  

- **Apply LiNGAM**:  
  - Use LiNGAM to infer a causal graph among content features and popularity (shares).  
  - Visualize the directed acyclic graph with NetworkX or Graphviz.  

- **Validation**:  
  - Perform robustness checks by running LiNGAM on subsets of data (e.g., tech articles vs lifestyle articles).  
  - Compare with regression or correlation-based approaches to highlight differences.  

- **Business Implications**:  
  - Discuss how publishers could use these insights to optimize article release strategies (timing, content structure).  

**Bonus Ideas (Optional)**:  
- Extend analysis by adding **textual sentiment analysis** with NLTK or spaCy to refine features.  
- Compare causal results with machine learning predictive models (e.g., Random Forests) for popularity prediction.  
