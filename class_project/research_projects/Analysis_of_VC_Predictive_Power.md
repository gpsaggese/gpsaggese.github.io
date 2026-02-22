# **An Analysis of VC Predictive Power**

## **Objective**

This project aims to rigorously test the hypothesis that venture capitalists (VCs) have no meaningful predictive power in selecting startups that will outperform others. Using causal inference methods, we will separate correlation from causation to evaluate whether VC investment decisions truly contain forward-looking information, or if observed outcomes are simply the result of ex-post rationalizations and survivorship bias.

## **Research Questions**

1. **Do startups funded by “top” VCs outperform similar startups that were not funded by them?**
2. **Is VC involvement a causal driver of success, or merely correlated with characteristics that already predicted success?**
3. **After controlling for confounders (market, sector, timing, founder background), does VC selection still predict long-term outcomes?**

## **Methodology**

### **1\. Data Collection**

- Startup funding rounds (Crunchbase, PitchBook, or open datasets).
- Performance proxies: exit valuations, IPOs, survival rate, employment growth, revenue (if available).
- Control variables: industry, founding year, geography, founder experience, macroeconomic conditions.

### **2\. Causal Inference Framework**

- **Treatment Variable:** VC backing (binary: backed/not backed; or categorical: tier of VC).
- **Outcome Variable:** Long-term startup success (valuation, IPO, acquisition, or survival).
- **Confounders:** Startup characteristics that influence both VC interest and success.

### **3\. Analytical Tools**

- **Propensity Score Matching (PSM):** Match VC-backed startups with similar non-VC-backed startups.
- **Difference-in-Differences (DiD):** Compare performance trends before and after VC funding vs. matched controls.
- **Instrumental Variables (IV):** Use shocks to VC fund liquidity or geographical proximity as instruments to isolate exogenous variation.
- **Causal Forests / Double Machine Learning:** Machine learning approaches to estimate heterogeneous treatment effects.

### **4\. Hypothesis Testing**

- **Null Hypothesis (H₀):** VC funding has no causal effect on startup success (no predictive power).
- **Alternative Hypothesis (H₁):** VC funding has a positive causal effect.

## **Implementation in Python**

We will use open-source causal inference libraries and data science stacks:

- **Data Cleaning & Prep:** `pandas`, `numpy`
- **Modeling:** `statsmodels`, `scikit-learn`
- **Causal Inference:**
  - `econml` (Microsoft’s causal inference package)
  - `dowhy` (causal graphs and treatment effect estimation)
  - `causalml` (uplift modeling and treatment effect heterogeneity)
- **Visualization:** `matplotlib`, `seaborn`
