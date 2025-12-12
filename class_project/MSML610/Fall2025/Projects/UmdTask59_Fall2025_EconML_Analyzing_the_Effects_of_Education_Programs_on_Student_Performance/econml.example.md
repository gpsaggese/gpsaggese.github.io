# EconML Example: Analyzing the Effects of Education Programs on Student Performance


## 1. Problem Statement
We analyze the impact of a student education program on final course grades using the UCI Student Performance dataset. The main goal is to estimate:
- The overall impact of participating in the program on the population's final grades (ATE),
- How this effect varies across different demographic and socioeconomic groups (CATEs).

## 2. Dataset Summary
We use the **Student Performance** dataset (from UCI ML Repository + Portuguese secondary school, math or Portuguese course).  
The dataset includes:
- Student demographics (sex, age, family structure),
- Family background and parental education,
- Study related and behavioral attributes,
- Final grades (G1, G2, G3) where G1 and G2 are 1st and 2nd period grades, and G3 is the final grade issued at the end of the 3rd period.

## 3. Plannned Causal Framing
- **Outcome (Y)**: final grade (G3).
- **Treatment (T)**: participation in an additional education program (interpreted from one of the support variables).
- **Features (X)**: selected demographic and socioeconomic attributes.
- **Controls (W)**: additional covariates to help adjust for confounding.

Details of the exact column choices and assumptions will be documented in this file and in `econml.example.ipynb`.

## 4. Planned Analysis Workflow
1. Load and clean the student performance data.
2. Define treatment and outcome variables.
3. Fit EconML estimators using the wrapper functions in `econml_utils.py`.
4. Estimate ATE and CATEs.
5. Visualize heterogeneity across demographic groups.
6. Discuss robustness and limitations.

The complete implementation is in `econml.example.ipynb`.