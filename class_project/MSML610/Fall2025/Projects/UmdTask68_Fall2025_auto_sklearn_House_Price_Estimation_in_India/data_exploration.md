# Data Exploration Summary

## Dataset Overview

- **Size**: 250,000 records with 23 features
- **Coverage**: 20 Indian states across 42 cities
- **Target**: Price_in_Lakhs (property prices)

## Key Findings

### Data Quality

- **Zero missing values** — all columns complete
- Clean and ready for modeling

### Feature Types

- 9 numeric features (BHK, Size, Year_Built, etc.)
- 12 categorical features (State, City, Furnished_Status, etc.)
- 1 target variable (Price_in_Lakhs)

### Target Distribution

- Broad, uniform price distribution across range
- No extreme outliers or sharp peaks

### Correlations

- **Weak linear correlations** between features and target
- Suggests non linear relationships
- Dataset appears synthetically generated
- **Implication**: Ensemble/non linear models needed (ideal for auto sklearn)

### Data Balance

- Categorical values fairly balanced
- No severe class imbalance issues
- Geographic spread across diverse regions

## Takeaway

Dataset is **clean and complex** — perfect for testing auto sklearn's ability to handle non linear prediction tasks with weak feature correlations.
