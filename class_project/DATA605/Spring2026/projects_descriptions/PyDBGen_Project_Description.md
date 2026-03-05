# PyDBGen

## Description
- PyDBGen is a Python library designed for generating synthetic databases for
  testing and development purposes.
- It allows users to create complex data models with custom schemas, including
  various data types and relationships.
- The tool supports the generation of large datasets quickly, making it ideal
  for performance testing and algorithm training.
- Users can easily customize the generated data according to specific
  requirements, including constraints and distributions.
- PyDBGen integrates seamlessly with popular data science libraries, allowing
  for easy manipulation and analysis of generated data.

## Project Objective
The goal of this project is to generate a synthetic dataset for a fictional
e-commerce platform and build a machine learning model to predict customer
purchase behavior based on their demographics and browsing history.
Specifically, students will optimize a classification model to predict whether a
customer will make a purchase.

## Dataset Suggestions
1. **Kaggle E-commerce Data**
   - **Source**: Kaggle
   - **URL**:
     [E-commerce Data](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
   - **Data Contains**: Information on customer transactions, products, and
     reviews.
   - **Access Requirements**: Free to use; requires a Kaggle account to
     download.

2. **Open Government Data - Retail Sales**
   - **Source**: Data.gov
   - **URL**:
     [Retail Trade Data](https://catalog.data.gov/dataset/retail-trade-data)
   - **Data Contains**: Monthly retail sales data across various sectors.
   - **Access Requirements**: Free to use; no authentication required.

3. **Hugging Face Datasets - Customer Reviews**
   - **Source**: Hugging Face
   - **URL**:
     [Amazon Customer Reviews](https://huggingface.co/datasets/amazon_polarity)
   - **Data Contains**: Customer reviews and ratings for various products.
   - **Access Requirements**: Free to use; requires Hugging Face account for
     some datasets.

4. **Kaggle - Online Retail Dataset**
   - **Source**: Kaggle
   - **URL**:
     [Online Retail](https://www.kaggle.com/datasets/mashlyn/online-retail)
   - **Data Contains**: Invoice-level data from a UK-based online retailer.
   - **Access Requirements**: Free to use; requires a Kaggle account to
     download.

## Tasks
- **Task 1: Data Generation**  
  Use PyDBGen to create a synthetic dataset that simulates customer demographics
  and browsing behavior for an e-commerce platform.

- **Task 2: Data Preprocessing**  
  Clean and preprocess the generated data, handling missing values and encoding
  categorical variables as necessary.

- **Task 3: Exploratory Data Analysis (EDA)**  
  Perform EDA to understand the distributions and relationships in the data,
  visualizing key features that influence purchase behavior.

- **Task 4: Model Selection and Training**  
  Choose an appropriate classification algorithm (e.g., Logistic Regression,
  Decision Trees) and train the model using the synthetic dataset.

- **Task 5: Model Evaluation**  
  Evaluate the model's performance using metrics such as accuracy, precision,
  recall, and F1 score, and perform cross-validation.

- **Task 6: Interpretation and Reporting**  
  Interpret the results, discussing the implications of the findings, and
  prepare a report summarizing the project outcomes.

## Bonus Ideas
- Extend the project by adding more features to the synthetic dataset, such as
  seasonal trends or promotional events, and analyze their impact on purchase
  behavior.
- Compare the performance of different classification algorithms and conduct
  hyperparameter tuning to optimize model performance.
- Implement an anomaly detection task to identify unusual customer behavior
  patterns in the generated dataset.

## Useful Resources
- [PyDBGen GitHub Repository](https://github.com/pydbgen/pydbgen)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [Hugging Face Datasets Documentation](https://huggingface.co/docs/datasets/index)
- [Data.gov - Open Government Data](https://www.data.gov/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/user_guide.html)
