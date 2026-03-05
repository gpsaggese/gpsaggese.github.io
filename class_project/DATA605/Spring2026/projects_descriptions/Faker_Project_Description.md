# Faker

## Description
- Faker is a Python library that generates fake data for various purposes,
  including testing, data privacy, and development.
- It can produce a wide range of data types, such as names, addresses, emails,
  dates, and even entire profiles.
- The library supports multiple locales, allowing for the generation of
  culturally relevant data.
- Faker can be easily integrated into Python applications and can be used in
  conjunction with other libraries for data manipulation and analysis.
- It is particularly useful for creating synthetic datasets when real data is
  scarce or sensitive.

## Project Objective
The goal of this project is to create a synthetic dataset that simulates
customer transactions for an e-commerce platform. Students will generate data
that can be used for predicting customer behavior and sales trends, optimizing
marketing strategies, and analyzing purchasing patterns.

## Dataset Suggestions
1. **Kaggle E-Commerce Data**
   - Source: Kaggle
   - URL:
     [E-Commerce Data](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
   - Data Contains: Customer transactions, product details, and reviews.
   - Access Requirements: Free account on Kaggle to download datasets.

2. **Open Payments Data**
   - Source: Centers for Medicare & Medicaid Services (CMS)
   - URL: [Open Payments Data](https://openpaymentsdata.cms.gov/)
   - Data Contains: Transactions between healthcare providers and pharmaceutical
     companies.
   - Access Requirements: No authentication required; datasets can be downloaded
     directly.

3. **Fake Name Generator**
   - Source: Fake Name Generator API
   - URL: [Fake Name Generator](https://www.fakenamegenerator.com/)
   - Data Contains: Randomly generated names, addresses, and emails.
   - Access Requirements: No API key required; data can be accessed freely.

## Tasks
- **Data Generation**: Use the Faker library to create a synthetic dataset of
  customer transactions, including fields like customer ID, product ID, purchase
  date, and transaction amount.
- **Data Cleaning**: Prepare the generated data by ensuring it meets the
  required formats and removing any anomalies.
- **Exploratory Data Analysis (EDA)**: Analyze the dataset to uncover trends,
  distributions, and correlations among different features.
- **Machine Learning Model**: Implement a regression model to predict future
  sales based on historical transaction data. Use libraries like Scikit-learn
  for model training and evaluation.
- **Model Evaluation**: Assess the model's performance using metrics such as
  Mean Absolute Error (MAE) and R-squared, and visualize the results.

## Bonus Ideas
- **Advanced Modeling**: Experiment with different regression algorithms (e.g.,
  Random Forest, Gradient Boosting) and compare their performance.
- **Time Series Analysis**: Incorporate a time component to analyze seasonal
  trends in customer purchases.
- **Data Augmentation**: Generate additional features using the synthetic data,
  such as customer demographics or product categories, to enhance the model.

## Useful Resources
- [Faker Documentation](https://faker.readthedocs.io/en/master/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [Open Payments Data](https://openpaymentsdata.cms.gov/)
- [Fake Name Generator](https://www.fakenamegenerator.com/)
