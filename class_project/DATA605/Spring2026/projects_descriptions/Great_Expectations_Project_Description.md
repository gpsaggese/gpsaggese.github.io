# Great Expectations

## Description
- Great Expectations is an open-source Python library designed for data quality
  and validation, enabling data teams to define, document, and validate data
  expectations.
- It provides a framework for creating "expectations," which are assertions
  about data properties that help ensure data quality throughout the data
  pipeline.
- The tool integrates seamlessly with various data sources, including databases,
  data frames, and data lakes, allowing for flexible data validation.
- Great Expectations supports data profiling, allowing users to generate reports
  that summarize data distributions and quality metrics.
- It includes built-in support for generating documentation, making it easier
  for teams to communicate data expectations and quality standards.

## Project Objective
The goal of this project is to implement data validation and quality checks
using Great Expectations on a publicly available dataset. Students will optimize
the data quality by identifying and correcting anomalies, ensuring that the data
meets predefined expectations for further analysis or modeling.

## Dataset Suggestions
1. **California Housing Prices**
   - **Source**: Kaggle
   - **URL**:
     [California Housing Prices Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
   - **Data Contains**: Features related to housing prices, including location,
     size, number of rooms, etc.
   - **Access Requirements**: Free to use after creating a Kaggle account.

2. **US Flight Delays**
   - **Source**: Bureau of Transportation Statistics
   - **URL**:
     [US Flight Delays Dataset](https://www.transtats.bts.gov/OT_Delay/OT_Delay)
   - **Data Contains**: Information on flight delays, including departure and
     arrival times, airlines, and weather conditions.
   - **Access Requirements**: Open access, no authentication needed.

3. **World Happiness Report**
   - **Source**: Kaggle
   - **URL**:
     [World Happiness Report Dataset](https://www.kaggle.com/unsdsn/world-happiness)
   - **Data Contains**: Happiness scores and factors affecting happiness across
     different countries.
   - **Access Requirements**: Free to use after creating a Kaggle account.

4. **COVID-19 Cases and Vaccination Data**
   - **Source**: Our World in Data
   - **URL**: [COVID-19 Data](https://covid.ourworldindata.org/)
   - **Data Contains**: Daily COVID-19 cases, deaths, and vaccination rates
     globally.
   - **Access Requirements**: Open access, no authentication needed.

## Tasks
- **Setup Great Expectations**: Install Great Expectations and configure it for
  the chosen dataset to prepare for data validation.
- **Define Expectations**: Create a set of expectations for the dataset,
  including checks for missing values, data types, and value ranges.
- **Validate Data**: Run the expectations against the dataset to identify any
  data quality issues or anomalies.
- **Generate Documentation**: Use Great Expectations to create a data quality
  report that documents the expectations and validation results.
- **Refine Dataset**: Based on validation results, clean and preprocess the
  dataset to address any identified issues.

## Bonus Ideas
- Implement a comparison of data quality before and after applying data
  validation techniques.
- Explore the integration of Great Expectations with other tools like Apache
  Airflow for automated data quality checks in data pipelines.
- Challenge students to create custom expectations specific to their dataset and
  evaluate their effectiveness.

## Useful Resources
- [Great Expectations Official Documentation](https://docs.greatexpectations.io/)
- [Great Expectations GitHub Repository](https://github.com/great-expectations/great_expectations)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [Bureau of Transportation Statistics](https://www.bts.gov/)
- [Our World in Data COVID-19 Dataset](https://covid.ourworldindata.org/)
