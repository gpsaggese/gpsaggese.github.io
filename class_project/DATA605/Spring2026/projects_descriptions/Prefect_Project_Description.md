# Prefect

## Description
- Prefect is a modern workflow orchestration tool designed for data engineering
  and data science, allowing users to build, schedule, and monitor data
  pipelines.
- It provides a user-friendly interface and Python API for defining tasks and
  workflows, making it easy to manage complex data processes.
- Prefect supports both local and cloud execution, enabling scalability and
  flexibility in deploying workflows.
- The tool includes features for error handling, retries, and logging, ensuring
  robust and reliable data pipelines.
- It integrates seamlessly with popular data storage and processing platforms,
  such as AWS, GCP, and Azure, enhancing its versatility for various data
  projects.

## Project Objective
The goal of the project is to build a robust data pipeline using Prefect that
ingests, processes, and analyzes a publicly available dataset. The project will
focus on optimizing the workflow for efficiency and reliability, allowing
students to practice orchestrating data tasks and integrating machine learning
for predictive analysis.

## Dataset Suggestions
1. **COVID-19 Data**
   - **Source**: COVID-19 Data Repository by the Center for Systems Science and
     Engineering (CSSE) at Johns Hopkins University
   - **URL**:
     [COVID-19 Data Repository](https://github.com/CSSEGISandData/COVID-19)
   - **Data Contains**: Daily confirmed cases, deaths, and recoveries globally.
   - **Access Requirements**: Publicly accessible GitHub repository.

2. **Air Quality Data**
   - **Source**: OpenAQ
   - **URL**: [OpenAQ API](https://docs.openaq.org/)
   - **Data Contains**: Air quality measurements from various locations
     worldwide, including pollutants like PM2.5 and PM10.
   - **Access Requirements**: Free API access with no authentication needed.

3. **Global Temperature Data**
   - **Source**: NASA's Goddard Institute for Space Studies (GISS)
   - **URL**:
     [GISS Surface Temperature Analysis](https://datahub.io/core/global-temp)
   - **Data Contains**: Historical global surface temperature data from 1880 to
     present.
   - **Access Requirements**: Available as a downloadable CSV file.

4. **Movie Ratings Dataset**
   - **Source**: MovieLens
   - **URL**: [MovieLens Datasets](https://grouplens.org/datasets/movielens/)
   - **Data Contains**: User ratings and movie metadata, suitable for
     collaborative filtering tasks.
   - **Access Requirements**: Free to download without authentication.

## Tasks
- **Task 1: Setup Prefect Environment**  
  Install and configure Prefect, ensuring students can run workflows locally or
  in the cloud.
- **Task 2: Data Ingestion**  
  Create a Prefect task to fetch and load data from one of the suggested
  datasets into a suitable format (e.g., Pandas DataFrame).
- **Task 3: Data Processing**  
  Implement a series of tasks for cleaning and transforming the dataset,
  including handling missing values and feature engineering.
- **Task 4: Machine Learning Model Training**  
  Use a pre-trained model or train a simple regression/classification model on
  the processed dataset, integrating it into the Prefect workflow.
- **Task 5: Workflow Monitoring and Logging**  
  Set up logging and monitoring for the workflow, allowing students to track the
  execution and handle any errors that arise.

- **Task 6: Deployment**  
  Deploy the Prefect workflow to a cloud service (e.g., Prefect Cloud or a local
  server) to demonstrate how to run the pipeline in a production-like
  environment.

## Bonus Ideas
- Extend the project by adding a real-time data ingestion component using
  streaming data.
- Compare the performance of different machine learning models and visualize the
  results within the workflow.
- Implement a notification system (e.g., email alerts) for workflow failures or
  completions.
- Explore advanced Prefect features such as dynamic tasks or parameterized
  flows.

## Useful Resources
- [Prefect Official Documentation](https://docs.prefect.io/)
- [Prefect GitHub Repository](https://github.com/PrefectHQ/prefect)
- [Prefect Examples](https://github.com/PrefectHQ/prefect/tree/main/examples)
- [OpenAQ API Documentation](https://docs.openaq.org/)
- [MovieLens Dataset Information](https://grouplens.org/datasets/movielens/)
