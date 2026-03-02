# Celery

## Description
- Celery is an open-source distributed task queue system that enables the
  execution of asynchronous tasks and scheduling of jobs in Python applications.
- It supports a variety of message brokers (like RabbitMQ and Redis), allowing
  for flexible task management and communication between different components of
  an application.
- Celery is designed to handle large volumes of tasks, making it ideal for
  applications that require background processing or long-running tasks.
- The tool includes features such as task retries, scheduling, and monitoring,
  which help ensure that tasks are executed reliably and efficiently.
- Celery integrates seamlessly with web frameworks like Django and Flask, making
  it easy to incorporate into existing applications for improved performance.

## Project Objective
The goal of this project is to build a web application that allows users to
submit data processing tasks (e.g., data transformation, model training)
asynchronously using Celery. Students will optimize the task execution time and
monitor the performance of their tasks.

## Dataset Suggestions
1. **Kaggle - House Prices: Advanced Regression Techniques**
   - **URL**:
     [Kaggle House Prices Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
   - **Data Contains**: Features of houses in Ames, Iowa, including sale prices,
     which can be used for regression tasks.
   - **Access Requirements**: Free account on Kaggle to download the dataset.

2. **UCI Machine Learning Repository - Wine Quality Dataset**
   - **URL**:
     [Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality)
   - **Data Contains**: Attributes related to red and white wine
     characteristics, with quality ratings for classification tasks.
   - **Access Requirements**: Publicly available without authentication.

3. **Hugging Face Datasets - IMDb Movie Reviews**
   - **URL**: [IMDb Dataset](https://huggingface.co/datasets/imdb)
   - **Data Contains**: A collection of movie reviews for binary sentiment
     classification (positive/negative).
   - **Access Requirements**: Free to use via the Hugging Face library.

4. **Open Government Data - COVID-19 Vaccination Data**
   - **URL**:
     [COVID-19 Vaccination Data](https://data.cdc.gov/dataset/COVID-19-Vaccination-Data)
   - **Data Contains**: Daily vaccination counts and demographic information for
     COVID-19 vaccinations.
   - **Access Requirements**: Publicly accessible without authentication.

## Tasks
- **Set Up Celery**: Install and configure Celery with a message broker (e.g.,
  Redis) to manage task queues.
- **Create Asynchronous Tasks**: Develop tasks for data processing or model
  training that can be executed asynchronously.
- **Implement Task Monitoring**: Use Celery's built-in monitoring tools to track
  task execution times and success rates.
- **Optimize Task Performance**: Experiment with different configurations (e.g.,
  concurrency settings) to improve the execution speed of tasks.
- **Build a Web Interface**: Create a simple web application (using Flask or
  Django) where users can submit tasks and view the status of their submissions.

## Bonus Ideas
- Implement a retry mechanism for failed tasks and analyze the impact on overall
  performance.
- Compare the execution times of synchronous vs. asynchronous task processing
  using Celery.
- Extend the project to include user authentication and role-based access
  control for task submissions.

## Useful Resources
- [Celery Official Documentation](https://docs.celeryproject.org/en/stable/)
- [Redis Documentation](https://redis.io/documentation)
- [Flask Documentation](https://flask.palletsprojects.com/en/2.0.x/)
- [Django Documentation](https://docs.djangoproject.com/en/stable/)
- [GitHub - Celery Examples](https://github.com/celery/celery/tree/master/examples)
