# Huey

## Description
- Huey is a lightweight task queue and job scheduler for Python applications,
  designed to help manage asynchronous tasks and background jobs.
- It supports multiple message brokers, including Redis and SQLite, making it
  versatile for different project architectures.
- Huey allows for task prioritization, scheduling, and retries, which is
  essential for building robust applications that handle long-running or delayed
  tasks.
- It provides a simple API for defining and managing tasks, making it accessible
  for beginners while still powerful enough for advanced users.
- Huey integrates seamlessly with web frameworks like Flask and Django, allowing
  for easy incorporation into existing applications.

## Project Objective
The goal of this project is to build a web application that allows users to
submit data processing tasks (e.g., image processing or data analysis) that are
handled asynchronously using Huey. The project will focus on optimizing task
execution time and ensuring reliability through retries and error handling.

## Dataset Suggestions
1. **CIFAR-10 Dataset**
   - **Source**: Kaggle
   - **URL**: [CIFAR-10 Dataset](https://www.kaggle.com/c/cifar-10)
   - **Data Contains**: 60,000 32x32 color images in 10 classes, with 6,000
     images per class.
   - **Access Requirements**: Free to use; requires a Kaggle account for
     download.

2. **UCI Machine Learning Repository - Wine Quality Dataset**
   - **Source**: UCI
   - **URL**:
     [Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality)
   - **Data Contains**: Characteristics of red and white wine samples, including
     chemical properties and quality ratings.
   - **Access Requirements**: Publicly available; no authentication required.

3. **OpenAI's GPT-3 API**
   - **Source**: OpenAI
   - **URL**: [OpenAI API](https://beta.openai.com/)
   - **Data Contains**: Access to various language processing tasks, such as
     text generation and summarization.
   - **Access Requirements**: Free tier available; requires sign-up for an API
     key.

4. **Common Crawl**
   - **Source**: Common Crawl
   - **URL**: [Common Crawl](https://commoncrawl.org/)
   - **Data Contains**: A massive repository of web crawl data, including text
     and metadata from billions of web pages.
   - **Access Requirements**: Publicly available; no authentication required.

## Tasks
- **Task Definition**: Set up a Flask web application where users can submit
  data processing requests to be handled by Huey.
- **Task Implementation**: Create specific tasks for processing images or
  analyzing datasets asynchronously using Huey.
- **Task Scheduling**: Implement scheduling of tasks for periodic execution,
  such as daily data analysis or image processing.
- **Error Handling**: Add mechanisms to retry failed tasks and log errors for
  debugging purposes.
- **User Interface**: Develop a simple frontend to allow users to view the
  status of their submitted tasks.

## Bonus Ideas
- Implement a dashboard to visualize task execution times and success rates.
- Compare the performance of different message brokers (e.g., Redis vs. SQLite)
  in handling tasks.
- Extend the project to implement user authentication and allow users to manage
  their own task queues.
- Explore using pre-trained models for image classification or sentiment
  analysis on text data using the submitted tasks.

## Useful Resources
- [Huey Documentation](https://huey.readthedocs.io/en/latest/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [OpenAI API Documentation](https://beta.openai.com/docs/)
- [Common Crawl Documentation](https://commoncrawl.org/the-data/get-started/)
