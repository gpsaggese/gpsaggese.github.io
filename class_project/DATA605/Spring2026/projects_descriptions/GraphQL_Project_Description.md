# GraphQL

## Description
- GraphQL is a query language for APIs that allows clients to request only the
  data they need, making data retrieval more efficient.
- It provides a runtime for executing those queries by using a type system that
  defines the capabilities of the API.
- GraphQL enables developers to create a single endpoint for all data requests,
  simplifying the API structure compared to traditional REST APIs.
- It supports real-time data updates through subscriptions, allowing clients to
  receive live updates whenever data changes.
- The schema is strongly typed, which helps in validating queries and responses,
  improving the robustness of applications.

## Project Objective
The goal of this project is to build a data-driven application that predicts the
popularity of movies based on various features such as genre, release year, and
user ratings. Students will optimize a regression model to estimate the expected
box office revenue for upcoming movies.

## Dataset Suggestions
1. **Kaggle Movies Dataset**
   - **Source Name**: Kaggle
   - **URL**:
     [Kaggle Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)
   - **Data Contains**: Information on movies including titles, genres, release
     dates, ratings, and revenue.
   - **Access Requirements**: Free account on Kaggle to download the dataset.

2. **The Movie Database (TMDb) API**
   - **Source Name**: TMDb
   - **API Endpoint**:
     [TMDb API](https://developers.themoviedb.org/3/getting-started/introduction)
   - **Data Contains**: Movie details, including genres, budgets, revenue, and
     user ratings.
   - **Access Requirements**: Free API key required (easy to obtain).

3. **Open Movie Database (OMDb) API**
   - **Source Name**: OMDb
   - **API Endpoint**: [OMDb API](http://www.omdbapi.com/)
   - **Data Contains**: Movie details including title, year, genre, and box
     office earnings.
   - **Access Requirements**: Free access without authentication.

4. **Kaggle IMDB Movie Reviews Dataset**
   - **Source Name**: Kaggle
   - **URL**:
     [IMDB Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-movie-reviews)
   - **Data Contains**: Reviews and ratings which can be used to assess
     sentiment towards movies.
   - **Access Requirements**: Free account on Kaggle to download the dataset.

## Tasks
- **Data Collection**: Use GraphQL to fetch movie data from TMDb or OMDb API,
  ensuring to select relevant fields for analysis.
- **Data Preprocessing**: Clean and preprocess the collected data, handling
  missing values and encoding categorical variables such as genres.
- **Exploratory Data Analysis (EDA)**: Conduct EDA to visualize relationships
  between features (e.g., genre vs. revenue) and identify patterns.
- **Model Development**: Implement a regression model (e.g., linear regression,
  random forest) to predict box office revenue based on the features.
- **Model Evaluation**: Assess model performance using metrics such as RMSE and
  R², and refine the model based on evaluation results.
- **Visualization**: Create visualizations to present findings, including
  predicted vs. actual revenue comparisons.

## Bonus Ideas
- **Feature Engineering**: Experiment with additional features such as user
  sentiment from reviews to improve model accuracy.
- **Model Comparison**: Compare the performance of different regression models
  and discuss the trade-offs.
- **Real-time Updates**: Implement a subscription feature using GraphQL to
  receive real-time updates on movie data and predictions.
- **Deployment**: Deploy the model as a simple web application where users can
  input movie features to get revenue predictions.

## Useful Resources
- [GraphQL Official Documentation](https://graphql.org/learn/)
- [TMDb API Documentation](https://developers.themoviedb.org/3/getting-started/introduction)
- [OMDb API Documentation](http://www.omdbapi.com/)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [Scikit-learn Documentation for Regression](https://scikit-learn.org/stable/supervised_learning.html#supervised-learning)
