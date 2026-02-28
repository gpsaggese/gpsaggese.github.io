# Graphene

## Description
- Graphene is a Python library designed for building GraphQL APIs easily and
  efficiently.
- It allows developers to define their schemas using Python classes, making it
  intuitive for those familiar with Python programming.
- The library supports integration with various ORMs (Object-Relational Mappers)
  to simplify data fetching and manipulation.
- Graphene provides tools for handling complex queries, mutations, and
  subscriptions, facilitating real-time data updates.
- It supports built-in tools for testing and debugging GraphQL queries,
  enhancing the development experience.

## Project Objective
The goal of this project is to build a GraphQL API that serves as a backend for
a movie recommendation system. The system will predict user preferences based on
historical ratings and provide personalized movie suggestions. Students will
optimize the recommendation algorithm to improve user satisfaction.

## Dataset Suggestions
1. **MovieLens Dataset**
   - **Source**: GroupLens Research
   - **URL**: [MovieLens Datasets](https://grouplens.org/datasets/movielens/)
   - **Data Contains**: User ratings, movie titles, genres, and user
     demographics.
   - **Access Requirements**: Free to use, no authentication required.

2. **OMDb API**
   - **Source**: OMDb API
   - **URL**: [OMDb API](http://www.omdbapi.com/)
   - **Data Contains**: Movie information including title, year, genre,
     director, and ratings.
   - **Access Requirements**: Free tier available, no authentication required
     for basic queries.

3. **Kaggle Movie Dataset**
   - **Source**: Kaggle
   - **URL**:
     [Kaggle Movie Dataset](https://www.kaggle.com/datasets/ashishpatel26/movie-dataset)
   - **Data Contains**: Information on movies like titles, genres, and user
     ratings.
   - **Access Requirements**: Free to use, requires a Kaggle account.

## Tasks
- **Task 1: Define the GraphQL Schema**  
  Create a schema that includes types for movies, users, and ratings, using
  Graphene's class-based definitions.

- **Task 2: Implement Data Fetching Logic**  
  Integrate with the MovieLens dataset to fetch user ratings and movie details,
  ensuring efficient data retrieval.

- **Task 3: Develop Queries and Mutations**  
  Implement GraphQL queries to retrieve movie recommendations based on user
  input and mutations to submit user ratings.

- **Task 4: Build the Recommendation Engine**  
  Create a basic recommendation algorithm (e.g., collaborative filtering) that
  uses the user ratings to predict and suggest movies.

- **Task 5: Testing and Debugging**  
  Utilize Graphene's built-in testing tools to ensure the API functions
  correctly and returns accurate data.

## Bonus Ideas
- **Advanced Recommendation Techniques**: Implement more sophisticated
  algorithms like content-based filtering or deep learning-based approaches for
  recommendations.
- **Real-time Updates**: Add subscription capabilities to notify users when new
  movies are added that match their preferences.
- **User Interface Development**: Create a simple front-end application using a
  framework like React or Vue.js to interact with the GraphQL API.

## Useful Resources
- [Graphene Documentation](https://docs.graphene-python.org/en/latest/)
- [GraphQL Official Documentation](https://graphql.org/learn/)
- [MovieLens Dataset](https://grouplens.org/datasets/movielens/)
- [OMDb API Documentation](http://www.omdbapi.com/)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
