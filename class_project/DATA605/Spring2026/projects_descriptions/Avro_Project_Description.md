# Avro

## Description
- **Data Serialization Framework**: Avro is a framework for data serialization
  that provides a compact, fast, binary data format. It allows for rich data
  structures and is especially useful for data exchange between programs written
  in different languages.
- **Schema Evolution**: Avro supports schema evolution, which means that the
  data structure can change over time without breaking compatibility with
  existing data. This is essential for maintaining and evolving large datasets.
- **Integration with Big Data Tools**: Avro is commonly used with big data
  technologies like Apache Hadoop and Apache Spark, making it a popular choice
  for data processing pipelines.
- **JSON-based Schemas**: Data schemas in Avro are defined in JSON format,
  making them easy to read and manage. This also allows for seamless integration
  with other JSON-based systems.
- **Fast Serialization/Deserialization**: Avro's binary encoding allows for
  efficient serialization and deserialization of complex data structures, making
  it ideal for high-performance applications.

## Project Objective
The goal of this project is to build a data processing pipeline that ingests,
processes, and analyzes a public dataset using Avro for data serialization.
Students will optimize the pipeline for speed and efficiency, focusing on a
machine learning task such as classification or regression.

## Dataset Suggestions
1. **UCI Machine Learning Repository - Wine Quality Dataset**
   - **Source**: UCI Machine Learning Repository
   - **URL**:
     [Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality)
   - **Data Contains**: Attributes related to wine characteristics and their
     quality ratings.
   - **Access Requirements**: Publicly accessible; no authentication required.

2. **Kaggle - House Prices: Advanced Regression Techniques**
   - **Source**: Kaggle
   - **URL**:
     [House Prices Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
   - **Data Contains**: Features of houses including size, location, and
     amenities, along with their sale prices.
   - **Access Requirements**: Free to use with a Kaggle account (can be created
     for free).

3. **Hugging Face Datasets - IMDb Movie Reviews**
   - **Source**: Hugging Face Datasets
   - **URL**: [IMDb Dataset](https://huggingface.co/datasets/imdb)
   - **Data Contains**: Reviews of movies labeled as positive or negative.
   - **Access Requirements**: Publicly accessible through the Hugging Face API.

4. **Open Government Data - NYC Taxi Trip Data**
   - **Source**: NYC Open Data
   - **URL**: [NYC Taxi Trip Data](https://opendata.cityofnewyork.us/)
   - **Data Contains**: Records of taxi trips including pickup and drop-off
     locations and fare amounts.
   - **Access Requirements**: Publicly available; no authentication required.

## Tasks
- **Data Ingestion**: Use Avro to read the selected dataset and convert it into
  Avro format for efficient processing.
- **Data Processing**: Implement data cleaning and preprocessing steps, ensuring
  the data is ready for analysis.
- **Feature Engineering**: Create relevant features that will enhance the
  predictive power of the model.
- **Model Training**: Select and train a machine learning model (e.g., Random
  Forest, Logistic Regression) on the processed dataset.
- **Model Evaluation**: Evaluate the model using appropriate metrics (e.g.,
  accuracy, RMSE) and visualize the results.
- **Pipeline Optimization**: Optimize the data processing pipeline for speed and
  efficiency, ensuring the use of Avro's features for serialization.

## Bonus Ideas
- **Model Comparison**: Experiment with different machine learning algorithms
  and compare their performance.
- **Hyperparameter Tuning**: Implement hyperparameter tuning to optimize model
  performance further.
- **Real-time Data Ingestion**: Explore the possibility of integrating a
  streaming data source and processing it in real-time using Avro.
- **Deploying the Model**: Create a simple API to serve the trained model for
  predictions.

## Useful Resources
- [Apache Avro Official Documentation](https://avro.apache.org/docs/current/)
- [Avro Data Serialization on GitHub](https://github.com/apache/avro)
- [Kaggle API Documentation](https://www.kaggle.com/docs/api)
- [Hugging Face Datasets Documentation](https://huggingface.co/docs/datasets/index)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)
