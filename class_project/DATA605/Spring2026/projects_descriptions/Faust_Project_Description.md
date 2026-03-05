# Faust

## Description
- **Faust** is a stream processing library for Python that allows for the
  building of real-time data processing applications.
- It provides a simple and intuitive syntax for defining data transformations
  and processing logic, making it accessible for both beginners and experienced
  developers.
- Faust integrates seamlessly with Kafka, allowing for easy consumption and
  production of messages in real-time, making it suitable for event-driven
  architectures.
- The library supports windowing, stateful processing, and event time handling,
  enabling complex stream processing scenarios.
- Faust also includes built-in support for monitoring and managing stream
  processing applications, providing insights into performance and data flow.

## Project Objective
The goal of this project is to build a real-time sentiment analysis application
that processes tweets about a specific topic using Faust. The application will
classify tweets as positive, negative, or neutral based on their sentiment,
optimizing for accuracy and response time.

## Dataset Suggestions
1. **Sentiment140 Dataset**
   - **Source**: Kaggle
   - **URL**:
     [Sentiment140 Dataset](https://www.kaggle.com/kazanova/sentiment140)
   - **Data Contains**: 1.6 million tweets labeled with sentiment
     (positive/negative).
   - **Access Requirements**: Free to download after creating a Kaggle account.

2. **Twitter API (Filtered Stream)**
   - **Source**: Twitter Developer
   - **URL**:
     [Twitter API](https://developer.twitter.com/en/docs/twitter-api/tweets/filtered-stream/introduction)
   - **Data Contains**: Real-time tweets matching specific keywords (e.g.,
     "climate change").
   - **Access Requirements**: Requires a Twitter Developer account, but the
     filtered stream is free to use.

3. **Hugging Face Datasets - Twitter Sentiment Analysis**
   - **Source**: Hugging Face
   - **URL**:
     [Hugging Face Twitter Sentiment](https://huggingface.co/datasets/twitter-sentiment)
   - **Data Contains**: A collection of tweets with sentiment labels.
   - **Access Requirements**: Publicly accessible and free to use.

4. **Kaggle - COVID-19 Tweets**
   - **Source**: Kaggle
   - **URL**:
     [COVID-19 Tweets](https://www.kaggle.com/datasets/sbhatti/covid19-tweets)
   - **Data Contains**: Tweets related to COVID-19 with sentiment labels.
   - **Access Requirements**: Free to download after creating a Kaggle account.

## Tasks
- **Set Up Faust Environment**: Install Faust and set up a local Kafka instance
  to run the stream processing application.
- **Data Ingestion**: Implement a Kafka consumer in Faust to ingest tweets from
  the chosen dataset or Twitter API in real-time.
- **Sentiment Analysis Model**: Utilize a pre-trained sentiment analysis model
  (e.g., from Hugging Face) to classify the sentiment of incoming tweets.
- **Data Processing Logic**: Define the processing logic in Faust to transform,
  filter, and classify the tweets based on their sentiment.
- **Monitoring and Visualization**: Set up monitoring tools to visualize the
  data flow and performance metrics of the stream processing application.

## Bonus Ideas
- Implement a dashboard using Streamlit or Dash to visualize sentiment trends
  over time.
- Compare the performance of the Faust application with a batch processing
  approach using Pandas.
- Experiment with different sentiment analysis models and evaluate their
  accuracy on the same dataset.
- Add a feature to store the processed tweets and their sentiments in a database
  for historical analysis.

## Useful Resources
- [Faust Documentation](https://faust.readthedocs.io/en/latest/)
- [Kafka Documentation](https://kafka.apache.org/documentation/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [Twitter Developer Documentation](https://developer.twitter.com/en/docs)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
