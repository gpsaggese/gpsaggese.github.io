# Small-Text

## Description
- **small-text** is a Python library designed for efficient training and
  evaluation of machine learning models on small datasets, particularly in
  natural language processing (NLP) tasks.
- It supports few-shot learning, allowing users to leverage pre-trained models
  and adapt them to new tasks with minimal data.
- The library provides a straightforward API for fine-tuning transformer models,
  making it accessible for students and practitioners alike.
- It includes features for data augmentation and active learning, enhancing
  model performance even with limited labeled data.
- **small-text** integrates seamlessly with popular libraries like Hugging
  Face's Transformers, enabling easy access to state-of-the-art NLP models.

## Project Objective
The goal of this project is to develop a sentiment analysis model that
classifies movie reviews as positive or negative using a small dataset. The
project will optimize the model's accuracy while minimizing the amount of
labeled data required for effective training.

## Dataset Suggestions
1. **IMDb Movie Reviews**
   - **Source**: Kaggle
   - **URL**:
     [IMDb Movie Reviews Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
   - **Data Contains**: 50,000 movie reviews labeled as positive or negative.
   - **Access Requirements**: Free to use after signing up for a Kaggle account.

2. **Sentiment140**
   - **Source**: Sentiment140
   - **URL**: [Sentiment140 Dataset](http://help.sentiment140.com/for-students/)
   - **Data Contains**: 1.6 million tweets labeled for sentiment (positive,
     negative).
   - **Access Requirements**: Free to use, no authentication needed.

3. **Hugging Face Datasets**
   - **Source**: Hugging Face
   - **URL**: [Hugging Face Datasets](https://huggingface.co/datasets)
   - **Data Contains**: Various sentiment analysis datasets (e.g., Amazon
     Reviews, Twitter Sentiment).
   - **Access Requirements**: Free to use, can be accessed via the Hugging Face
     library.

4. **Yelp Reviews**
   - **Source**: Yelp Open Dataset
   - **URL**: [Yelp Open Dataset](https://www.yelp.com/dataset)
   - **Data Contains**: Reviews of businesses with star ratings (1 to 5 stars).
   - **Access Requirements**: Free to use, requires agreement to Yelp's terms of
     use.

## Tasks
- **Data Exploration**: Conduct exploratory data analysis (EDA) on the chosen
  dataset to understand the distribution of labels and text characteristics.
- **Data Preprocessing**: Clean and preprocess the text data, including
  tokenization and normalization, to prepare it for model training.
- **Few-Shot Learning Setup**: Utilize the small-text library to set up a
  few-shot learning environment, selecting a small subset of labeled data for
  training.
- **Model Fine-Tuning**: Fine-tune a pre-trained transformer model (e.g., BERT)
  using the small-text library to adapt it specifically for sentiment analysis.
- **Model Evaluation**: Evaluate the model's performance on a validation set
  using metrics such as accuracy, precision, and recall.
- **Results Analysis**: Analyze the results and visualize the model's
  performance, discussing potential improvements and challenges encountered.

## Bonus Ideas
- Experiment with different data augmentation techniques to increase the size of
  the training dataset artificially.
- Compare the performance of the few-shot model against a baseline model trained
  on a larger dataset.
- Implement active learning strategies to iteratively select the most
  informative samples for labeling and retraining the model.
- Extend the project to include multi-class sentiment classification (e.g.,
  neutral, positive, negative).

## Useful Resources
- [small-text Documentation](https://small-text.readthedocs.io/en/latest/)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/index)
- [Kaggle API Documentation](https://www.kaggle.com/docs/api)
- [Sentiment140 Dataset Documentation](http://help.sentiment140.com/for-students/)
- [Yelp Dataset Documentation](https://www.yelp.com/dataset/documentation)
