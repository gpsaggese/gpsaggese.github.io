# Haystack

## Description
- Haystack is an open-source framework designed for building end-to-end search
  systems that utilize natural language processing (NLP) and machine learning.
- It supports various backends for document stores, enabling users to easily
  index and retrieve documents based on user queries.
- The framework allows for the integration of pre-trained models for tasks such
  as question-answering, summarization, and document retrieval, facilitating
  rapid development of intelligent search applications.
- Haystack provides a user-friendly API that simplifies the process of building
  complex pipelines, allowing for easy customization and scalability.
- It includes built-in support for popular NLP libraries like Hugging Face
  Transformers, making it easy to implement state-of-the-art models in search
  applications.

## Project Objective
The goal of this project is to develop a question-answering system that can
retrieve relevant information from a set of documents based on user queries. The
project will focus on optimizing the accuracy of the answers provided by the
system while ensuring efficient document retrieval.

## Dataset Suggestions
1. **SQuAD (Stanford Question Answering Dataset)**
   - **Source**: Hugging Face Datasets
   - **URL**: [SQuAD Dataset](https://huggingface.co/datasets/squad)
   - **Data**: Contains a collection of questions posed on a set of Wikipedia
     articles, along with the corresponding answers.
   - **Access Requirements**: Free to use, no authentication needed.

2. **Wikipedia Articles**
   - **Source**: Wikimedia Foundation
   - **URL**: [Wikipedia API](https://www.mediawiki.org/wiki/API:Main_page)
   - **Data**: Access to a wide range of articles covering various topics,
     useful for building a knowledge base.
   - **Access Requirements**: Free to use, no authentication needed.

3. **Natural Questions Dataset**
   - **Source**: Google Research
   - **URL**:
     [Natural Questions Dataset](https://ai.google.com/research/NaturalQuestions)
   - **Data**: Contains questions and answers derived from real user queries on
     Google, along with the corresponding Wikipedia articles.
   - **Access Requirements**: Free to use, no authentication needed.

4. **Open Trivia Database**
   - **Source**: Open Trivia Database
   - **URL**: [Open Trivia Database API](https://opentdb.com/api_config.php)
   - **Data**: A collection of trivia questions across various categories, which
     can be used to generate a diverse set of queries.
   - **Access Requirements**: Free to use, no authentication needed.

## Tasks
- **Set Up Haystack**: Install and configure Haystack in a local or Google Colab
  environment, ensuring all dependencies are met.
- **Data Ingestion**: Retrieve and preprocess the selected dataset(s) to create
  a document store that Haystack can query.
- **Pipeline Creation**: Build a question-answering pipeline using Haystack's
  components, integrating a pre-trained NLP model for answer extraction.
- **Model Training/Fine-tuning**: If needed, fine-tune the pre-trained model on
  the dataset to improve accuracy in answering questions.
- **Evaluation**: Implement metrics to evaluate the performance of the
  question-answering system, such as accuracy and F1 score, based on a
  validation set.
- **Documentation and Presentation**: Prepare a report detailing the project
  process, findings, and potential improvements for future work.

## Bonus Ideas
- **Explore Different Models**: Compare the performance of different pre-trained
  models (e.g., BERT, RoBERTa) on the question-answering task.
- **User Interface**: Develop a simple web interface that allows users to input
  questions and receive answers from the Haystack system.
- **Multi-Document Retrieval**: Implement functionality to retrieve answers from
  multiple documents and rank the results based on relevance.
- **Real-time Updates**: Integrate a feature to update the document store with
  new articles or data dynamically.

## Useful Resources
- [Haystack Documentation](https://haystack.deepset.ai/docs/introduction)
- [Haystack GitHub Repository](https://github.com/deepset-ai/haystack)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/index)
- [Hugging Face Datasets Documentation](https://huggingface.co/docs/datasets/index)
- [Kaggle Datasets](https://www.kaggle.com/datasets) - for exploring additional
  datasets related to NLP and question-answering.
