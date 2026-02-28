# Alipy

## Description
- **Active Learning Framework**: Alipy is a Python library designed to
  facilitate active learning, enabling users to efficiently select the most
  informative data points for labeling.
- **Multiple Query Strategies**: The tool supports various query strategies,
  including uncertainty sampling, query-by-committee, and diversity-based
  methods, allowing users to tailor their approach based on specific project
  needs.
- **Integration with Machine Learning Libraries**: Alipy is compatible with
  popular machine learning libraries like Scikit-learn, making it easy to
  integrate into existing workflows.
- **User-Friendly API**: The library features an intuitive API that simplifies
  the implementation of active learning processes, making it accessible even for
  those new to the concept.
- **Evaluation Metrics**: Alipy provides built-in metrics to evaluate the
  effectiveness of different query strategies, helping users to understand their
  performance and make data-driven decisions.

## Project Objective
The goal of this project is to build a machine learning model for classifying
handwritten digits from the MNIST dataset using active learning techniques. The
project aims to optimize the model's performance while minimizing the number of
labeled samples needed for training.

## Dataset Suggestions
1. **MNIST Handwritten Digits**
   - **Source**: Kaggle
   - **URL**: [MNIST Dataset](https://www.kaggle.com/c/digit-recognizer/data)
   - **Data Contains**: Images of handwritten digits (0-9) along with their
     corresponding labels.
   - **Access Requirements**: Free to use; requires creating a Kaggle account.

2. **Fashion MNIST**
   - **Source**: Kaggle
   - **URL**:
     [Fashion MNIST Dataset](https://www.kaggle.com/zalando-research/fashionmnist)
   - **Data Contains**: Images of clothing items categorized into 10 classes
     (e.g., T-shirt, Trouser, Pullover).
   - **Access Requirements**: Free to use; requires creating a Kaggle account.

3. **CIFAR-10**
   - **Source**: Kaggle
   - **URL**: [CIFAR-10 Dataset](https://www.kaggle.com/c/cifar-10)
   - **Data Contains**: 60,000 32x32 color images in 10 classes, with 6,000
     images per class.
   - **Access Requirements**: Free to use; requires creating a Kaggle account.

## Tasks
- **Data Preparation**: Load the chosen dataset and preprocess the images (e.g.,
  normalization, reshaping) for use in the model.
- **Model Selection**: Choose a suitable machine learning model (e.g., a simple
  CNN or a decision tree) for digit classification.
- **Active Learning Implementation**: Utilize Alipy to implement active learning
  strategies to select the most informative samples for labeling.
- **Model Training and Evaluation**: Train the model using the labeled samples
  and evaluate its performance using standard metrics (e.g., accuracy,
  precision, recall).
- **Analysis of Active Learning**: Compare the performance of the model with and
  without active learning to assess its effectiveness.

## Bonus Ideas
- **Experiment with Different Query Strategies**: Implement multiple query
  strategies available in Alipy and compare their effectiveness on the chosen
  dataset.
- **Hyperparameter Tuning**: Use techniques like grid search or random search to
  optimize hyperparameters of the chosen model.
- **Visualization of Learning Process**: Create visualizations to show how the
  model's performance improves with the addition of labeled samples through
  active learning.
- **Transfer Learning**: Investigate the use of pre-trained models to enhance
  classification accuracy and reduce training time.

## Useful Resources
- [Alipy Documentation](https://alipy.readthedocs.io/en/latest/)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [Active Learning in Machine Learning: A Survey](https://www.sciencedirect.com/science/article/pii/S0957417421000196)
- [GitHub Repository for Alipy](https://github.com/Alipy/Alipy)
