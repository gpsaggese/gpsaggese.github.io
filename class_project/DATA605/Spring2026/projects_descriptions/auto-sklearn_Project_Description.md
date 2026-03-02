# Auto-Sklearn

## Description
- **Automated Machine Learning**: auto-sklearn is an open-source tool that
  automates the process of selecting machine learning algorithms and
  hyperparameters, making it easier for users to develop effective models
  without extensive machine learning expertise.
- **Ensemble Learning**: It automatically constructs ensembles of different
  models, which can lead to improved predictive performance by combining the
  strengths of various algorithms.
- **Feature Engineering**: The tool includes built-in functionality for
  preprocessing and feature selection, helping to optimize the input data for
  better model performance.
- **Meta-Learning**: It leverages meta-learning techniques to learn from past
  experiences and datasets to make informed decisions about model selection and
  hyperparameter tuning.
- **Scikit-Learn Integration**: auto-sklearn is built on top of Scikit-Learn,
  making it compatible with the Scikit-Learn ecosystem and allowing users to
  easily integrate it into existing workflows.

## Project Objective
The goal of this project is to build a predictive model that can classify images
of handwritten digits from the MNIST dataset. Students will optimize the model's
accuracy and evaluate its performance using various metrics.

## Dataset Suggestions
1. **MNIST Handwritten Digits**
   - **Source**: Kaggle
   - **URL**: [MNIST Dataset](https://www.kaggle.com/c/digit-recognizer/data)
   - **Data Contains**: 70,000 images of handwritten digits (0-9) with pixel
     values.
   - **Access Requirements**: Free to use; no authentication required.

2. **Fashion MNIST**
   - **Source**: Kaggle
   - **URL**:
     [Fashion MNIST Dataset](https://www.kaggle.com/zalando-research/fashionmnist)
   - **Data Contains**: 70,000 grayscale images of clothing items classified
     into 10 categories.
   - **Access Requirements**: Free to use; no authentication required.

3. **CIFAR-10**
   - **Source**: Kaggle
   - **URL**: [CIFAR-10 Dataset](https://www.kaggle.com/c/cifar-10)
   - **Data Contains**: 60,000 32x32 color images in 10 different classes (e.g.,
     airplane, car, bird).
   - **Access Requirements**: Free to use; no authentication required.

4. **Street View House Numbers (SVHN)**
   - **Source**: SVHN Dataset Official Site
   - **URL**: [SVHN Dataset](http://ufldl.stanford.edu/housenumbers/)
   - **Data Contains**: Real-world images of house numbers collected from Google
     Street View, labeled with digit classes.
   - **Access Requirements**: Free to use; no authentication required.

## Tasks
- **Data Loading and Preprocessing**: Load the chosen dataset into the
  environment, perform necessary preprocessing steps (e.g., normalization,
  reshaping).
- **Model Training with auto-sklearn**: Utilize auto-sklearn to automatically
  select and train various models on the dataset.
- **Hyperparameter Optimization**: Experiment with hyperparameter settings to
  improve model accuracy and performance.
- **Model Evaluation**: Evaluate the trained models using metrics such as
  accuracy, precision, recall, and F1-score.
- **Result Comparison**: Compare the performance of the auto-sklearn model
  against a manually selected model (e.g., a simple logistic regression) to
  understand the benefits of automation.

## Bonus Ideas
- **Ensemble Techniques**: Explore additional ensemble techniques beyond what
  auto-sklearn provides and compare their performance.
- **Transfer Learning**: Investigate using pre-trained convolutional neural
  networks as a feature extractor before applying auto-sklearn models.
- **Visualization**: Create visualizations of model performance and confusion
  matrices to better understand classification results.
- **Real-time Prediction**: Develop a simple web app using Flask to allow users
  to input handwritten digits and receive predictions from the trained model.

## Useful Resources
- [auto-sklearn Documentation](https://automl.github.io/auto-sklearn/master/)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [Scikit-Learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [GitHub Repository for auto-sklearn](https://github.com/automl/auto-sklearn)
- [Introduction to AutoML](https://towardsdatascience.com/introduction-to-automated-machine-learning-automl-7d967c4e1e0b)
