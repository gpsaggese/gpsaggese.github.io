# Gradio

## Description
- Gradio is an open-source Python library that allows users to create
  customizable user interfaces for machine learning models in just a few lines
  of code.
- It supports a variety of input and output types, including text, images,
  audio, and video, making it versatile for different ML applications.
- Gradio enables real-time interaction with models, allowing users to test and
  visualize predictions instantly through a web interface.
- The tool seamlessly integrates with popular ML frameworks like TensorFlow,
  PyTorch, and Scikit-learn, facilitating easy deployment of models.
- Gradio also provides the option to share interfaces with others via a unique
  URL, enhancing collaboration and feedback collection.

## Project Objective
The goal of this project is to build an interactive web application that allows
users to classify images of handwritten digits using a machine learning model.
The project will focus on optimizing the model's accuracy and user experience
through Gradio's interface.

## Dataset Suggestions
1. **MNIST Handwritten Digits Dataset**
   - **Source**: Kaggle
   - **URL**: [MNIST Dataset](https://www.kaggle.com/c/digit-recognizer/data)
   - **Data Contains**: Images of handwritten digits (0-9) along with their
     corresponding labels.
   - **Access Requirements**: Free to use; requires a Kaggle account for
     download.

2. **Fashion MNIST Dataset**
   - **Source**: Kaggle
   - **URL**:
     [Fashion MNIST](https://www.kaggle.com/zalando-research/fashionmnist)
   - **Data Contains**: Images of clothing items categorized into 10 classes
     (e.g., T-shirt, dress, sneaker).
   - **Access Requirements**: Free to use; requires a Kaggle account for
     download.

3. **CIFAR-10 Dataset**
   - **Source**: TensorFlow Datasets
   - **URL**:
     [CIFAR-10](https://www.tensorflow.org/datasets/community_catalog/hub/tensorflow/cifar10)
   - **Data Contains**: 60,000 32x32 color images in 10 different classes, with
     6,000 images per class.
   - **Access Requirements**: Free to use; available directly through TensorFlow
     Datasets.

## Tasks
- **Data Exploration**: Load and visualize the dataset to understand the
  distribution of classes and characteristics of the images.
- **Model Selection**: Choose a suitable pre-trained model (e.g., CNN) for image
  classification and fine-tune it using the selected dataset.
- **Model Training**: Train the model on the dataset, ensuring to monitor
  performance metrics like accuracy and loss.
- **Gradio Interface Development**: Create a Gradio interface that allows users
  to upload images and receive predictions from the trained model.
- **User Testing**: Gather feedback on the interface from peers and iterate on
  the design and functionality based on user experience.

## Bonus Ideas
- Implement an additional feature that allows users to view the confidence
  scores for each prediction.
- Compare the performance of different models (e.g., CNN vs. traditional ML
  classifiers) and visualize the results using Gradio.
- Challenge students to add image preprocessing options (e.g., rotation, zoom)
  in the Gradio interface before the model makes predictions.

## Useful Resources
- [Gradio Documentation](https://gradio.app/docs/)
- [Kaggle MNIST Dataset](https://www.kaggle.com/c/digit-recognizer/data)
- [Kaggle Fashion MNIST Dataset](https://www.kaggle.com/zalando-research/fashionmnist)
- [CIFAR-10 Dataset on TensorFlow](https://www.tensorflow.org/datasets/community_catalog/hub/tensorflow/cifar10)
- [GitHub - Gradio Examples](https://github.com/gradio-app/gradio/tree/main/examples)
