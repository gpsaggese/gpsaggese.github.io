Drug–Drug Interaction Prediction using PyTorch Geometric

Course: MSML610 – Advanced Machine Learning
Semester: Fall 2025
Tool: PyTorch Geometric
Project Type: Class Project Tutorial
Difficulty Level: Hard

1. Project Motivation and Background

Drug–Drug Interactions (DDIs) occur when one drug alters the effect of another drug, potentially leading to adverse side effects or reduced therapeutic efficacy. Detecting such interactions early is an important problem in healthcare and pharmaceutical research. Many traditional machine learning approaches represent drugs as fixed feature vectors, which limits their ability to capture the rich structural information present in molecular compounds.

This project addresses this limitation by modeling drugs as graphs and applying Graph Neural Networks using PyTorch Geometric. Since molecules are naturally structured as graphs composed of atoms and chemical bonds, graph-based learning provides a more faithful representation of drug behavior. The project is designed as a hands-on tutorial that demonstrates how PyTorch Geometric can be used in a realistic end-to-end machine learning workflow.

2. Project Objective

The primary objective of this project is to predict whether a given pair of drugs will interact based on their molecular structures. Each drug is represented as a graph, and a Graph Attention Network is used to learn expressive molecular embeddings. These embeddings are then combined to perform binary classification, indicating whether an interaction exists between the two drugs.

Beyond prediction accuracy, the project also emphasizes interpretability, modular system design, reproducibility, and comparison with traditional machine learning baselines.

3. Why PyTorch Geometric is Used

PyTorch Geometric is a deep learning framework built on top of PyTorch that is specifically designed for graph-structured data. It provides efficient abstractions for message passing, graph batching, and neighborhood aggregation, which are difficult to implement correctly using standard tensor-based libraries.

In molecular machine learning tasks, PyTorch Geometric enables direct modeling of atoms as nodes and bonds as edges. This allows the model to learn from molecular structure without relying on manually engineered features, making it particularly suitable for drug interaction prediction.

4. Dataset Description

The project uses a Drug–Drug Interaction dataset obtained from Kaggle. The dataset contains pairs of drugs along with binary labels indicating whether the drugs interact. Each drug is associated with molecular structure information that is used to construct graph representations.

The dataset is naturally imbalanced, which makes evaluation metrics such as ROC-AUC and Precision-Recall AUC more appropriate than simple accuracy.

5. Graph Representation of Drugs

Each drug molecule is converted into a graph where nodes represent atoms and edges represent chemical bonds. Node features encode atomic properties, while edges capture bonding relationships. This graph representation preserves the relational and structural properties of molecules, allowing the model to reason about how different substructures contribute to drug interactions.

6. Model Architecture

The core model used in this project is a Graph Attention Network (GAT). The GAT applies attention mechanisms during message passing, allowing the model to assign different importance weights to neighboring atoms. This is particularly important in chemistry, where certain atoms or functional groups play a larger role in determining molecular behavior.

Each drug graph is passed through a shared GAT encoder to produce a fixed-length embedding. The embeddings of two drugs are then combined and passed through a classifier to predict whether the drugs interact.

7. Training and Evaluation Strategy

The model is trained using supervised learning with binary cross-entropy loss. Performance is evaluated using ROC-AUC and Precision-Recall AUC metrics, which are well-suited for imbalanced datasets.

The training pipeline is modular, separating data loading, graph construction, model definition, training, and evaluation into reusable components.

8. Comparison with Traditional Machine Learning Baseline

As a baseline, the project implements a traditional cheminformatics approach using Morgan fingerprints combined with logistic regression. This baseline is commonly used in drug interaction tasks and provides a strong point of comparison.

While the baseline achieves reasonable performance, it relies on fixed, hand-engineered features. In contrast, the graph-based model learns task-specific representations directly from molecular structure, making it more flexible and extensible.

9. Why the Graph Neural Network Model is Better

The graph-based model offers several advantages over traditional approaches. First, it operates directly on molecular graphs, preserving structural information that is lost in flattened feature vectors. Second, the attention mechanism enables the model to focus on chemically important atoms and substructures. Third, the learned representations adapt during training, whereas traditional fingerprints remain static.

Even when evaluation metrics are comparable, the graph neural network provides superior interpretability and a stronger foundation for future extensions.

10. Model Interpretability using GNNExplainer

To improve transparency, the project includes model interpretability using GNNExplainer. This method identifies which atoms and bonds contribute most strongly to a predicted interaction. By highlighting important substructures, the model’s predictions become more understandable and trustworthy.

Interpretability is particularly important in healthcare applications, where model decisions must be explainable.

11. Self-Supervised Pretraining (Bonus Component)

As an additional enhancement, the project includes a self-supervised pretraining step using a contrastive learning objective. The graph encoder is pretrained on augmented molecular graphs without using interaction labels. This allows the model to learn more robust molecular representations from unlabeled data.

After fine-tuning on the interaction prediction task, the pretrained model shows a consistent improvement in ROC-AUC, demonstrating the benefit of leveraging unlabeled chemical data.

12. Project Structure

The project follows a clean and modular structure consistent with the course tutorial template. All reusable logic is implemented in Python utility modules, while notebooks focus on documentation, experimentation, and results. The separation between API and example layers ensures clarity and maintainability.

The project directory is organized as follows:

class_project/MSML610/Fall2025/projects/
└── UmdTask23_Fall2025_PyTorch_Geometric_Drug_Drug_Interaction_Prediction/
    ├── README.md
    ├── Dockerfile
    ├── db_drug_interactions.csv
    ├── Drug_utils.py
    ├── utils_data_io.py
    ├── utils_post_processing.py
    ├── Drug_API.ipynb
    ├── Drug_API.md
    ├── Drug_example.ipynb
    └── Drug_example.md


The utility files contain reusable functions for data loading, graph construction, model definition, training, and evaluation. The API files document the internal interfaces and abstractions built on top of PyTorch Geometric. The example files demonstrate a complete end-to-end application of the system.

13. Docker and Reproducibility

Step 1: Build the Docker Image

From the project root directory (where the Dockerfile is located), run:

docker build -t drug-ddi-project .


This command:

Installs all required Python packages

Sets up RDKit, PyTorch, and PyTorch Geometric

Prepares the environment for running the project

Step 2: Run the Docker Container

After the image is built successfully, run:

docker run -it --rm drug-ddi-project


This will:

Start the container

Execute the project in the configured environment

Run the pipeline exactly as intended by the project setup

Optional: Running with Jupyter Notebook

If you want to interact with the notebook inside Docker, you can expose a port:

docker run -it -p 8888:8888 drug-ddi-project


Then open the Jupyter URL printed in the terminal in your browser.

Notes on Reproducibility

Docker ensures that the same versions of libraries are used across different systems

This avoids issues related to local environment differences

The project can be run consistently on any machine that supports Docker

Common Issues

If Docker reports permission errors, ensure Docker is running and your user has access

If the dataset is missing, Docker will raise an error indicating the file path

14. Learning Outcomes

This project demonstrates how graph-based deep learning can be applied to real-world problems. It shows how PyTorch Geometric integrates into modern machine learning workflows and highlights the advantages of graph neural networks over traditional approaches in molecular prediction tasks.

15. Conclusion

This project presents an end-to-end system for drug–drug interaction prediction using PyTorch Geometric. By modeling drugs as graphs and applying graph neural networks, the system captures structural information that traditional methods cannot. The resulting model is flexible, interpretable, and well-aligned with modern data science and machine learning practices.