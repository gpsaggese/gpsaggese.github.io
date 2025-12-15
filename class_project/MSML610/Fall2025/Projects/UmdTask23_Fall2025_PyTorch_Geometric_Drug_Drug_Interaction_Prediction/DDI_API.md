# Drug API – Drug–Drug Interaction Prediction

This document describes the API and design choices behind the Drug–Drug Interaction (DDI) prediction tool implemented using PyTorch Geometric. The API provides a clean and stable interface for molecular graph construction, model definition, training, and inference. It is designed to support experimentation in a notebook while keeping core logic modular and reusable.

---

## Tool Overview: PyTorch Geometric

PyTorch Geometric (PyG) is a graph deep learning library built on top of PyTorch. It provides efficient data structures and message-passing abstractions for learning on graph-structured data.

In this project, PyG is used to:
- Represent molecules as graphs using `Data` objects
- Batch molecular graphs efficiently during training
- Apply graph attention layers for message passing
- Pool node-level embeddings into graph-level representations

---

## Project Context

- **Project**: Drug–Drug Interaction Prediction  
- **Objective**: Predict whether two drugs interact based on molecular structure  
- **Input**: Two drug molecules represented as molecular graphs  
- **Output**: Interaction probability (binary classification)  
- **Evaluation Metrics**: ROC-AUC and PR-AUC  

The project focuses on learning molecular representations directly from chemical structure rather than relying solely on handcrafted descriptors.

---

## API Design Philosophy

The API follows three guiding principles:

1. **Separation of concerns**  
   Graph construction, model definition, and training logic are separated from the notebook.

2. **Stable interfaces**  
   Functions and classes expose simple and predictable inputs and outputs.

3. **Notebook-friendly usage**  
   All components can be called directly from an interactive notebook without complex setup.

---

## Core Data Structures

### Molecular Graph Representation

Each drug molecule is represented as a `torch_geometric.data.Data` object with the following fields:

- `x`: Node feature matrix of shape `[num_atoms, num_features]`
- `edge_index`: Undirected bond connectivity of shape `[2, num_edges]`
- `batch` (optional): Maps each node to a graph ID when batching multiple graphs

---

## Configuration

### TrainConfig

```python
@dataclass
class TrainConfig:
    lr: float
    weight_decay: float
    epochs: int
    batch_size: int
    device: str


## Graph Construction

smiles_to_pyg_graph

smiles_to_pyg_graph(smiles: str) -> Data


This function converts a SMILES string into a PyTorch Geometric molecular graph.

## Node Features

- Atomic number

- Atom degree

- Formal charge

- Aromaticity indicator

These features are intentionally simple and interpretable, serving as a clear baseline for graph-based learning.

## Edges

- Chemical bonds are represented as undirected edges

- Each bond is added in both directions

The output is a Data object compatible with PyG batching and GNN layers.


# Model Components
DrugGATEncoder

The DrugGATEncoder converts a single molecular graph into a fixed-length embedding.

## Inputs

- x: Node feature matrix [num_atoms, in_dim]

- edge_index: Edge connectivity [2, num_edges]

- batch (optional): Node-to-graph assignment

If batch is not provided, the encoder assumes a single molecular graph.

## Output

- Graph embedding of shape [num_graphs, embedding_dim]

## Architecture

- Two Graph Attention (GAT) convolution layers

- ELU activation

- Global mean pooling to aggregate node embeddings

The same encoder instance is shared across all drugs.


# DrugInteractionModel

The DrugInteractionModel predicts whether two drugs interact.

Inputs

- drugA: PyG Data or Batch for Drug A

- drugB: PyG Data or Batch for Drug B

Processing Steps

1. Encode Drug A using the shared DrugGATEncoder

2. Encode Drug B using the same encoder

3. Concatenate the two graph embeddings

4. Pass the combined representation through an MLP

Output

- A single interaction logit per drug pair, suitable for BCEWithLogitsLoss


# Training and Inference Helpers
train_one_epoch
train_one_epoch(model, loader, optimizer, device) -> float


Trains the model for one epoch.

## Expected batch attributes

batch.drugA: PyG batch of Drug A graphs

batch.drugB: PyG batch of Drug B graphs

batch.y: Binary labels (0 or 1)

Returns the average training loss for the epoch.

predict_proba
predict_proba(model, loader, device) -> (y_true, y_prob)


Runs inference on a dataset and returns:

- Ground-truth labels

-Predicted interaction probabilities

This function is used to compute ROC-AUC and PR-AUC metrics.


## End-to-End Workflow

The full prediction pipeline follows these steps:

Drug name
   ↓
SMILES string
   ↓
RDKit molecule parsing
   ↓
Molecular graph construction (PyG Data)
   ↓
Graph attention encoder (DrugGATEncoder)
   ↓
Graph-level drug embeddings
   ↓
Pairwise interaction model (DrugInteractionModel)
   ↓
Interaction logit
   ↓
Sigmoid → interaction probability


## Extensibility

The API is designed to support future extensions:

- Richer atom and bond features can be added inside smiles_to_pyg_graph

- The encoder can be swapped with other GNNs such as GIN or GraphSAGE

- The interaction head can be modified without changing data loaders

- Pretraining or regularization can be applied at the encoder level

## Summary

This API provides a clean, modular, and reproducible interface for drug–drug interaction prediction using graph neural networks. It supports molecular graph construction, model training, inference, and evaluation, while remaining lightweight and easy to integrate into experimental workflows.