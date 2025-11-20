# PyTorch Geometric – API Overview (MSML610 Project)

## 1. What is PyTorch Geometric?

PyTorch Geometric (PyG) is a library for deep learning on graph-structured data.  
This API document explains the core tools used in the project.

---

## 2. Core Concepts

### `Data`
A single graph with:
- `x`: node features  
- `edge_index`: connectivity  
- `edge_attr`: bond features  
- `y`: label (optional)

### `Batch`
Mini-batches of graphs.  
Used during training.

### GNN Layers
Planned:
- `GCNConv`
- `GATConv`

---

## 3. Minimal PyG Example (Toy Graph)

```python
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

edge_index = torch.tensor([[0, 1],
                           [1, 2]], dtype=torch.long)

x = torch.tensor([[1.0, 0.0],
                  [0.0, 1.0],
                  [1.0, 1.0]], dtype=torch.float)

data = Data(x=x, edge_index=edge_index)

conv = GCNConv(2, 4)
out = conv(data.x, data.edge_index)

print(out.shape)

4. Planned API for the Project

These function signatures will be implemented later:

load_raw_ddi_tables(...)

prepare_ddi_pairs(...)

build_molecular_graph(smiles)

DrugGNN

DDIInteractionModel

train_one_epoch(...)

evaluate_model(...)


---

# ✅ **3. Fall2025_PyTorch_Geometric_Drug_Drug_Interaction_Prediction.example.md**

**File:**  
`Fall2025_PyTorch_Geometric_Drug_Drug_Interaction_Prediction.example.md`

```markdown
# End-to-End Example: Drug–Drug Interaction Prediction

## 1. Overview

This example notebook (and this document) describe how to perform full DDI prediction
using PyTorch Geometric.

For Phase 1, this file contains only the structure.

---

## 2. Planned Steps

1. Load Kaggle DDI dataset  
2. Convert each drug’s SMILES → graph  
3. Convert drug pairs → model inputs  
4. Build a GNN: GCN or GAT  
5. Train a classifier to predict interaction  
6. Evaluate ROC–AUC  

---

## 3. Utility Files Used

- `utils_data_io.py` – dataset loading  
- `Fall2025_..._utils.py` – graph building + model code  
- `utils_post_processing.py` – metrics  

---

## 4. Phase 1 Status
- Placeholder only  
- Real implementation begins in Phase 2  
