# TutorTask23 – Fall 2025 PyTorch Geometric Drug–Drug Interaction Prediction

## 1. Overview

This project is part of the MSML610 class project.  
The goal is to build a tutorial-style example demonstrating how to use **PyTorch Geometric** (PyG) for **Drug–Drug Interaction (DDI)** prediction.

This folder contains:
- API documentation and a corresponding API notebook  
- A complete worked example (markdown + notebook)  
- Utility Python modules  
- A Dockerfile defining a reproducible environment  

This README explains the motivation, goals, and structure for Phase 1.

---

## 2. Tool Used: PyTorch Geometric

PyTorch Geometric is a graph deep learning library that extends PyTorch.  
Key features:

- Graph Convolution Networks (GCN)
- Graph Attention Networks (GAT)
- Graph data structures (`Data`, `Batch`)
- Efficient message passing

In this project:
- Each **drug** will be modeled as a molecular graph  
- A GNN will embed each drug  
- Embeddings of drug pairs will be combined to predict interactions

---

## 3. Project Objective

**Goal:** Predict whether a drug pair exhibits an interaction based on molecular structure.

This requires:

1. Loading a Kaggle Drug–Drug Interaction dataset  
2. Converting SMILES → graphs  
3. Building a GNN encoder using PyTorch Geometric  
4. Training a classifier over drug pairs  
5. Evaluating performance (ROC–AUC etc.)

Phase 1 only sets up folder structure + file stubs.

---

## 4. Project Structure

UmdTask23_Fall2025_PyTorch_Geometric_Drug_Drug_Interaction_Prediction/
│
├── README.md
├── Dockerfile
│
├── Fall2025_PyTorch_Geometric_Drug_Drug_Interaction_Prediction.API.md
├── Fall2025_PyTorch_Geometric_Drug_Drug_Interaction_Prediction.API.ipynb
│
├── Fall2025_PyTorch_Geometric_Drug_Drug_Interaction_Prediction.example.md
├── Fall2025_PyTorch_Geometric_Drug_Drug_Interaction_Prediction.example.ipynb
│
├── Fall2025_PyTorch_Geometric_Drug_Drug_Interaction_Prediction_utils.py
├── utils_data_io.py
└── utils_post_processing.py

markdown
Copy code

---

## 5. Phase 1 Status

- ✅ All required files created  
- ✅ API & Example markdown placeholders  
- ✅ API & Example notebooks placeholders  
- ✅ Utility modules created (empty structure)  
- ⬜ Dataset loading  
- ⬜ Graph construction  
- ⬜ GNN model  
- ⬜ Training loop  
- ⬜ Evaluation  

This completes **Phase 1 (Initial Project Setup)**.