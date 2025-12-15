# Drug–Drug Interaction Prediction Using Graph Neural Networks

## Project Overview

This project focuses on predicting potential drug–drug interactions (DDIs) using molecular structure information. Each drug is represented as a molecular graph, where atoms form the nodes and chemical bonds form the edges. A graph neural network (GNN) is then trained to learn meaningful molecular representations and predict whether a given pair of drugs is likely to interact.

The motivation behind this approach is to move beyond traditional handcrafted features and allow the model to learn interaction-relevant chemical patterns directly from molecular structure. In addition to predictive performance, the project also emphasizes interpretability and model transparency by analyzing atom- and bond-level contributions to predictions.

The final pipeline includes a strong baseline comparison, interpretability analysis, and a bonus self-supervised pretraining step.

---

## Objectives

The key objectives of this project are:

- To convert drug molecules into graph representations using atom- and bond-level features  
- To design an edge-aware graph neural network that learns molecular embeddings  
- To predict drug–drug interactions using learned representations of drug pairs  
- To evaluate model performance using ROC-AUC and PR-AUC metrics  
- To compare the GNN against a traditional fingerprint-based baseline  
- To interpret model predictions using atom- and bond-level explanations  
- To explore transfer learning through self-supervised pretraining on molecular graphs  

---

## Real-World Importance

Drug–drug interactions are a critical issue in healthcare, particularly for patients taking multiple medications simultaneously. Undetected interactions can reduce drug effectiveness or cause serious adverse effects.

Accurate DDI prediction systems can:
- Assist clinicians in prescribing safer drug combinations  
- Support pharmacovigilance and drug safety monitoring  
- Reduce experimental costs in early-stage drug development  

By learning directly from molecular structure, this project demonstrates how machine learning can complement traditional pharmacological analysis and improve drug safety.

---

## Dataset Description

The dataset used in this project is the Drug–Drug Interaction dataset from Kaggle. It consists of pairs of drugs along with binary labels indicating whether an interaction is known to occur.

Each record contains:
- Name of Drug A  
- Name of Drug B  
- Interaction label (1 = interaction, 0 = no known interaction)

SMILES strings for each drug are retrieved and cached to enable molecular graph construction.

---

## Data Features

### Atom-Level Features

For each atom in a molecule, the following features are extracted:

- Atomic number  
- Atom degree  
- Formal charge  
- Aromaticity indicator  
- Hybridization state  
- Total number of hydrogens  
- Ring membership  
- Valence  
- Number of radical electrons  
- Chirality indicator  

These features allow the model to capture local chemical environments.

### Bond-Level Features

Each bond is represented using:

- Bond type (single, double, etc.)  
- Aromaticity  
- Conjugation  
- Ring membership  

Including bond features enables the model to learn how connectivity and chemical relationships influence interactions.

---

## Patterns Observed in the Data

Several patterns were observed during exploration:

- The dataset is moderately imbalanced, with fewer interacting pairs than non-interacting pairs  
- Certain drugs appear in multiple interaction pairs, acting as hubs  
- Chemically similar drugs often show similar interaction behavior  
- Hard negative pairs (chemically similar but non-interacting) significantly increase task difficulty and realism  

---

## Project Workflow

The full workflow of the project is illustrated below:

+----------------------------------------------------+
| 1. DATA LOADING & SMILES RETRIEVAL |
| Load CSV → Normalize drug names → Cache SMILES |
+----------------------------------------------------+
↓
+----------------------------------------------------+
| 2. PAIR CONSTRUCTION |
| Positive pairs → Hard negative sampling |
+----------------------------------------------------+
↓
+----------------------------------------------------+
| 3. DATA SPLITTING |
| Train (64%) → Validation (16%) → Test (20%) |
+----------------------------------------------------+
↓
+----------------------------------------------------+
| 4. GRAPH CONSTRUCTION |
| Atom features → Bond features → PyG graphs |
+----------------------------------------------------+
↓
+----------------------------------------------------+
| 5. GRAPH CACHING |
| Precompute graphs → Save to disk |
+----------------------------------------------------+
↓
+----------------------------------------------------+
| 6. DATASET & DATALOADER |
| Custom Dataset → Batch drug pairs |
+----------------------------------------------------+
↓
+----------------------------------------------------+
| 7. GNN MODEL |
| Edge-aware encoder → Pairwise interaction head |
+----------------------------------------------------+
↓
+----------------------------------------------------+
| 8. TRAINING & VALIDATION |
| BCE loss → Early stopping → Model selection |
+----------------------------------------------------+
↓
+----------------------------------------------------+
| 9. FINAL EVALUATION |
| Test ROC-AUC / PR-AUC |
+----------------------------------------------------+
↓
+----------------------------------------------------+
| 10. BASELINE COMPARISON |
| Morgan fingerprints + Logistic Regression |
+----------------------------------------------------+
↓
+----------------------------------------------------+
| 11. INTERPRETABILITY |
| GNNExplainer → Atom & bond importance |
+----------------------------------------------------+
↓
+----------------------------------------------------+
| 12. BONUS: PRETRAINING |
| Self-supervised contrastive learning |
+----------------------------------------------------+


---

## Step-by-Step Implementation

### Step 1  
**Data Loading, Environment Setup, and SMILES Cache Initialization**

The dataset is loaded and preprocessed. SMILES strings are retrieved and cached locally to avoid repeated external lookups and to ensure reproducibility.

---

### Step 2  
**Construction of Positive Drug Pairs and Hard Negative Sampling**

Known interacting drug pairs are treated as positive examples. Hard negative samples are generated by pairing chemically similar drugs that do not have known interactions, making the classification task more realistic.

---

### Step 3  
**Train–Validation–Test Split**

The dataset is split using stratified sampling to preserve class balance across the training, validation, and test sets.

---

### Step 4  
**Atom and Bond Feature Engineering and Graph Construction**

Each drug molecule is converted into a graph with rich atom- and bond-level features using RDKit and PyTorch Geometric.

---

### Step 5  
**Graph Caching**

All molecular graphs are precomputed and cached to significantly reduce training time.

---

### Step 6  
**Custom Dataset and DataLoader**

A custom PyTorch dataset returns paired molecular graphs along with interaction labels, enabling efficient mini-batch training.

---

### Step 7  
**Edge-Aware GNN and Interaction Model**

An edge-aware GNN encoder learns molecular embeddings. These embeddings are combined using a pairwise interaction head that explicitly models relationships between two drugs.

---

### Step 8  
**Model Training and Validation**

The model is trained using a weighted binary cross-entropy loss to address class imbalance. Early stopping is applied based on validation ROC-AUC.

---

### Step 9  
**Final Test Evaluation**

The best-performing model is evaluated on the held-out test set using ROC-AUC and PR-AUC metrics.

---

### Step 10 (Baseline)  
**Morgan Fingerprints with Logistic Regression**

A traditional baseline using Morgan fingerprints and logistic regression is implemented to provide a strong point of comparison.

---

### Step 11  
**Interpretability Using GNNExplainer**

GNNExplainer is used to identify atoms and bonds that contribute most strongly to predicted interactions. Bond importance is derived from atom importance to provide chemically meaningful explanations.

---

### Step 12  
**Self-Supervised Pretraining (Bonus)**

A contrastive learning objective is applied to pretrain the molecular encoder on unlabeled molecular graphs. The pretrained encoder is then fine-tuned on the DDI task.

---

## Results and Detailed Analysis

The final test performance is summarized below:

- **Graph Neural Network ROC-AUC:** 0.8643  
- **Baseline ROC-AUC:** 0.8631  

The GNN slightly outperforms the fingerprint-based baseline. While the improvement is modest, it is meaningful given the strength of Morgan fingerprints. Importantly, the GNN achieves this performance while also providing interpretability and flexibility that fingerprint-based models lack.

The improvement can be attributed to:
- Explicit modeling of bond-level information  
- Rich atom-level features  
- Interaction-aware pairwise representation learning  
- Self-supervised pretraining of the molecular encoder  

---

## Key Learnings

- Bond features play a crucial role in molecular interaction prediction  
- Pairwise interaction modeling significantly improves performance  
- Self-supervised pretraining helps close the gap with strong baselines  
- Interpretability is a major advantage of graph-based models  

---

## Challenges Faced

Initially, the model was trained using fewer atom features and no pretraining, resulting in poor performance. Incremental improvements, including richer features, better interaction modeling, and pretraining, led to substantial gains. Debugging interpretability tools required careful handling of graph masking behavior.

---

## Future Work

- Incorporate large-scale pretrained molecular encoders  
- Integrate pharmacokinetic and clinical metadata  
- Extend the model to multi-task drug safety prediction  

---

## Conclusion

This project demonstrates a complete and interpretable approach to drug–drug interaction prediction using graph neural networks. By combining molecular graph representations, edge-aware modeling, self-supervised pretraining, and interpretability analysis, the proposed method achieves competitive performance while offering valuable insights into the chemical basis of predicted interactions.
