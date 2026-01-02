# BoTorch: Multi-Objective Bayesian Optimization for Drug Discovery

## Project Overview

BoTorch Optimization  is a comprehensive implementation of multi-objective Bayesian optimization using BoTorch for drug discovery applications. The project demonstrates how to efficiently select chemical compounds by simultaneously optimizing conflicting objectives: **maximizing biological potency** while **minimizing synthetic complexity and cost**.


**Course**: MSML610 - Advanced Machine Learning



## Problem Statement

### The Drug Discovery Challenge

In early-stage drug discovery, medicinal chemists face a fundamental trade-off:

**High Potency** (strong biological activity) ↔ **High Complexity** (expensive, hard to synthesize)

Traditional approaches optimize these objectives separately, leading to suboptimal choices:
- **Potency-only selection**: Identifies highly active compounds that may be synthetically intractable
- **Cost-only selection**: Yields cheap, easily-made compounds with insufficient activity
- **Sequential optimization**: First potency, then filter by cost → misses optimal trade-offs

### Our Solution

**Multi-Objective Bayesian Optimization** using BoTorch to:
1. **Simultaneously optimize** both objectives
2. **Identify the Pareto front** - the set of optimal trade-off solutions
3. **Quantify uncertainty** in predictions via Gaussian Processes
4. **Enable informed decision-making** by presenting multiple viable options

---

## Methodology

### Pipeline Overview

```
┌─────────────────────────────────────────────────────────────┐
│  1. DATA LOADING & CLEANING                                 │
│     • ChEMBL AChE bioactivity data (15,542 compounds)       │
│     • Filter IC50 measurements in nM units                  │
│     • Remove missing/invalid entries                        │
│     • Deduplicate by SMILES → 4,624 unique compounds        │
└─────────────────┬───────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────┐
│  2. FEATURE ENGINEERING                                     │
│     • Calculate 13 molecular descriptors (RDKit)            │
│     • Compute synthetic accessibility (SA) score            │
│     • Define cost metric: 0.5*MW + 0.5*SA                   │
└─────────────────┬───────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────┐
│  3. TRAIN-TEST SPLIT                                        │
│     • 70% training (3,237 compounds)                        │
│     • 30% test/candidates (1,387 compounds)                 │
│     • Stratified by potency distribution                    │
└─────────────────┬───────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────┐
│  4. GAUSSIAN PROCESS MODELING                               │
│     • GP₁: Predicts potency (pIC50)                         │
│     • GP₂: Predicts cost metric                             │
│     • Hyperparameter optimization via MLL                   │
│     • Standardized features (StandardScaler)                │
└─────────────────┬───────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────┐
│  5. MULTI-OBJECTIVE OPTIMIZATION                            │
│     • Predict both objectives for all candidates            │
│     • Apply is_non_dominated() from BoTorch                 │
│     • Identify Pareto front (~5-10% of candidates)          │
└─────────────────┬───────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────┐
│  6. STRATEGY COMPARISON                                     │
│     • Random selection                                      │
│     • Top potency only                                      │
│     • Lowest cost only                                      │
│     • Balanced ratio (potency/cost)                         │
│     • Pareto-optimal (BoTorch)                              │
└─────────────────┬───────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────┐
│  7. VISUALIZATION & EXPORT                                  │
│     • Pareto front scatter plot                             │
│     • Strategy comparison bar charts                        │
│     • Molecular structure grids                             │
│     • CSV export for experimental testing                   │
└─────────────────────────────────────────────────────────────┘
```

---

## Dataset

### Source
**ChEMBL Database** - Manually curated bioactivity data from scientific literature

**Target Protein**: Acetylcholinesterase (AChE)
- Enzyme involved in neurotransmitter breakdown
- Drug target for Alzheimer's disease, myasthenia gravis
- Well-studied with extensive bioactivity data

### Dataset Statistics

| Metric | Value |
|--------|-------|
| Raw records | 15,542 |
| After IC50 filter | 11,234 |
| After quality filters | 6,892 |
| Unique compounds (final) | 4,624 |
| Retention rate | 29.8% |

### Data Quality Criteria
- **Measurement type**: IC50 only (half-maximal inhibitory concentration)
- **Units**: Nanomolar (nM) only
- **Relation**: Exact measurements (=) only, no inequalities (<, >, ~)
- **Validity**: No flagged data quality issues
- **SMILES**: Valid, parseable canonical SMILES strings
- **Deduplication**: Median aggregation for compounds with multiple measurements

### pIC50 Calculation
```
pIC50 = -log₁₀(IC50 × 10⁻⁹)
```
Where IC50 is in nM units.

**Interpretation**:
- pIC50 = 5: IC50 = 10 μM (weak)
- pIC50 = 7: IC50 = 100 nM (moderate)
- pIC50 = 9: IC50 = 1 nM (strong)
- pIC50 = 11: IC50 = 10 pM (very strong)

**Dataset Distribution**:
- Mean pIC50: 6.18 (≈660 nM)
- Range: 4.00 - 10.96 (10 μM to 0.11 nM)
- Standard deviation: 1.85

---

## Implementation Details

### Technologies Used

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| **Optimization** | BoTorch | 0.16.1 | Bayesian optimization framework |
| **GP Backend** | GPyTorch | 1.14.3 | Gaussian Process library |
| **Deep Learning** | PyTorch | 2.9.0 | Tensor operations, GPU support |
| **Chemistry** | RDKit | 2025.9.3 | Molecular descriptors, SMILES parsing |
| **ML Utils** | scikit-learn | 1.6.1 | Preprocessing, metrics |
| **Data** | pandas | 2.x | Data manipulation |
| **Compute** | NumPy | 2.0.2 | Numerical operations |
| **Visualization** | Matplotlib, Seaborn | 3.x, 0.12 | Plotting |
| **Notebook** | Jupyter | 1.0.0 | Interactive environment |

### Molecular Descriptors Calculated

| Descriptor | Description | Range | Unit |
|------------|-------------|-------|------|
| `mol_weight` | Molecular weight | 46-1200 | Da |
| `logp` | Lipophilicity (partition coefficient) | -5 to +10 | - |
| `hbd` | Hydrogen bond donors | 0-20 | count |
| `hba` | Hydrogen bond acceptors | 0-30 | count |
| `rotatable_bonds` | Rotatable bonds | 0-40 | count |
| `tpsa` | Topological polar surface area | 0-300 | Ų |
| `rings` | Ring count | 0-10 | count |
| `aromatic_rings` | Aromatic rings | 0-8 | count |
| `heavy_atoms` | Non-hydrogen atoms | 5-100 | count |
| `num_heteroatoms` | Heteroatoms (N, O, S, P, etc.) | 0-30 | count |

### Cost Metric Design

The cost metric combines two components:

```python
cost = 0.5 × (MW / 1000) + 0.5 × (SA_score / 10)
```

**Components**:
1. **Molecular Weight (MW)**: Normalized by 1000 Da
   - Proxy for synthesis complexity (larger molecules = more steps)
   - Typical drug MW: 300-500 Da

2. **Synthetic Accessibility (SA) Score**: Heuristic estimate (1=easy, 10=hard)
   - Based on molecular complexity factors:
     - Number and size of rings
     - Chiral centers
     - Functional group diversity
     - Steric hindrance

**Rationale**:
- Equal weights (0.5, 0.5) treat both factors as equally important
- Easily adjustable for different organizational priorities
- Range: ~0.08 to 1.12 (normalized scale)


### Complete Workflow

See `Botoch.example.ipynb` for the full pipeline:

1. **Data Loading** (Cells 1-3)
   - Load ChEMBL data
   - Apply quality filters
   - Compute pIC50

2. **Feature Engineering** (Cells 4-6)
   - Calculate molecular descriptors
   - Compute cost metric
   - Visualize distributions

3. **Model Training** (Cells 7-9)
   - Train-test split
   - Build GP models for both objectives
   - Validate on held-out data

4. **Optimization** (Cells 10-12)
   - Predict objectives for all candidates
   - Identify Pareto front
   - Rank solutions

5. **Strategy Comparison** (Cells 13-15)
   - Compare against baselines
   - Compute performance metrics
   - Visualize trade-offs

6. **Results Export** (Cells 16-18)
   - Export top compounds to CSV
   - Generate visualizations
   - Create molecular structure images

---

## Results

### Model Performance

#### Gaussian Process Validation

| Model | R² Score | RMSE | MAE | Interpretation |
|-------|----------|------|-----|----------------|
| **Potency (pIC50)** | 0.88 | 0.47 | 0.35 | Excellent |
| **Cost Metric** | 0.99 | 0.019 | 0.014 | Near-perfect |

**Analysis**:
- Potency model achieves R² = 0.88, explaining 88% of variance
- Average prediction error: ±0.47 pIC50 units (≈3× IC50 factor)
- Cost model is highly accurate (R² = 0.99) because it's derived from predictable properties

---

### Multi-Objective Optimization Results

#### Pareto Front Statistics

| Metric | Value |
|--------|-------|
| Total candidates | 1,387 |
| Pareto-optimal | 57 |
| Pareto percentage | 4.1% |
| Average potency (Pareto) | 7.88 |
| Average cost (Pareto) | 0.42 |
| Average potency (all) | 6.18 |
| Average cost (all) | 0.45 |

**Key Finding**: Pareto compounds have **27% higher potency** (+1.7 pIC50 units) with **7% lower cost** compared to the overall population.

---

### Strategy Comparison

| Strategy | Avg Potency | Avg Cost | Max Potency | Trade-off Score | Rank |
|----------|-------------|----------|-------------|-----------------|------|
| **Balanced Ratio** | 7.73 | 0.20 | 10.98 | 39.46 | 🥇 1st |
| **Pareto-Optimal (BoTorch)** | 7.88 | 0.23 | 10.87 | 33.61 | 🥈 2nd |
| **Lowest Cost** | 6.19 | 0.14 | 6.98 | 36.16 | 🥉 3rd |
| **Top Potency** | 10.45 | 0.49 | 10.98 | 20.46 | 4th |
| **Random** | 6.18 | 0.45 | 9.70 | 13.80 | 5th |

**Trade-off Score** = Average Potency / Average Cost (higher is better)

**Insights**:
1. **Balanced Ratio** wins overall by maximizing potency/cost ratio explicitly
2. **Pareto-Optimal** achieves highest average potency (7.88) while maintaining low cost
3. **BoTorch outperforms random by 2.4×** (143% improvement)
4. **Top Potency** has highest maximum (10.98) but at 2× the cost
5. **Lowest Cost** is cheap (0.14) but weak potency (6.19)

---

### Top Selected Compounds

#### Top 5 Pareto-Optimal Compounds (by Potency)

| Rank | SMILES (abbreviated) | pIC50 | Cost | MW (Da) | SA Score |
|------|---------------------|-------|------|---------|----------|
| 1 | COc1ccc(CC(=O)Nc2cc...)  | 10.59 | 0.39 | 450.5 | 3.9 |
| 2 | O=C(Nc1ccc(O)cc1)c2... | 10.78 | 0.37 | 382.4 | 3.2 |
| 3 | CCN(CC)C(=O)c1cc(N)... | 10.87 | 0.41 | 468.6 | 4.1 |
| 4 | COc1cccc(NC(=O)c2n... | 9.74 | 0.30 | 354.4 | 2.8 |
| 5 | C1CCNCC1 | 6.76 | 0.08 | 99.2 | 1.2 |

**Interpretation**:
- Compounds 1-3: High potency (pIC50 > 10), moderate cost
- Compound 4: Balanced trade-off
- Compound 5: Very simple, cheap, but weaker



## Key Features

### 1. Multi-Objective Optimization
- **Simultaneous optimization** of conflicting objectives
- **Pareto front identification** via BoTorch's `is_non_dominated()`
- **Trade-off visualization** with interactive plots

### 2. Gaussian Process Surrogates
- **Probabilistic predictions** with uncertainty quantification
- **Hyperparameter optimization** via marginal likelihood
- **Flexible kernel choice** (Matérn, RBF, Rational Quadratic)

### 3. Comprehensive Data Processing
- **Automated data cleaning** with quality filters
- **13 molecular descriptors** computed via RDKit
- **Synthetic accessibility scoring** (heuristic-based)
- **SMILES validation** and error handling

### 4. Strategy Comparison Framework
- **5 baseline strategies** for benchmarking
- **Multiple evaluation metrics** (avg potency, cost, trade-off)
- **Statistical comparison** with confidence intervals

### 5. Production-Ready Code
- **Modular design** with reusable utilities
- **Comprehensive error handling**
- **Extensive documentation** (docstrings, type hints)
- **Reproducible results** (fixed random seeds)

### 6. Rich Visualizations
- **Pareto front scatter plots** with strategy overlays
- **Strategy comparison bar charts** (4 metrics)
- **Molecular structure grids** (RDKit rendering)
- **Distribution histograms** for all descriptors

---

## Performance Metrics

### Computational Performance

| Metric | Value | Hardware |
|--------|-------|----------|
| Data loading | 2.3s | Standard laptop |
| Descriptor calculation | 18.5s | 4,624 compounds |
| GP training (both models) | 12.7s | CPU (8 cores) |
| Pareto identification | 0.4s | 1,387 candidates |


    
### Software & Tools

- **BoTorch**: https://botorch.org/
- **GPyTorch**: https://gpytorch.ai/
- **RDKit**: https://www.rdkit.org/
- **ChEMBL_Dataset**: https://www.kaggle.com/datasets/gauravan/human-acetylcholinesterase-dataset-from-chembl 