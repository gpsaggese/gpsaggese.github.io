# Optuna: Hyperparameter Optimization Framework

## What is Optuna?

**Optuna** is a Bayesian hyperparameter optimization framework that intelligently searches the hyperparameter space to find the best parameters for your machine learning model.

Instead of manually trying different hyperparameters (grid search) or random guessing (random search), Optuna **learns from past trials** and suggests promising hyperparameters to test next.

---

## Why Use Optuna?

### Problem: Manual Hyperparameter Tuning is Inefficient

When training clustering models, you need to choose:
- **K-Means**: n_clusters, init method, n_init
- **Hierarchical**: n_clusters, linkage method
- **DBSCAN**: eps (epsilon), min_samples

**Without Optuna:**
- Grid search: Test every combination (slow, exhaustive)
- Random search: Random combinations (inefficient)
- Manual tuning: Guess and check (subjective)

### Solution: Optuna Uses Bayesian Optimization

**Bayesian Optimization** (TPE sampler):
1. Tries hyperparameters (exploratory phase)
2. Observes results
3. Learns which regions are promising
4. Samples more densely in promising regions
5. Converges to optimal hyperparameters

**Result:** Finds good parameters in 10-20% fewer trials than random search.

---

## How Optuna Works: The Core Concepts

### 1. **Study**
Container for the entire optimization process. Stores all trials and results.

```python
study = optuna.create_study(direction='maximize')
```

- `direction`: Maximize or minimize objective?
- Holds all trials and best trial information

### 2. **Trial**
Single model evaluation with specific hyperparameters.

Each trial:
- Receives hyperparameters from Optuna
- Trains a model
- Returns an objective value (score)
- Is stored in the study

```python
def objective(trial):
    param1 = trial.suggest_int('param1', 1, 10)  # Trial suggests param1
    model = MyModel(param1=param1)
    score = evaluate(model)
    return score  # Trial returns score
```

### 3. **Objective Function**
The function Optuna optimizes. It:
- Receives a `trial` object
- Suggests hyperparameters using `trial.suggest_*()`
- Trains and evaluates a model
- Returns a single numeric value

```python
def objective(trial):
    n_clusters = trial.suggest_int('n_clusters', 2, 8)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)
    silhouette = silhouette_score(X, labels)
    return silhouette  # Optuna tries to maximize this
```

### 4. **Sampler**
Algorithm that decides which hyperparameters to suggest next.

**TPE (Tree-structured Parzen Estimator) - Default:**
- Bayesian sampler
- Models the distribution of good vs. bad hyperparameters
- Samples from regions likely to have good parameters
- Most efficient for most problems

**Other samplers:**
- Grid sampler: Exhaustive grid search
- Random sampler: Random combinations
- CMA-ES sampler: Evolution strategy

### 5. **Pruner**
Early stopping strategy. Stops unpromising trials before they finish.

**Example:** If a trial has bad silhouette score after 50% of iterations, stop it.

```python
def objective(trial):
    for epoch in range(100):
        # Train one epoch
        score = evaluate_model()
        
        # Tell Optuna intermediate value
        trial.report(score, epoch)
        
        # Optuna decides: should we continue or prune?
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return final_score
```

**MedianPruner:** Stops trials worse than median of previous trials.

### 6. **Best Trial**
After optimization, the trial with the best objective value.

```python
study.best_trial          # Trial object
study.best_params         # Dict of best hyperparameters
study.best_value          # Best objective value achieved
```

---

## The Optuna Workflow: 4 Steps

### Step 1: Define Objective Function

```python
def objective(trial):
    # Suggest hyperparameters
    n_clusters = trial.suggest_int('n_clusters', 2, 10)
    init = trial.suggest_categorical('init', ['k-means++', 'random'])
    
    # Train model
    model = KMeans(n_clusters=n_clusters, init=init, random_state=42)
    labels = model.fit_predict(X)
    
    # Evaluate and return score
    score = silhouette_score(X, labels)
    return score
```

**Key:** The objective function tells Optuna what to optimize.

### Step 2: Create Study

```python
study = optuna.create_study(direction='maximize')
```

- `direction='maximize'`: Find hyperparameters that maximize objective
- `direction='minimize'`: Find hyperparameters that minimize objective

### Step 3: Optimize

```python
study.optimize(objective, n_trials=100)
```

Runs 100 trials:
1. Trial 1: Optuna suggests random hyperparameters
2. Trial 1: Objective function evaluates model, returns score
3. Trial 2: Optuna suggests hyperparameters (informed by trial 1)
4. Trial 2: Objective function evaluates model
5. ... repeat until n_trials

**Optuna learns:** Which hyperparameters regions produce good scores.

### Step 4: Extract Best Parameters

```python
best_params = study.best_params
print(f"Best silhouette: {study.best_value:.4f}")
print(f"Best parameters: {best_params}")

# Train final model with best parameters
final_model = KMeans(**best_params, random_state=42)
final_labels = final_model.fit_predict(X)
```

---

## Hyperparameter Suggestion Methods

Optuna provides methods to define your search space:

### Integer Parameters
```python
n_clusters = trial.suggest_int('n_clusters', 2, 10)
# Returns integer between 2 and 10
```

### Float Parameters
```python
learning_rate = trial.suggest_float('learning_rate', 0.001, 0.1)
# Returns float between 0.001 and 0.1

# Optional: Log scale (useful for wide ranges)
learning_rate = trial.suggest_float('learning_rate', 0.001, 0.1, log=True)
```

### Categorical Parameters
```python
linkage = trial.suggest_categorical('linkage', ['ward', 'complete', 'average'])
# Returns one of the three options
```

### Discrete Uniform (specific step)
```python
n_init = trial.suggest_int('n_init', 5, 50, step=5)
# Returns 5, 10, 15, 20, ... 50
```

---

## Optuna in Practice: K-Means Example

```python
import optuna
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 1. Define objective
def objective(trial):
    n_clusters = trial.suggest_int('n_clusters', 2, 8)
    init = trial.suggest_categorical('init', ['k-means++', 'random'])
    n_init = trial.suggest_int('n_init', 5, 20)
    
    kmeans = KMeans(n_clusters=n_clusters, init=init, n_init=n_init, random_state=42)
    labels = kmeans.fit_predict(X)
    
    silhouette = silhouette_score(X, labels)
    return silhouette

# 2. Create study
study = optuna.create_study(direction='maximize')

# 3. Optimize
study.optimize(objective, n_trials=100)

# 4. Extract best
print(f"Best silhouette: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")

# Train final model
best_kmeans = KMeans(**study.best_params, random_state=42)
final_labels = best_kmeans.fit_predict(X)
```

---

## Optuna with Pruning: DBSCAN Example

Some algorithms (like DBSCAN) can fail with bad hyperparameters. Use pruning to stop bad trials early.

```python
def objective_dbscan(trial):
    eps = trial.suggest_float('eps', 0.1, 2.0)
    min_samples = trial.suggest_int('min_samples', 2, 10)
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)
    
    # Check validity
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    # Prune if invalid
    if n_clusters < 2:
        raise optuna.TrialPruned()  # Stop this trial
    if n_noise / len(labels) > 0.5:
        raise optuna.TrialPruned()  # Stop this trial
    
    valid_mask = labels != -1
    silhouette = silhouette_score(X[valid_mask], labels[valid_mask])
    return silhouette

# Create study with pruner
study = optuna.create_study(
    direction='maximize',
    pruner=optuna.pruners.MedianPruner()
)

# Optimize (pruner stops bad trials automatically)
study.optimize(objective_dbscan, n_trials=100)
```

**Result:** DBSCAN often gets pruned because different eps values create completely different cluster structures. Pruning saves time by rejecting invalid configurations early.

---

## Understanding Optuna's Progress

As Optuna runs trials, it learns:

```
Trial 1:  silhouette = 0.35  (random guess)
Trial 2:  silhouette = 0.42  (random guess)
Trial 3:  silhouette = 0.38  (Optuna learns trials 1-2 aren't good)
Trial 4:  silhouette = 0.48  (Optuna samples promising region)
Trial 5:  silhouette = 0.51  (continues in promising region)
...
Trial 50: silhouette = 0.54  (converges near optimum)
Trial 51: silhouette = 0.543 (still exploring, small improvements)
```

**Key observation:** Best value improves quickly, then plateaus (convergence).

---

## Optuna vs. Other Methods

| Method | Speed | Quality | Effort |
|--------|-------|---------|--------|
| **Grid Search** | Slow | Depends on grid | Low (specify grid) |
| **Random Search** | Medium | Okay | Low (random) |
| **Optuna (Bayesian)** | Fast | Excellent | Low (define function) |
| **Manual Tuning** | Slow | Depends on human | High (many iterations) |

---

## Key Optuna Features

 **TPE Sampler:** Bayesian optimization learns from trials

 **Pruning:** Early stopping for efficiency

 **Parallelization:** Run multiple trials in parallel (advanced)

 **Visualization:** Built-in plotting of optimization history

 **Simple API:** Just define objective, create study, optimize

 **Framework Agnostic:** Works with any model (sklearn, keras, pytorch, xgboost, etc.)

---

## When to Use Optuna

 **Good for:**
- Tuning many hyperparameters
- Expensive models (reduces number of trials needed)
- Want reproducible, automated tuning
- Clustering/classification/regression models

 **Not needed for:**
- Single hyperparameter
- Pre-tuned defaults work well
- Quick experiments (just use random search)

---

## Summary: Why Optuna?

1. **Intelligent search** - Learns from past trials
2. **Efficient** - Finds good hyperparameters quickly
3. **Pruning** - Stops bad trials early
4. **Simple** - 4-step workflow (define → create → optimize → extract)
5. **Universal** - Works with any ML model
6. **Reproducible** - Same code, same results

**Next step:** Use Optuna-optimized hyperparameters to train your final model!

---

## References

- **Official Docs:** https://optuna.readthedocs.io/
- **Paper:** Akiba et al. (2019) "Optuna: A Next-generation Hyperparameter Optimization Framework"
- **TPE Sampler:** Bergstra et al. (2011) "Algorithms for Hyperparameter Optimization"
