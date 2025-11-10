# API Tutorial: Model Context Protocol (MCP)

This document explains the *tool*, MCP, not the real estate project. The goal is to teach a classmate how to use `mcp` for their own project.

## 1. What is MCP?

The Model Context Protocol (MCP) is a library for managing the machine learning lifecycle. It allows you to "wrap" your training runs in a **Context**, making it easy to log:

* **Parameters:** The settings you used (e.g., `learning_rate`).
* **Artifacts:** The files you produced (e.g., the trained `model.json`).
* **Metrics:** The results you got (e.g., `test_mse: 0.15`).

## 2. Core Usage: `mcp.Context`

You use MCP with a Python `with` statement.

```python
import mcp

params = {'learning_rate': 0.1, 'n_estimators': 100}

with mcp.Context(name="my-first-run", params=params) as ctx:
    # Your model training code goes here
    model = train_my_model(params)
    
    # Log results
    metrics = {'accuracy': 0.95}
    ctx.log_metrics(metrics)
    
    # Log outputs
    ctx.log_artifact("model.pkl", model)