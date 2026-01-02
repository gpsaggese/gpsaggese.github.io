import argparse
import json
import os
from pathlib import Path
from typing import Dict, Tuple

import dgl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from src.models.gnn_fraud import TX_NODE_TYPE, build_model
from src.models.tabular_baselines import BASIC_FEATS
from src.utils.splits import indices_to_mask, temporal_group_split_indices
from torch_geometric.data import HeteroData




# === Paths / column names (tweak here if needed) ===

DATA_PROCESSED_DIR = Path("data/processed")
ARTIFACTS_DIR = Path("data/artifacts")

FEATURES_PATH = DATA_PROCESSED_DIR / "features.parquet"
GRAPH_PATH = ARTIFACTS_DIR / "hetero_graph.pt"
GNN_METRICS_PATH = ARTIFACTS_DIR / "gnn_metrics.json"
GNN_MODEL_PATH = ARTIFACTS_DIR / "gnn_model.pt"
GNN_VAL_TEST_PREDS_PATH = ARTIFACTS_DIR / "gnn_val_test_preds.parquet"

# TODO: adjust label column if different
LABEL_COLUMN_CANDIDATES = ["isFraud", "is_fraud", "label"]

# If you have a transaction ID column in features that *must* align with graph node IDs,
# set it here (otherwise we'll assume row order matches transaction node IDs).
TX_ID_COLUMN_CANDIDATES = ["TransactionID", "transaction_id"]
GNN_FEATURES = BASIC_FEATS


# === Helper functions ===


def find_first_existing_column(df: pd.DataFrame, candidates) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"None of the candidate columns exist in features: {candidates}")


def load_features(selected_features: Tuple[str, ...] | None = None) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    """
    Returns
    -------
    df : pd.DataFrame
        Full feature dataframe including metadata columns (TransactionDT, card1, ...)
    X : np.ndarray [num_nodes, num_features]
    y : np.ndarray [num_nodes]
    feature_cols : list[str]
        Names of the columns used in X (for reporting / reproducibility).
    """
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Expected features at {FEATURES_PATH}, but file does not exist.")

    df = pd.read_parquet(FEATURES_PATH)

    label_col = find_first_existing_column(df, LABEL_COLUMN_CANDIDATES)
    y = df[label_col].to_numpy(dtype=np.float32)

    drop_cols = [label_col]
    for cand in TX_ID_COLUMN_CANDIDATES:
        if cand in df.columns:
            drop_cols.append(cand)

    if selected_features:
        missing = [c for c in selected_features if c not in df.columns]
        if missing:
            raise ValueError(f"Selected features missing in dataframe: {missing}")
        feature_cols = list(selected_features)
    else:
        numeric_cols = df.drop(columns=drop_cols, errors="ignore").select_dtypes(include=["number"]).columns
        feature_cols = list(numeric_cols)

    feature_df = df[feature_cols].copy()
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
    num_missing = int(feature_df.isna().sum().sum())
    if num_missing > 0:
        print(f"[load_features] Filling {num_missing} missing feature values with 0.0")
    feature_df = feature_df.fillna(0.0)

    X = feature_df.to_numpy(dtype=np.float32)
    return df, X, y, feature_cols


def scale_features(X: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, StandardScaler]:
    """
    Fit a StandardScaler on the training subset (to avoid leakage) and transform all splits.
    """
    scaler = StandardScaler()
    scaler.fit(X[train_idx])
    X_scaled = scaler.transform(X)
    return X_scaled.astype(np.float32), scaler


def pick_best_threshold(y_true: np.ndarray, probs: np.ndarray, default: float = 0.5) -> tuple[float, float]:
    """
    Pick the threshold that maximizes F1 on the precision-recall curve.
    Returns (threshold, f1_at_threshold).
    """
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    if len(thresholds) == 0:
        return default, 0.0
    f1_scores = (2 * precision * recall) / np.clip(precision + recall, 1e-12, None)
    best_idx = np.nanargmax(f1_scores)
    best_thresh = float(thresholds[min(best_idx, len(thresholds) - 1)])
    return best_thresh, float(f1_scores[best_idx])


def simple_time_split_masks(df: pd.DataFrame, train_frac: float = 0.6, val_frac: float = 0.2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fallback split for tiny samples: simple chronological 60/20/20 masks.
    """
    df_ord = df.sort_values("TransactionDT").reset_index(drop=True)
    n = len(df_ord)
    if n < 10:
        raise ValueError("Not enough rows to build even a tiny split; increase sample or max_rows.")
    tr_end = int(n * train_frac)
    va_end = int(n * (train_frac + val_frac))
    idx = np.arange(n)
    return idx[:tr_end], idx[tr_end:va_end], idx[va_end:]


def heterodata_to_dgl(obj: HeteroData) -> dgl.DGLHeteroGraph:
    """Convert a torch_geometric HeteroData graph to a DGLHeteroGraph using only structure.

    We ignore all node/edge attributes and only keep:
      - edge_index for each edge type
      - num_nodes per node type (if present, otherwise inferred from edges)
    """
    data_dict = {}
    num_nodes_dict = {}

    # 1) Build edge index dict for DGL
    for edge_type in obj.edge_types:
        # edge_type is a triple: (src_type, rel_type, dst_type)
        src_type, rel_type, dst_type = edge_type
        store = obj[edge_type]

        if "edge_index" not in store:
            raise ValueError(f"No edge_index found for edge type {edge_type}")

        edge_index = store["edge_index"]  # shape [2, E]
        src = edge_index[0].long()
        dst = edge_index[1].long()

        data_dict[(src_type, rel_type, dst_type)] = (src, dst)

    # 2) Infer num_nodes for each node type
    for node_type in obj.node_types:
        store = obj[node_type]
        n = getattr(store, "num_nodes", None)

        if n is None:
            # Infer from all incident edges
            max_idx = -1
            for edge_type in obj.edge_types:
                s_type, _, d_type = edge_type
                edge_index = obj[edge_type]["edge_index"]
                if node_type == s_type:
                    max_idx = max(max_idx, int(edge_index[0].max()))
                if node_type == d_type:
                    max_idx = max(max_idx, int(edge_index[1].max()))
            n = max_idx + 1

        num_nodes_dict[node_type] = int(n)

    # 3) Build DGL heterograph
    g = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)
    return g


def load_hetero_graph() -> dgl.DGLHeteroGraph:
    """Load the heterogeneous graph for GNN training.

    Supports both:
    - DGLHeteroGraph saved directly (preferred)
    - torch_geometric.data.HeteroData saved in hetero_graph.pt (converted to DGL)
    """
    if not GRAPH_PATH.exists():
        raise FileNotFoundError(
            f"Hetero graph file not found at {GRAPH_PATH}. "
            "Run the graph-building script first."
        )

    obj = torch.load(GRAPH_PATH, map_location="cpu")

    # Case 1: already a DGL graph (ideal)
    if isinstance(obj, dgl.DGLHeteroGraph):
        return obj

    # Case 2: saved as torch_geometric HeteroData -> convert to DGL (structure only)
    if isinstance(obj, HeteroData):
        g = heterodata_to_dgl(obj)

        if not isinstance(g, dgl.DGLHeteroGraph):
            raise TypeError(
                f"Conversion from HeteroData to DGL returned unexpected type: {type(g)}"
            )
        return g



    # Anything else is unexpected
    raise TypeError(
        "Could not interpret object loaded from "
        f"{GRAPH_PATH} as a DGLHeteroGraph. Got type: {type(obj)}"
    )



def compute_class_pos_weight(y: torch.Tensor) -> float:
    """
    For BCEWithLogitsLoss(pos_weight=...), we want (N_neg / N_pos).
    """
    pos = y.sum().item()
    neg = y.numel() - pos
    if pos == 0:
        return 1.0
    return float(neg / pos)


def evaluate_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute ROC-AUC and PR-AUC on a masked subset.

    Robust to:
    - empty masks
    - NaN / inf probabilities
    - subsets where all labels are the same class
    """
    metric_template = {
        "roc_auc": float("nan"),
        "pr_auc": float("nan"),
        "precision": float("nan"),
        "recall": float("nan"),
        "f1": float("nan"),
    }

    if mask.sum().item() == 0:
        return metric_template.copy()

    probs = torch.sigmoid(logits[mask]).detach().cpu().numpy().astype(np.float64)
    y_true = labels[mask].detach().cpu().numpy().astype(np.float64)

    # Filter out NaN / inf in probs
    finite_mask = np.isfinite(probs)
    if finite_mask.sum() == 0:
        return metric_template.copy()

    probs = probs[finite_mask]
    y_true = y_true[finite_mask]

    # If only one class present, sklearn will throw; handle gracefully
    if np.unique(y_true).size < 2:
        return metric_template.copy()

    try:
        roc = roc_auc_score(y_true, probs)
        pr = average_precision_score(y_true, probs)
    except ValueError:
        return metric_template.copy()

    yhat = (probs >= threshold).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, yhat, average="binary", zero_division=0
    )

    return {
        "roc_auc": float(roc),
        "pr_auc": float(pr),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
    }



# === Main training loop ===


def train_gnn(
    num_epochs: int = 10,
    lr: float = 1e-3,
    hidden_dim: int = 128,
    num_layers: int = 2,
    dropout: float = 0.2,
    pos_weight_scale: float = 1.2,
    account_key: str = "card1",
    val_days: int = 3,
    test_days: int = 3,
    threshold: float = 0.5,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, Dict[str, float]]:
    """
    Train a heterograph GNN for fraud detection.

    Returns
    -------
    metrics : dict
        Nested dict with 'train', 'val', 'test' metrics.
    """
    # Load data
    df, X_raw, y, feature_cols = load_features(tuple(GNN_FEATURES))
    label_col = find_first_existing_column(df, LABEL_COLUMN_CANDIDATES)
    graph = load_hetero_graph()

    # Basic sanity check: transaction node count vs feature rows
    num_tx_nodes = graph.num_nodes(TX_NODE_TYPE)
    if num_tx_nodes != X_raw.shape[0]:
        raise ValueError(
            f"Number of transaction nodes in graph ({num_tx_nodes}) "
            f"!= number of rows in features ({X_raw.shape[0]}). "
            f"Check your construct_hetero_graph.py and feature building pipeline."
        )

    # Train/val/test splits
    try:
        splits = temporal_group_split_indices(df, account_key, val_days, test_days)
        train_indices = splits.train_idx
        val_indices = splits.val_idx
        test_indices = splits.test_idx
        train_mask_np = indices_to_mask(len(df), train_indices)
        val_mask_np = indices_to_mask(len(df), val_indices)
        test_mask_np = indices_to_mask(len(df), test_indices)
        split_sizes = {
            "train": int(len(train_indices)),
            "val": int(len(val_indices)),
            "test": int(len(test_indices)),
        }
        split_strategy = "temporal_group"
    except ValueError:
        tr_idx, va_idx, te_idx = simple_time_split_masks(df)
        train_indices, val_indices, test_indices = tr_idx, va_idx, te_idx
        train_mask_np = indices_to_mask(len(df), tr_idx)
        val_mask_np = indices_to_mask(len(df), va_idx)
        test_mask_np = indices_to_mask(len(df), te_idx)
        split_sizes = {
            "train": int(len(tr_idx)),
            "val": int(len(va_idx)),
            "test": int(len(te_idx)),
        }
        split_strategy = "time_fallback"
        print("[train_gnn] Using fallback 60/20/20 time split for small sample.")

    # Scale features using train split only to avoid temporal leakage
    X_scaled, scaler = scale_features(X_raw, train_indices)

    # Numpy -> Torch
    X_t = torch.from_numpy(X_scaled)
    y_t = torch.from_numpy((y > 0.5).astype(np.float32))

    device = torch.device(device)
    X_t = X_t.to(device)
    y_t = y_t.to(device)
    graph = graph.to(device)
    train_mask = torch.from_numpy(train_mask_np).to(device)
    val_mask = torch.from_numpy(val_mask_np).to(device)
    test_mask = torch.from_numpy(test_mask_np).to(device)

    # Build model
    input_dim = X_t.shape[1]
    cfg_overrides = {
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "dropout": dropout,
    }
    model = build_model(graph, input_dim=input_dim, cfg_overrides=cfg_overrides)
    model.to(device)

    # Boost positive class weight slightly to favor recall / PR-AUC on skewed data
    pos_weight = compute_class_pos_weight(y_t[train_mask]) * pos_weight_scale
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    metrics_history: Dict[str, Dict[str, float]] = {}
    best_state = None
    best_val_pr = -np.inf
    best_epoch = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(graph, X_t)  # [num_nodes]
        loss = criterion(logits[train_mask], y_t[train_mask])
        loss.backward()
        optimizer.step()

        # Evaluation
        model.eval()
        with torch.no_grad():
            logits_eval = model(graph, X_t)
            train_metrics = evaluate_logits(logits_eval, y_t, train_mask, threshold)
            val_metrics = evaluate_logits(logits_eval, y_t, val_mask, threshold)
            test_metrics = evaluate_logits(logits_eval, y_t, test_mask, threshold)

        metrics_history[epoch] = {
            "loss": float(loss.item()),
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
        }

        if val_metrics["pr_auc"] > best_val_pr:
            best_val_pr = val_metrics["pr_auc"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"[Epoch {epoch:03d}] "
            f"loss={loss.item():.4f} "
            f"val_PR={val_metrics['pr_auc']:.4f} "
            f"val_F1={val_metrics['f1']:.4f} "
            f"test_PR={test_metrics['pr_auc']:.4f} "
            f"test_F1={test_metrics['f1']:.4f}"
        )

    # Restore best epoch weights for final reporting
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        logits_final = model(graph, X_t)
    probs_final = torch.sigmoid(logits_final).cpu().numpy().astype(np.float64)

    best_threshold, best_val_f1 = pick_best_threshold(y[val_mask_np], probs_final[val_mask_np], default=threshold)

    train_metrics = evaluate_logits(logits_final, y_t, train_mask, best_threshold)
    val_metrics = evaluate_logits(logits_final, y_t, val_mask, best_threshold)
    test_metrics = evaluate_logits(logits_final, y_t, test_mask, best_threshold)

    # Save final model
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "input_dim": input_dim,
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "dropout": dropout,
                "account_key": account_key,
                "val_days": val_days,
                "test_days": test_days,
                "threshold": best_threshold,
                "feature_cols": feature_cols,
                "split_strategy": split_strategy,
            },
            "tx_node_type": TX_NODE_TYPE,
        },
        GNN_MODEL_PATH,
    )

    # Save final epoch metrics as a compact summary
    final_metrics = {
        "epoch": num_epochs,
        "best_epoch": best_epoch,
        "threshold": best_threshold,
        "best_val_pr_auc": best_val_pr,
        "best_val_f1": best_val_f1,
        "split_sizes": split_sizes,
        "split_strategy": split_strategy,
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "feature_cols": feature_cols,
    }

    with open(GNN_METRICS_PATH, "w") as f:
        json.dump(final_metrics, f, indent=2)

    # Persist validation/test predictions for error analysis
    try:
        rows = []
        for split_name, mask_np in [("val", val_mask_np), ("test", test_mask_np)]:
            subset = df.loc[mask_np].copy()
            subset["prob"] = probs_final[mask_np]
            subset["pred_label"] = (subset["prob"] >= best_threshold).astype(int)
            subset["split"] = split_name
            subset["error_type"] = np.select(
                [
                    (subset[label_col] == 1) & (subset["pred_label"] == 0),
                    (subset[label_col] == 0) & (subset["pred_label"] == 1),
                ],
                ["FN", "FP"],
                default="correct",
            )
            keep_cols = ["split"]
            for cand in ["TransactionID", account_key, "TransactionDT", label_col]:
                if cand in subset.columns:
                    keep_cols.append(cand)
            keep_cols += ["prob", "pred_label", "error_type"]
            rows.append(subset[keep_cols])
        error_table = pd.concat(rows, ignore_index=True)
        error_table.to_parquet(GNN_VAL_TEST_PREDS_PATH, index=False)
    except Exception as exc:  # pragma: no cover - best effort artifact
        print(f"[train_gnn] warning: could not write error analysis table: {exc}")

    print(f"\nSaved model to {GNN_MODEL_PATH}")
    print(f"Saved metrics to {GNN_METRICS_PATH}")

    return {
        "history": metrics_history,
        "best_epoch": best_epoch,
        "best_threshold": best_threshold,
        "best_val_pr_auc": best_val_pr,
        "best_val_f1": best_val_f1,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train DGL heterograph GNN for IEEE fraud detection.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--pos-weight-scale", type=float, default=1.2, help="Multiplier on pos_weight to favor recall/PR-AUC.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    val_days = cfg["splits"]["val_days"]
    test_days = cfg["splits"].get("test_days", val_days)
    train_gnn(
        num_epochs=args.epochs,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        account_key=cfg["graph"]["account_key"],
        val_days=val_days,
        test_days=test_days,
        threshold=args.threshold,
        pos_weight_scale=args.pos_weight_scale,
        device=args.device,
    )


if __name__ == "__main__":
    main()
