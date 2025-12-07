"""
dgl_utils.py

Core toolkit for the MSML610 "Hard" DGL Recommender project.
Covers the full data science life cycle:

1) Data loading & cleaning
2) Graph construction
3) EDA & visualization
4) Feature engineering
5) Splits (random & temporal)
6) Core models (homogeneous baseline used by API)
7) Training loops (homogeneous + early stopping)
8) Metrics (pairs & user->set styles + adapters)
9) Baselines (Popularity & MF via TruncatedSVD)
10) RMSE helpers (edge-wise rating regression)
11) Recommendation helpers
12) Advanced (heterogeneous encoder for Example notebook)
"""

from __future__ import annotations

# 0) Imports & logging
# -----------------------------------------------------------------------------

import logging
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

import os
import dgl
import dgl.nn as dglnn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MultiLabelBinarizer
import random
from typing import Dict, List, Tuple
from tqdm import tqdm






_LOG = logging.getLogger(__name__)


# 1) Data loading & cleaning
# -----------------------------------------------------------------------------

def load_dummy_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create a small, clean, dummy MovieLens-like dataset for fast demos.
    Returns:
        movies_df, ratings_df
    """
    movies_df = pd.DataFrame(
        {
            "movieId": [1, 2, 3, 4, 5],
            "title": ["Movie A", "Movie B", "Movie C", "Movie D", "Movie E"],
            "genres": [
                "Action|Adventure",
                "Comedy|Romance",
                "Action|Sci-Fi",
                "Comedy",
                "Drama|Sci-Fi",
            ],
        }
    )
    ratings_df = pd.DataFrame(
        {
            "userId": [101, 101, 102, 103, 103, 103, 104, 104],
            "movieId": [1, 2, 3, 2, 3, 4, 1, 5],
            "rating": [5.0, 4.0, 3.5, 4.5, 5.0, 1.5, 5.0, 4.0],
            "timestamp": [
                964982703,
                964982704,
                964982705,
                964982706,
                964982707,
                964982708,
                964982709,
                964982710,
            ],
        }
    )
    return movies_df, ratings_df


# 2) Graph construction
# -----------------------------------------------------------------------------

def create_hetero_graph_from_pandas(
    ratings_df: pd.DataFrame, movies_df: Optional[pd.DataFrame] = None
) -> Tuple[dgl.DGLHeteroGraph, Dict[str, Dict[int, int]]]:
    """
    Build a DGL heterogeneous graph from raw DataFrames.

    Steps:
      1. Clean ratings (dropna on essential columns).
      2. Create 0-indexed mappings for users and movies (dict raw_id -> idx).
      3. Construct edges for ('user','rates','movie') and reverse ('movie','rated_by','user').

    Returns:
        g: DGLHeteroGraph
        maps: {'user': {raw_user_id->idx}, 'movie': {raw_movie_id->idx}}
    """
    _LOG.info("Creating heterogeneous graph...")

    cols = ["userId", "movieId", "rating", "timestamp"]
    ratings_df = ratings_df[cols].dropna()

    user_ids = ratings_df["userId"].unique()
    movie_ids = ratings_df["movieId"].unique()

    # Ensure movies that never got rated but exist in movies_df are included
    if movies_df is not None:
        extra_movie_ids = set(movies_df["movieId"].unique()) - set(movie_ids)
        if extra_movie_ids:
            movie_ids = np.concatenate([movie_ids, list(extra_movie_ids)])

    user_id_map = {int(u): i for i, u in enumerate(user_ids)}
    movie_id_map = {int(m): i for i, m in enumerate(movie_ids)}

    maps = {"user": user_id_map, "movie": movie_id_map}

    src = ratings_df["userId"].map(user_id_map).to_numpy()
    dst = ratings_df["movieId"].map(movie_id_map).to_numpy()

    g = dgl.heterograph({
        ("user","rates","movie"): (torch.tensor(src, dtype=torch.int64),
                                   torch.tensor(dst, dtype=torch.int64)),
        ("movie","rated_by","user"): (torch.tensor(dst, dtype=torch.int64),
                                      torch.tensor(src, dtype=torch.int64)),
    })

    _LOG.info("Graph created: n_user=%d n_movie=%d n_edges=%d",
              g.num_nodes("user"), g.num_nodes("movie"),
              g.num_edges(("user","rates","movie")))
    return g, maps


# 3) EDA & visualization
# -----------------------------------------------------------------------------

def get_graph_summary(g: dgl.DGLHeteroGraph) -> str:
    """
    Return a human-readable summary of a graph's structure and feature keys.
    """
    s = ["--- Graph Summary ---"]
    for ntype in g.ntypes:
        s.append(f"Node Type '{ntype}': {g.num_nodes(ntype)} nodes")
    for et in g.canonical_etypes:
        s.append(f"Edge Type {et}: {g.num_edges(et)} edges")
    s.append("---------------------")
    s.append(f"Node data keys: { {nt: list(g.nodes[nt].data.keys()) for nt in g.ntypes} }")
    s.append(f"Edge data keys: { {et: list(g.edges[et].data.keys()) for et in g.canonical_etypes} }")
    s.append("---------------------")
    return "\n".join(s)


def plot_degree_distribution(g: dgl.DGLHeteroGraph, ntype: str = "movie") -> None:
    _LOG.info("Plotting in-degree distribution for ntype='%s'", ntype)
    if ntype == "movie":
        deg = g.in_degrees(etype=("user","rates","movie")).cpu().numpy()
        title = "Movie In-Degree (ratings per movie)"
        xlabel = "Number of Ratings (In-Degree)"
    elif ntype == "user":
        deg = g.in_degrees(etype=("movie","rated_by","user")).cpu().numpy()
        title = "User In-Degree (ratings given)"
        xlabel = "Number of Ratings Given (In-Degree)"
    else:
        raise ValueError("ntype must be 'movie' or 'user'")
    plt.figure(figsize=(10,4))
    plt.hist(deg, bins=50, alpha=0.8, label=f"{ntype} in-degrees")
    plt.title(title); plt.xlabel(xlabel); plt.ylabel("Count")
    plt.yscale("log"); plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend(); plt.show()


# 4) Feature engineering
# -----------------------------------------------------------------------------

def add_edge_features(
    g: dgl.DGLHeteroGraph, ratings_df: pd.DataFrame, maps: Dict[str, Dict[int, int]]
) -> None:
    """
    Attach 'rating' (float32) and 'timestamp'/'ts' (int64) to both edge directions.
    Assumes graph edges were built in the same order as ratings_df after dropna.
    """
    rclean = ratings_df.dropna(subset=["userId", "movieId", "rating", "timestamp"])
    ratings = torch.tensor(rclean["rating"].to_numpy(), dtype=torch.float32)
    timestamps = torch.tensor(rclean["timestamp"].to_numpy(), dtype=torch.int64)

    for et in [("user", "rates", "movie"), ("movie", "rated_by", "user")]:
        g.edges[et].data["rating"] = ratings
        g.edges[et].data["timestamp"] = timestamps
        g.edges[et].data["ts"] = timestamps  # alias used by temporal split helpers

    _LOG.info("Added edge features: ['rating', 'timestamp', 'ts']")


def add_movie_node_features(
    g: dgl.DGLHeteroGraph, movies_df: Optional[pd.DataFrame], maps: Dict[str, Dict[int, int]]
) -> None:
    """
    One-hot encode movie genres and attach as g.nodes['movie'].data['feat'].
    """
    if movies_df is None:
        _LOG.warning("No movies_df provided; skipping movie node features.")
        return

    _LOG.info("Processing movie genre features ...")
    movie_id_map: Dict[int, int] = maps["movie"]

    movies_df = movies_df.copy()
    movies_df["genres_list"] = movies_df["genres"].fillna("").astype(str).str.split("|")
    mlb = MultiLabelBinarizer()
    mlb.fit(movies_df["genres_list"])

    genre_mat = mlb.transform(movies_df["genres_list"])
    features_df = pd.DataFrame(
        genre_mat, index=movies_df["movieId"].astype(int), columns=mlb.classes_
    )

    # reorder by contiguous index 0..M-1
    pairs = sorted(movie_id_map.items(), key=lambda kv: kv[1])  # (raw_id, idx)
    raw_ids_sorted = [raw for raw, _ in pairs]
    final_df = features_df.reindex(raw_ids_sorted).fillna(0.0)

    feat = torch.tensor(final_df.to_numpy(), dtype=torch.float32)
    g.nodes["movie"].data["feat"] = feat
    _LOG.info("Added movie 'feat' with shape=%s (#genres=%d)", tuple(feat.shape), feat.shape[1])


def add_user_node_features(g: dgl.DGLHeteroGraph, embedding_dim: int = 32) -> nn.Embedding:
    """
    Attach a learnable nn.Embedding for users as a demo of node features with no static inputs.
    NOTE: The training loops below use *separate* embedding tables for clarity. This function is for API demonstration.
    """
    num_users = g.num_nodes("user")
    emb = nn.Embedding(num_users, embedding_dim)
    nn.init.xavier_uniform_(emb.weight)
    g.nodes["user"].data["h"] = emb.weight  # demo-only; trainer has its own tables
    _LOG.info("Added user embedding 'h' (dim=%d) as demo node feature.", embedding_dim)
    return emb


# 5) Splits (random & temporal)
# -----------------------------------------------------------------------------

def make_edge_splits(
    g: dgl.DGLHeteroGraph,
    etype: Tuple[str, str, str] = ("user", "rates", "movie"),
    test_size: float = 0.1,
    val_size: float = 0.1,
    seed: int = 42,
) -> Dict[str, torch.Tensor]:
    rng = np.random.RandomState(seed)
    eids = np.arange(g.num_edges(etype=etype))
    rng.shuffle(eids)
    n = len(eids)

    # round, then enforce at least 1 for tiny graphs (leave >=1 for train)
    n_test = int(round(n * test_size))
    n_val  = int(round(n * val_size))
    if n >= 3:
        if test_size > 0: n_test = max(1, n_test)
        if val_size  > 0: n_val  = max(1, n_val)
        if n_test + n_val >= n:
            overflow = n_test + n_val - (n - 1)
            # reduce val first, then test
            reduce_val = min(n_val, overflow)
            n_val -= reduce_val
            overflow -= reduce_val
            n_test = max(0, n_test - overflow)

    test_eids = eids[:n_test]
    val_eids  = eids[n_test:n_test + n_val]
    train_eids = eids[n_test + n_val:]
    return {
        "train_eids": torch.tensor(train_eids, dtype=torch.int64),
        "val_eids":   torch.tensor(val_eids,   dtype=torch.int64),
        "test_eids":  torch.tensor(test_eids,  dtype=torch.int64),
    }


def make_temporal_edge_splits(
    g: dgl.DGLHeteroGraph,
    etype: Tuple[str, str, str] = ("user", "rates", "movie"),
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    ts_key: str = "ts",  # expect alias written by add_edge_features()
) -> Dict[str, torch.Tensor]:
    """
    Temporal split: older → train, then newer → val/test.
    Requires integer timestamps in g.edges[etype].data[ts_key].
    """
    ts = g.edges[etype].data.get(ts_key, None)
    if ts is None:
        raise ValueError(f"Edge timestamp '{ts_key}' not found on {etype}.")
    ts_np = ts.cpu().numpy()
    eids = np.arange(len(ts_np))
    order = np.argsort(ts_np)  # ascending time
    eids = eids[order]

    n = len(eids)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    test_eids = eids[-n_test:] if n_test > 0 else np.array([], dtype=int)
    val_eids = eids[-(n_test + n_val) : -n_test] if n_val > 0 else np.array([], dtype=int)
    train_eids = eids[: n - n_test - n_val]
    return {
        "train_eids": torch.tensor(train_eids, dtype=torch.int64),
        "val_eids": torch.tensor(val_eids, dtype=torch.int64),
        "test_eids": torch.tensor(test_eids, dtype=torch.int64),
    }


def eids_to_pairs(g, eids, etype=("user","rates","movie")) -> List[Tuple[int,int]]:
    if isinstance(eids, torch.Tensor):
        eid_t = eids.to(dtype=torch.int64, device=g.device)
    else:
        eid_t = torch.tensor(list(eids), dtype=torch.int64, device=g.device)
    u, v = g.find_edges(eid_t, etype=etype)
    return list(zip(u.tolist(), v.tolist()))


# 6) Core models (homogeneous baseline used by API)
# -----------------------------------------------------------------------------

class HomoGraphSAGEEncoder(nn.Module):
    """
    Homogeneous GraphSAGE encoder (didactic baseline).
    """
    def __init__(self, in_feats: int, hidden_feats: int = 64, out_feats: int = 64, num_layers: int = 2):
        super().__init__()
        from dgl.nn import SAGEConv
        dims = [in_feats] + [hidden_feats] * (num_layers - 1) + [out_feats]
        self.layers = nn.ModuleList([SAGEConv(dims[i], dims[i + 1], aggregator_type="mean") for i in range(num_layers)])
        self.act = nn.ReLU()

    def forward(self, g: dgl.DGLGraph, x: torch.Tensor) -> torch.Tensor:
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(g, h)
            if i < len(self.layers) - 1:
                h = self.act(h)
        return h


class LinkPredictorDot(nn.Module):
    """Dot-product scorer for link prediction."""
    def forward(self, user_emb: torch.Tensor, movie_emb: torch.Tensor) -> torch.Tensor:
        return (user_emb * movie_emb).sum(dim=-1)

    @staticmethod
    def predict_scores(user_embeds: torch.Tensor, movie_embeds: torch.Tensor) -> torch.Tensor:
        # [num_users, num_movies]
        return user_embeds @ movie_embeds.T


def negative_sampling_uniform(
    g: dgl.DGLHeteroGraph, pos_eids: torch.Tensor, num_neg: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Uniform negative sampling over items for each positive edge.
    Returns (neg_users, neg_items) aligned with pos edges.
    """
    num_items = g.num_nodes("movie")
    u_pos, _ = g.find_edges(pos_eids, etype=("user", "rates", "movie"))
    neg_users = u_pos.repeat_interleave(num_neg)
    neg_items = torch.randint(0, num_items, (pos_eids.numel() * num_neg,), device=neg_users.device)
    return neg_users, neg_items


# 7) Training loops (homogeneous)
# -----------------------------------------------------------------------------

def train_link_prediction(
    g: dgl.DGLHeteroGraph,
    splits: Dict[str, torch.Tensor],
    embed_dim: int = 32,
    epochs: int = 3,
    lr: float = 1e-3,
    device: str = "cpu",
    movie_feat_tensor: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    Didactic end-to-end training on homogeneous view:
      - Trainable tables per node type
      - Optional fusion of static movie features via a projector
      - Homo GraphSAGE encoder + dot-product scoring with BCE loss
    Returns:
      {'user_emb': Tensor[U,D], 'movie_emb': Tensor[M,D]} on CPU
    """
    dev = torch.device(device)
    uN, mN = g.num_nodes("user"), g.num_nodes("movie")
    u_table = nn.Embedding(uN, embed_dim).to(dev)
    m_table = nn.Embedding(mN, embed_dim).to(dev)

    proj = None
    if movie_feat_tensor is not None:
        proj = nn.Linear(movie_feat_tensor.shape[1], embed_dim).to(dev)

    hg = dgl.to_homogeneous(g)
    encoder = HomoGraphSAGEEncoder(embed_dim, embed_dim, embed_dim, num_layers=2).to(dev)
    scorer = LinkPredictorDot().to(dev)

    params = list(encoder.parameters()) + list(scorer.parameters()) + \
             list(u_table.parameters()) + list(m_table.parameters())
    if proj is not None:
        params += list(proj.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    bce = nn.BCEWithLogitsLoss()

    train_eids = splits["train_eids"]

    for ep in range(1, epochs + 1):
        x = torch.empty((uN + mN, embed_dim), device=dev)
        x[:uN] = u_table.weight
        x[uN:] = m_table.weight
        if proj is not None:
            x[uN:] = x[uN:] + proj(movie_feat_tensor.to(dev))

        z = encoder(hg, x)
        z_user, z_movie = z[:uN], z[uN:]

        u_pos, v_pos = g.find_edges(train_eids, etype=("user", "rates", "movie"))
        u_pos, v_pos = u_pos.to(dev), v_pos.to(dev)
        pos_scores = scorer(z_user[u_pos], z_movie[v_pos])
        pos_labels = torch.ones_like(pos_scores)

        neg_u, neg_v = negative_sampling_uniform(g, train_eids, num_neg=1)
        neg_u, neg_v = neg_u.to(dev), neg_v.to(dev)
        neg_scores = scorer(z_user[neg_u], z_movie[neg_v])
        neg_labels = torch.zeros_like(neg_scores)

        loss = bce(torch.cat([pos_scores, neg_scores]), torch.cat([pos_labels, neg_labels]))
        opt.zero_grad()
        loss.backward()
        opt.step()

        _LOG.info("Epoch %d/%d | loss=%.4f", ep, epochs, loss.item())

    return {"user_emb": z_user.detach().cpu(), "movie_emb": z_movie.detach().cpu()}


@torch.no_grad()
def _precision_at_k_from_pairs(user_emb, movie_emb, pairs, k=10) -> float:
    gt = defaultdict(set)
    for u, v in pairs:
        gt[int(u)].add(int(v))
    if not gt or movie_emb.numel() == 0:
        return 0.0
    k_eff = max(1, min(k, movie_emb.shape[0]))
    vals: List[float] = []
    for u in gt.keys():
        scores = (movie_emb @ user_emb[u].unsqueeze(1)).squeeze(1)
        topk = torch.topk(scores, k=k_eff).indices.cpu().tolist()
        hits = sum(1 for m in topk if m in gt[u])
        vals.append(hits / k_eff)
    return float(np.mean(vals)) if vals else 0.0


def train_link_prediction_with_early_stopping(
    g: dgl.DGLHeteroGraph,
    splits: Dict[str, torch.Tensor],
    embed_dim: int = 32,
    epochs: int = 20,
    lr: float = 1e-3,
    device: str = "cpu",
    movie_feat_tensor: Optional[torch.Tensor] = None,
    k_eval: int = 10,
    patience: int = 3,
    verbose: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Same as train_link_prediction(), but monitors val Precision@k and early-stops.
    Returns best {'user_emb','movie_emb'} observed on validation.
    """
    dev = torch.device(device)
    uN, mN = g.num_nodes("user"), g.num_nodes("movie")
    u_table = nn.Embedding(uN, embed_dim).to(dev)
    m_table = nn.Embedding(mN, embed_dim).to(dev)
    proj = nn.Linear(movie_feat_tensor.shape[1], embed_dim).to(dev) if movie_feat_tensor is not None else None

    hg = dgl.to_homogeneous(g)
    encoder = HomoGraphSAGEEncoder(embed_dim, embed_dim, embed_dim, num_layers=2).to(dev)
    scorer = LinkPredictorDot().to(dev)

    params = list(encoder.parameters()) + list(scorer.parameters()) + \
             list(u_table.parameters()) + list(m_table.parameters())
    if proj is not None:
        params += list(proj.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    bce = nn.BCEWithLogitsLoss()

    tr = splits["train_eids"]
    val_pairs = [(int(u), int(v)) for (u, v) in eids_to_pairs(g, splits["val_eids"])]

    best_p = -1.0
    best = {"user_emb": None, "movie_emb": None}
    stale = 0

    for ep in range(1, epochs + 1):
        x = torch.empty((uN + mN, embed_dim), device=dev)
        x[:uN] = u_table.weight
        x[uN:] = m_table.weight
        if proj is not None:
            x[uN:] = x[uN:] + proj(movie_feat_tensor.to(dev))

        z = encoder(hg, x)
        z_user, z_movie = z[:uN], z[uN:]

        u_pos, v_pos = g.find_edges(tr, etype=("user", "rates", "movie"))
        u_pos, v_pos = u_pos.to(dev), v_pos.to(dev)
        pos_scores = scorer(z_user[u_pos], z_movie[v_pos])
        pos_labels = torch.ones_like(pos_scores)

        neg_u, neg_v = negative_sampling_uniform(g, tr, num_neg=1)
        neg_u, neg_v = neg_u.to(dev), neg_v.to(dev)
        neg_scores = scorer(z_user[neg_u], z_movie[neg_v])
        neg_labels = torch.zeros_like(neg_scores)

        loss = bce(torch.cat([pos_scores, neg_scores]), torch.cat([pos_labels, neg_labels]))
        opt.zero_grad()
        loss.backward()
        opt.step()

        p_val = _precision_at_k_from_pairs(z_user, z_movie, val_pairs, k=k_eval)
        if verbose:
            _LOG.info("Epoch %d/%d | loss=%.4f | val P@%d=%.4f", ep, epochs, loss.item(), k_eval, p_val)

        if p_val > best_p + 1e-6:
            best_p = p_val
            best["user_emb"] = z_user.detach().cpu().clone()
            best["movie_emb"] = z_movie.detach().cpu().clone()
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                if verbose:
                    _LOG.info("Early stop (patience=%d). Best P@%d=%.4f", patience, k_eval, best_p)
                break

    if best["user_emb"] is None:
        best["user_emb"] = z_user.detach().cpu()
        best["movie_emb"] = z_movie.detach().cpu()
    return best


# 8) Metrics (two styles) + adapters
# -----------------------------------------------------------------------------

@torch.no_grad()
def evaluate_precision_recall_at_k(user_emb, movie_emb, test_pairs, k=10) -> Dict[str, float]:
    gt = defaultdict(set)
    for u, v in test_pairs:
        gt[int(u)].add(int(v))

    if movie_emb.numel() == 0:
        return {"precision@k": 0.0, "recall@k": 0.0}

    k_eff = max(1, min(k, movie_emb.shape[0]))
    precisions, recalls = [], []
    for u in gt.keys():
        scores = (movie_emb @ user_emb[u].unsqueeze(1)).squeeze(1)
        topk = torch.topk(scores, k=k_eff).indices.cpu().tolist()
        hits = sum(1 for m in topk if m in gt[u])
        precisions.append(hits / k_eff)
        recalls.append(hits / max(1, len(gt[u])))
    return {
        "precision@k": float(np.mean(precisions)) if precisions else 0.0,
        "recall@k": float(np.mean(recalls)) if recalls else 0.0,
    }

@torch.no_grad()
def evaluate_link_prediction_metrics(
    user_embeds: torch.Tensor,
    movie_embeds: torch.Tensor,
    test_user2items: Dict[int, Set[int]],
    k: int = 10,
) -> Dict[str, float]:
    """
    Precision@k / Recall@k where ground truth is dict user -> set(items).
    """
    scores_all = LinkPredictorDot.predict_scores(user_embeds, movie_embeds)
    precisions, recalls = [], []
    for u, items in test_user2items.items():
        if not items:
            continue
        scores = scores_all[u]
        _, topk_idx = torch.topk(scores, k)
        topk = set(topk_idx.cpu().numpy().tolist())
        hits = len(topk.intersection(items))
        precisions.append(hits / k)
        recalls.append(hits / len(items))
    return {f"precision_at_{k}": float(np.mean(precisions)) if precisions else 0.0,
            f"recall_at_{k}": float(np.mean(recalls)) if recalls else 0.0}


def pairs_to_user_sets(pairs: List[Tuple[int, int]]) -> Dict[int, Set[int]]:
    """Adapter: [(u,v)] -> {u: {v1,v2,...}}"""
    gt = defaultdict(set)
    for u, v in pairs:
        gt[int(u)].add(int(v))
    return gt


def user_sets_to_pairs(user2items: Dict[int, Set[int]]) -> List[Tuple[int, int]]:
    """Adapter: {u: {v1,v2}} -> [(u,v1),(u,v2), ...]"""
    return [(u, v) for u, vs in user2items.items() for v in vs]


# 9) Baselines (Popularity & MF)
# -----------------------------------------------------------------------------

def popularity_rank_from_train(
    g: dgl.DGLHeteroGraph,
    train_eids: torch.Tensor,
    etype: Tuple[str, str, str] = ("user", "rates", "movie"),
) -> np.ndarray:
    """
    Return global movie indices sorted by train popularity (descending).
    """
    _, v_tr = g.find_edges(train_eids, etype=etype)
    counts = np.bincount(v_tr.cpu().numpy(), minlength=g.num_nodes("movie"))
    return np.argsort(-counts)


def evaluate_popularity_at_k(
    g: dgl.DGLHeteroGraph,
    train_eids: torch.Tensor,
    test_pairs: List[Tuple[int, int]],
    k: int = 10,
    etype: Tuple[str, str, str] = ("user", "rates", "movie"),
) -> Dict[str, float]:
    """
    Recommend the same popularity list to all users (excluding their train items).
    """
    pop = popularity_rank_from_train(g, train_eids, etype=etype)

    # seen from train
    u_tr, v_tr = g.find_edges(train_eids, etype=etype)
    seen = defaultdict(set)
    for uu, vv in zip(u_tr.tolist(), v_tr.tolist()):
        seen[int(uu)].add(int(vv))

    # ground truth from test
    gt = defaultdict(set)
    for u, v in test_pairs:
        gt[int(u)].add(int(v))

    precisions, recalls = [], []
    for u in gt.keys():
        topk = [m for m in pop.tolist() if m not in seen.get(u, set())][:k]
        hits = sum(1 for m in topk if m in gt[u])
        precisions.append(hits / k)
        recalls.append(hits / max(1, len(gt[u])))
    return {"precision@k": float(np.mean(precisions)) if precisions else 0.0,
            "recall@k": float(np.mean(recalls)) if recalls else 0.0}


def build_user_item_csr(
    num_users: int, num_items: int, pairs: List[Tuple[int, int]], ratings: Iterable[float]
) -> csr_matrix:
    """
    Build a CSR user-item matrix from (u,v) pairs and ratings.
    """
    u_idx = [int(u) for (u, _) in pairs]
    v_idx = [int(v) for (_, v) in pairs]
    vals = np.asarray(list(ratings), dtype=np.float32)
    return csr_matrix((vals, (u_idx, v_idx)), shape=(num_users, num_items), dtype=np.float32)


def mf_truncated_svd_embeddings(
    train_csr: csr_matrix, n_components: int = 64, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Matrix factorization via TruncatedSVD. Returns user & item embeddings.
    Shapes: U:[num_users,k], V:[num_items,k]
    Notes:
      - scikit-learn randomized SVD requires n_components < min(n_samples, n_features)
      - Also must be <= n_features (num_items)
    """
    m, n = train_csr.shape  # m = num_users, n = num_items
    max_k = max(1, min(m, n) - 1)         # strict bound for randomized_svd
    k = min(n_components, max_k)

    svd = TruncatedSVD(n_components=k, random_state=random_state)
    U = svd.fit_transform(train_csr)  # [m, k]
    V = svd.components_.T             # [n, k]
    return U.astype(np.float32), V.astype(np.float32)


# 10) RMSE helpers (edge-wise rating regression)
# -----------------------------------------------------------------------------

def fit_edge_regressor_ridge(
    user_emb: torch.Tensor,
    movie_emb: torch.Tensor,
    pairs: List[Tuple[int, int]],
    ratings: Iterable[float],
    alpha: float = 1.0,
):
    """
    Fit a linear regressor on frozen embeddings to predict ratings (for RMSE).
    """
    X = [np.concatenate([user_emb[u].numpy(), movie_emb[v].numpy()]) for u, v in pairs]
    y = np.asarray(list(ratings), dtype=np.float32)
    model = Ridge(alpha=alpha).fit(np.asarray(X, dtype=np.float32), y)
    return model


def rmse_from_regressor(model, user_emb, movie_emb, pairs, ratings) -> float:
    if len(pairs) == 0:
        return float("nan")
    X = [np.concatenate([user_emb[u].numpy(), movie_emb[v].numpy()]) for u, v in pairs]
    y_true = np.asarray(list(ratings), dtype=np.float32)
    y_pred = model.predict(np.asarray(X, dtype=np.float32))
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# 11) Recommendation helpers
# -----------------------------------------------------------------------------

def build_user_seen_map_from_graph(
    g: dgl.DGLHeteroGraph,
    eids: Iterable[int],
    etype: Tuple[str, str, str] = ("user", "rates", "movie"),
) -> Dict[int, Set[int]]:
    """
    Map user -> set(items) for edges identified by EIDs (e.g., training set).
    """
    u, v = g.find_edges(torch.tensor(list(eids), dtype=torch.int64), etype=etype)
    out: Dict[int, Set[int]] = defaultdict(set)
    for uu, vv in zip(u.tolist(), v.tolist()):
        out[int(uu)].add(int(vv))
    return out


def build_user_seen_map_from_df(
    ratings_df: pd.DataFrame, maps: Dict[str, Dict[int, int]]
) -> Dict[int, Set[int]]:
    """
    Map user_idx -> set(original_movie_ids) from a ratings DataFrame and ID maps.
    """
    user_map = maps["user"]
    seen: Dict[int, Set[int]] = defaultdict(set)
    for _, row in ratings_df.iterrows():
        u_idx = user_map.get(int(row["userId"]))
        if u_idx is not None:
            seen[u_idx].add(int(row["movieId"]))
    return seen


def build_user_seen_map(
    g_or_df: Union[dgl.DGLHeteroGraph, pd.DataFrame],
    eids_or_maps: Union[Iterable[int], Dict[str, Dict[int, int]]],
    etype: Tuple[str, str, str] = ("user", "rates", "movie"),
) -> Dict[int, Set[int]]:
    """
    Convenience wrapper: supports both graph+EIDs and DataFrame+maps forms.
    """
    if isinstance(g_or_df, dgl.DGLHeteroGraph):
        return build_user_seen_map_from_graph(g_or_df, eids_or_maps, etype=etype)  # type: ignore[arg-type]
    else:
        return build_user_seen_map_from_df(g_or_df, eids_or_maps)  # type: ignore[arg-type]


def recommend_topk_for_user(
    user_idx: int,
    user_embeds: torch.Tensor,
    movie_embeds: torch.Tensor,
    k: int,
    seen_items: Optional[Set[int]] = None,
    maps: Optional[Dict[str, Dict[int, int]]] = None,
) -> List[int]:
    """
    Recommend top-K movie indices (or original IDs if maps provided) for a user,
    filtering out 'seen_items' if provided.
    """
    scores = LinkPredictorDot.predict_scores(user_embeds, movie_embeds)[user_idx]
    ranked = torch.topk(scores, k=scores.shape[0]).indices.cpu().tolist()
    seen_idx: Set[int] = set()
    if seen_items:
        if maps is not None:
            raw2idx = maps["movie"]          # raw -> idx
            seen_idx = { raw2idx[r] for r in seen_items if r in raw2idx }
        else:
            seen_idx = set(seen_items)

    if seen_idx:
        ranked = [m for m in ranked if m not in seen_idx]

    ranked = ranked[:k]
    if maps is None:
        return ranked
    idx_to_raw = {idx: raw for raw, idx in maps["movie"].items()}
    return [idx_to_raw[m] for m in ranked]


def id_maps_to_title_lookup(
    movies_df: Optional[pd.DataFrame],
    item_map: Optional[Union[Dict[int, int], np.ndarray]],
) -> Dict[int, str]:
    """
    Build int-index -> movie title mapping based on the contiguous index space.
    Accepts either:
      - dict raw_id -> idx, or
      - np.ndarray where position i holds original movieId for index i
    """
    if movies_df is None or item_map is None:
        return {}

    title_by_idx: Dict[int, str] = {}
    if isinstance(item_map, dict):
        # dict raw -> idx
        for _, row in movies_df.iterrows():
            raw = int(row["movieId"])
            if raw in item_map:
                idx = int(item_map[raw])
                title_by_idx[idx] = str(row.get("title", f"movie_{raw}"))
    else:
        # ndarray style (idx -> raw)
        idx_by_raw = {int(raw): i for i, raw in enumerate(item_map)}
        for _, row in movies_df.iterrows():
            raw = int(row["movieId"])
            if raw in idx_by_raw:
                idx = int(idx_by_raw[raw])
                title_by_idx[idx] = str(row.get("title", f"movie_{raw}"))
    return title_by_idx


# 12) Advanced (Example notebook): hetero encoder (blocks-based)
# -----------------------------------------------------------------------------

class HeteroGraphSAGEEncoder(nn.Module):
    """
    Blocks-based Heterogeneous GraphSAGE encoder for neighbor sampling.
    This is intended for the Example notebook, not used by the API trainer.
    """
    def __init__(self, in_feats_dict: Dict[str, int], hid_feats: int, out_feats: int):
        super().__init__()
        self.conv1 = dglnn.HeteroGraphConv(
            {
                "rates": dglnn.SAGEConv(
                    in_feats=(in_feats_dict["user"], in_feats_dict["movie"]),
                    out_feats=hid_feats,
                    aggregator_type="mean",
                ),
                "rated_by": dglnn.SAGEConv(
                    in_feats=(in_feats_dict["movie"], in_feats_dict["user"]),
                    out_feats=hid_feats,
                    aggregator_type="mean",
                ),
            },
            aggregate="sum",
        )
        self.conv2 = dglnn.HeteroGraphConv(
            {
                "rates": dglnn.SAGEConv(
                    in_feats=(hid_feats, hid_feats),
                    out_feats=out_feats,
                    aggregator_type="mean",
                ),
                "rated_by": dglnn.SAGEConv(
                    in_feats=(hid_feats, hid_feats),
                    out_feats=out_feats,
                    aggregator_type="mean",
                ),
            },
            aggregate="sum",
        )

    def forward(self, blocks: List[dgl.DGLBlock], x_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        h = self.conv1(blocks[0], x_dict)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(blocks[1], h)
        return h



## UTIL FUNCTIONS FOR DGL.EXAMPLE.IPYNB


# ======================================================
# 1. DATA LOADING
# ======================================================

def load_movielens_data(raw_dir: str = "data/raw/"):
    """
    Load the core MovieLens CSVs from disk.

    Parameters
    ----------
    raw_dir : str
        Path to the folder containing the raw MovieLens CSVs:
        - ratings.csv with columns [userId, movieId, rating, timestamp]
        - movies.csv with columns [movieId, title, genres]

    Returns
    -------
    ratings : pd.DataFrame
        User-movie interactions (explicit ratings).
    movies : pd.DataFrame
        Movie metadata (title, genres, etc.).
    """

    ratings_path = os.path.join(raw_dir, "rating.csv")
    movies_path = os.path.join(raw_dir, "movie.csv")

    # Read CSVs
    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)

    # ---- Basic sanity checks to fail fast if data isn't correct ----
    expected_ratings_cols = {"userId", "movieId", "rating", "timestamp"}
    expected_movies_cols = {"movieId", "title", "genres"}

    if not expected_ratings_cols.issubset(ratings.columns):
        missing = expected_ratings_cols.difference(ratings.columns)
        raise ValueError(f"ratings.csv missing columns: {missing}")

    if not expected_movies_cols.issubset(movies.columns):
        missing = expected_movies_cols.difference(movies.columns)
        raise ValueError(f"movies.csv missing columns: {missing}")
    
    # --------- snaity check done ---------

    return ratings, movies


# ======================================================
# 2. SAMPLING USERS
# ======================================================


def sample_users(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    n_users: int = 10000,
    min_ratings_per_user: int = 20,
    out_dir: str = "data/processed/"
):
    """
    Create a smaller, consistent MovieLens slice by selecting a subset of users
    and keeping all their ratings + corresponding movies.

    Parameters
    ----------
    ratings : pd.DataFrame
        Raw ratings data with columns [userId, movieId, rating, timestamp].
    movies : pd.DataFrame
        Raw movies data with columns [movieId, title, genres].
    n_users : int, default=5000
        Number of distinct users to include.
    min_ratings_per_user : int, default=20
        Ensure sampled users have at least this many ratings.
    out_dir : str
        Directory to save the sampled CSVs.

    Returns
    -------
    ratings_sampled, movies_sampled : pd.DataFrame
        Filtered subsets ready for preprocessing.
    """

    #  ------ Filter out users with too few ratings first ------
    user_counts = ratings["userId"].value_counts()
    eligible_users = user_counts[user_counts >= min_ratings_per_user].index.tolist()

    if len(eligible_users) < n_users:
        raise ValueError(
            f"Only {len(eligible_users)} users meet the minimum ratings threshold."
        )

    # ------ Randomly select N users from the eligible set ------
    sampled_users = random.sample(eligible_users, n_users)
    ratings_sampled = ratings[ratings["userId"].isin(sampled_users)].copy()
 
    # ------ Keep only movies that appear in this subset ------
    sampled_movie_ids = ratings_sampled["movieId"].unique().tolist()
    movies_sampled = movies[movies["movieId"].isin(sampled_movie_ids)].copy()

    # ------ Create output directory and save ------
    os.makedirs(out_dir, exist_ok=True)
    ratings_out = os.path.join(out_dir, "ratings_sampled_raw.csv")
    movies_out = os.path.join(out_dir, "movies_sampled_raw.csv")

    ratings_sampled.to_csv(ratings_out, index=False)
    movies_sampled.to_csv(movies_out, index=False)

    print(
        f"Sampled {len(sampled_users)} users, "
        f"{len(sampled_movie_ids)} movies, "
        f"{len(ratings_sampled)} ratings."
    )
    print(f"Saved to: {ratings_out} and {movies_out}")

    return ratings_sampled, movies_sampled


# ======================================================
# 3. PREPROCESSING (ID MAPPING + GENRES)
# ======================================================

def build_id_mappings(ratings: pd.DataFrame, movies: pd.DataFrame) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    Create contiguous integer index mappings for users and movies.

    Parameters
    ----------
    ratings : pd.DataFrame
        Must contain 'userId' and 'movieId' columns from MovieLens.

    Returns
    -------
    user2idx : dict[int, int]
        Maps original userId -> 0..num_users-1

    movie2idx : dict[int, int]
        Maps original movieId -> 0..num_movies-1
    """
    unique_users = ratings["userId"].unique()
    unique_movies = movies["movieId"].unique()

    user2idx = {uid: i for i, uid in enumerate(sorted(unique_users))} #mapping with the help of dict and enumerate
    movie2idx = {mid: i for i, mid in enumerate(sorted(unique_movies))}

    return user2idx, movie2idx


def apply_id_mappings(ratings: pd.DataFrame, user2idx: Dict[int, int], movie2idx: Dict[int, int]) -> pd.DataFrame:
    """
    Add mapped `user_idx` and `movie_idx` columns to the ratings table.

    Any row whose userId/movieId is not found in the mapping will raise,
    which protects us from silent data drift.

    Returns
    -------
    ratings_idx : pd.DataFrame
        Columns: user_idx, movie_idx, rating, timestamp
    """
    ratings_idx = ratings.copy()

    ratings_idx["user_idx"] = ratings_idx["userId"].map(user2idx)
    ratings_idx["movie_idx"] = ratings_idx["movieId"].map(movie2idx)

    if ratings_idx["user_idx"].isna().any():
        raise ValueError("Found a rating with userId not in user2idx mapping.")
    if ratings_idx["movie_idx"].isna().any():
        raise ValueError("Found a rating with movieId not in movie2idx mapping.")

    # Keep only what we need downstream
    ratings_idx = ratings_idx[["user_idx", "movie_idx", "rating", "timestamp"]]

    return ratings_idx


def explode_genres(movies: pd.DataFrame) -> pd.DataFrame:
    """
    Turn pipe-separated 'genres' column into multi-hot indicator columns.

    Example:
    'Action|Sci-Fi' -> genre_Action=1, genre_Sci-Fi=1, all else=0

    We return a new DataFrame with movieId, title, genres (original string),
    plus the one-hot columns.
    """
    movies_proc = movies.copy()

    # Split "Action|Adventure|Sci-Fi" into lists
    movies_proc["genre_list"] = movies_proc["genres"].apply(lambda x: x.split("|") if isinstance(x, str) else [])

    # Get full set of unique genres
    all_genres = sorted({g for glist in movies_proc["genre_list"] for g in glist if g != "(no genres listed)"})

    # For each genre, create a binary column
    for g in all_genres:
        col_name = f"genre_{g}"
        movies_proc[col_name] = movies_proc["genre_list"].apply(lambda gl: 1 if g in gl else 0)

    # Drop helper list
    movies_proc = movies_proc.drop(columns=["genre_list"])

    return movies_proc, all_genres


def attach_movie_indices(movies: pd.DataFrame, movie2idx: Dict[int, int]) -> pd.DataFrame:
    """
    Add a `movie_idx` column to the movie metadata table.

    Returns
    -------
    movies_idx : pd.DataFrame
        Contains movie_idx, movieId, title, genres, and genre_* columns.
    """
    movies_idx = movies.copy()
    movies_idx["movie_idx"] = movies_idx["movieId"].map(movie2idx)

    # Some movies in movies.csv might never be rated in ratings.csv.
    # Those won't have movie_idx and can be dropped for training.
    movies_idx = movies_idx.dropna(subset=["movie_idx"]).copy()
    movies_idx["movie_idx"] = movies_idx["movie_idx"].astype(int)

    movies_idx = movies_idx.sort_values("movie_idx").reset_index(drop=True)


    return movies_idx


def save_processed(users_df: pd.DataFrame, movies_df: pd.DataFrame, ratings_df: pd.DataFrame, out_dir: str = "data/processed/") -> None:
    """
    Save cleaned / mapped data to disk for reuse by later pipeline steps.
    """
    os.makedirs(out_dir, exist_ok=True)

    users_df.to_csv(os.path.join(out_dir, "users.csv"), index=False)
    movies_df.to_csv(os.path.join(out_dir, "movies.csv"), index=False)
    ratings_df.to_csv(os.path.join(out_dir, "ratings.csv"), index=False)




def preprocess_and_save(ratings_raw: pd.DataFrame,
                        movies_raw: pd.DataFrame,
                        out_dir: str = "data/processed/") -> None:
    """
    Full preprocessing pipeline:
    - build ID mappings
    - map ratings to user_idx/movie_idx
    - build movie genre features
    - build users table
    - save everything
    """

    # 1. Build mappings
    user2idx, movie2idx = build_id_mappings(ratings_raw, movies_raw)

    # 2. Apply mappings to ratings
    ratings_idx = apply_id_mappings(ratings_raw, user2idx, movie2idx)

    # 3. Process movie genres into multi-hot
    movies_genre_expanded, all_genres = explode_genres(movies_raw)

    # 4. Attach movie_idx to the movie table (drop movies with no ratings)
    movies_idx = attach_movie_indices(movies_genre_expanded, movie2idx)

    # 5. Build users table
    #    We just need (original userId, user_idx) for reference later.
    users_df = (
        pd.DataFrame(list(user2idx.items()), columns=["userId", "user_idx"])
        .sort_values("user_idx")
        .reset_index(drop=True)
    )

    # 6. Save everything
    save_processed(users_df, movies_idx, ratings_idx, out_dir=out_dir)

    # We also save mappings, genres list, etc. as reusable artifacts)
    pd.Series(user2idx).to_json(os.path.join(out_dir, "user2idx.json"))
    pd.Series(movie2idx).to_json(os.path.join(out_dir, "movie2idx.json"))
    pd.Series(all_genres).to_json(os.path.join(out_dir, "all_genres.json"))



# ======================================================
# 4. TRAIN/VAL/TEST SPLIT
# ======================================================


def _split_user_interactions(
    df_user: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
):
    """
    Given all ratings for ONE user (already filtered and sorted by timestamp),
    break them into train/val/test by time.

    Returns three DataFrames: train_df, val_df, test_df.
    Some of these can be empty if the user doesn't have enough ratings.
    """

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "splits must sum to 1"

    n = len(df_user)
    if n == 0:
        return (
            df_user.iloc[0:0].copy(),
            df_user.iloc[0:0].copy(),
            df_user.iloc[0:0].copy(),
        )

    # indices for split boundaries
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_df = df_user.iloc[:train_end].copy()
    val_df = df_user.iloc[train_end:val_end].copy()
    test_df = df_user.iloc[val_end:].copy()

    # If user is tiny (like 3 ratings total), val/test may end up empty. That's fine.
    return train_df, val_df, test_df


def _generate_negative_samples_for_split(
    pos_df: pd.DataFrame,
    all_user_hist: Dict[int, set],
    all_movie_ids: np.ndarray,
    num_neg_per_pos: float = 1.0,
    seed: int = 42,
):
    """
    For a given split (train/val/test), we have a set of positive pairs
    (user_idx, movie_idx). We generate negative pairs by pairing each user
    with movies they have NOT interacted with.

    We try to keep roughly `num_neg_per_pos` negatives per positive.

    Returns
    -------
    neg_df : pd.DataFrame with columns:
        user_idx, movie_idx, label (=0)
    """

    rng = np.random.default_rng(seed)

    neg_rows = []

    # We'll group positives by user to be efficient.
    grouped = pos_df.groupby("user_idx")["movie_idx"].apply(list)

    for user_idx, pos_movies_for_user in grouped.items():
        already_seen = all_user_hist[user_idx]  # set of movie_idx
        num_pos = len(pos_movies_for_user)
        num_neg = int(num_pos * num_neg_per_pos)

        # sample candidates the user has NOT seen
        # we sample with replacement if needed to avoid edge cases
        available_movies = np.setdiff1d(all_movie_ids, np.array(list(already_seen)))

        if len(available_movies) == 0:
            # weird edge case: user has seen literally all movies in this subset
            continue

        if num_neg > len(available_movies):
            sampled_neg_movies = rng.choice(available_movies, size=num_neg, replace=True)
        else:
            sampled_neg_movies = rng.choice(available_movies, size=num_neg, replace=False)

        for m in sampled_neg_movies:
            neg_rows.append(
                {
                    "user_idx": user_idx,
                    "movie_idx": int(m),
                    "label": 0,
                }
            )

    if len(neg_rows) == 0:
        neg_df = pd.DataFrame(columns=["user_idx", "movie_idx", "label"])
    else:
        neg_df = pd.DataFrame(neg_rows)

    return neg_df


def make_splits(
    ratings_path: str = "data/processed/ratings.csv",
    out_dir: str = "data/processed/",
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    min_interactions_required: int = 5,
    num_neg_per_pos: float = 1.0,
):
    """
    Main driver for creating train/val/test splits + negative samples.

    Steps:
    1. Load processed ratings (already has user_idx, movie_idx).
    2. Sort each user's interactions by timestamp.
    3. Split by time into train/val/test.
    4. Build per-user interaction history set (for negative sampling).
    5. For each split, generate negatives.
    6. Save all CSVs to disk.

    Output CSV columns:
      * *_pos.csv: user_idx, movie_idx, rating, timestamp, label=1
      * *_neg.csv: user_idx, movie_idx, label=0
    """

    os.makedirs(out_dir, exist_ok=True)

    ratings = pd.read_csv(ratings_path)

    # safety check
    expected_cols = {"user_idx", "movie_idx", "rating", "timestamp"}
    if not expected_cols.issubset(ratings.columns):
        raise ValueError(f"ratings.csv missing columns {expected_cols - set(ratings.columns)}")

    # Sort by timestamp within each user
    ratings_sorted = ratings.sort_values(["user_idx", "timestamp"]).reset_index(drop=True)

    all_users = ratings_sorted["user_idx"].unique()
    all_movies = ratings_sorted["movie_idx"].unique()

    train_parts = []
    val_parts = []
    test_parts = []

    # We'll also build a dict of full user histories (all movies they've interacted with),
    # for later negative sampling.
    user_history = {int(u): set() for u in all_users}

    # We'll populate user_history first using ALL interactions (full timeline).
    for row in ratings_sorted.itertuples(index=False):
        user_history[int(row.user_idx)].add(int(row.movie_idx))

    # Now, for each user, split their rows into train/val/test based on timestamp order.
    for u in all_users:
        df_user = ratings_sorted[ratings_sorted["user_idx"] == u]

        # skip very cold users
        if len(df_user) < min_interactions_required:
            # Put ALL of them into train (model at least learns embedding),
            # and nothing into val/test for this user.
            df_user_train = df_user.copy()
            df_user_val = df_user.iloc[0:0].copy()
            df_user_test = df_user.iloc[0:0].copy()
        else:
            df_user_train, df_user_val, df_user_test = _split_user_interactions(
                df_user,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
            )

        train_parts.append(df_user_train)
        val_parts.append(df_user_val)
        test_parts.append(df_user_test)

    train_df = pd.concat(train_parts, ignore_index=True)
    val_df = pd.concat(val_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)

    # Add label=1 on positives
    train_pos = train_df.copy()
    train_pos["label"] = 1

    val_pos = val_df.copy()
    val_pos["label"] = 1

    test_pos = test_df.copy()
    test_pos["label"] = 1

    # Generate negatives for each split
    all_movie_ids = np.array(all_movies)

    train_neg = _generate_negative_samples_for_split(
        train_pos, user_history, all_movie_ids, num_neg_per_pos=num_neg_per_pos, seed=42
    )
    val_neg = _generate_negative_samples_for_split(
        val_pos, user_history, all_movie_ids, num_neg_per_pos=num_neg_per_pos, seed=123
    )
    test_neg = _generate_negative_samples_for_split(
        test_pos, user_history, all_movie_ids, num_neg_per_pos=num_neg_per_pos, seed=999
    )

    # Reorder / keep consistent columns for saving
    cols_pos = ["user_idx", "movie_idx", "rating", "timestamp", "label"]
    cols_neg = ["user_idx", "movie_idx", "label"]

    train_pos = train_pos[cols_pos]
    val_pos = val_pos[cols_pos]
    test_pos = test_pos[cols_pos]

    train_neg = train_neg[cols_neg]
    val_neg = val_neg[cols_neg]
    test_neg = test_neg[cols_neg]

    # Save all six CSVs
    train_pos.to_csv(os.path.join(out_dir, "train_pos.csv"), index=False)
    train_neg.to_csv(os.path.join(out_dir, "train_neg.csv"), index=False)

    val_pos.to_csv(os.path.join(out_dir, "val_pos.csv"), index=False)
    val_neg.to_csv(os.path.join(out_dir, "val_neg.csv"), index=False)

    test_pos.to_csv(os.path.join(out_dir, "test_pos.csv"), index=False)
    test_neg.to_csv(os.path.join(out_dir, "test_neg.csv"), index=False)

    print("Data split complete.")
    print(f"Train: {len(train_pos)} pos / {len(train_neg)} neg")
    print(f"Val:   {len(val_pos)} pos / {len(val_neg)} neg")
    print(f"Test:  {len(test_pos)} pos / {len(test_neg)} neg")

    return {
        "train_pos": train_pos,
        "train_neg": train_neg,
        "val_pos": val_pos,
        "val_neg": val_neg,
        "test_pos": test_pos,
        "test_neg": test_neg,
    }


# ======================================================
# 5. NEGATIVE SAMPLING
# ======================================================

# def negative_sampling(pos_df, num_movies=None, num_neg=5):
#     """
#     For each positive interaction, sample K negative movies.
#     """
#     if num_movies is None:
#         num_movies = pos_df["movie_idx"].max() + 1

#     neg_samples = []
#     user_pos = pos_df.groupby("user_idx")["movie_idx"].apply(set).to_dict()

#     for user, group in pos_df.groupby("user_idx"):
#         pos_movies = user_pos[user]
#         for movie in pos_movies:
#             for _ in range(num_neg):
#                 neg = np.random.randint(0, num_movies)
#                 while neg in pos_movies:
#                     neg = np.random.randint(0, num_movies)
#                 neg_samples.append((user, neg))

#     neg_df = pd.DataFrame(neg_samples, columns=["user_idx", "movie_idx"])
#     return neg_df


# ======================================================
# 6. DGL GRAPH CONSTRUCTION
# ======================================================

def build_graph(
    ratings_path="data/processed/train_pos.csv",
    movies_path="data/processed/movies.csv",
    out_path="data/graphs/train_graph.bin"
):
    """
    Build a DGL heterogeneous bipartite graph with *bidirectional* edges:

       user --rates--> movie
       movie --rev_rates--> user

    using only TRAIN positive edges to avoid data leakage.
    """

    # === Load data ===
    train_pos = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path).sort_values("movie_idx").reset_index(drop=True)

    # --- Sanity: dense movie_idx ---
    num_movies = movies["movie_idx"].max() + 1
    assert movies["movie_idx"].nunique() == num_movies, (
        f"movies.csv movie_idx not dense: max={num_movies-1}, "
        f"nunique={movies['movie_idx'].nunique()}"
    )

    # Filter edges to valid movies, just in case
    valid_movie_ids = set(movies["movie_idx"].unique())
    mask = train_pos["movie_idx"].isin(valid_movie_ids)
    if (~mask).sum() > 0:
        print(f"Dropping {(~mask).sum()} train edges with unknown movie_idx.")
    train_pos = train_pos[mask].copy()

    # === Edge lists ===
    src_users = torch.tensor(train_pos["user_idx"].values, dtype=torch.int64)
    dst_movies = torch.tensor(train_pos["movie_idx"].values, dtype=torch.int64)

    num_users = train_pos["user_idx"].max() + 1

    # === Build heterograph with forward + reverse relations ===
    g = dgl.heterograph(
        {
            ("user", "rates", "movie"): (src_users, dst_movies),
            ("movie", "rev_rates", "user"): (dst_movies, src_users),
        },
        num_nodes_dict={
            "user": num_users,
            "movie": num_movies,
        }
    )

    # === Add movie genre features ===
    genre_cols = [c for c in movies.columns if c.startswith("genre_")]
    movie_feats = torch.tensor(movies[genre_cols].values, dtype=torch.float32)
    assert movie_feats.shape[0] == num_movies

    g.nodes["movie"].data["genre"] = movie_feats

    # Optional: store ratings on edges (only on forward edges)
    g.edges["rates"].data["rating"] = torch.tensor(
        train_pos["rating"].values, dtype=torch.float32
    )

    # === Save graph ===
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    dgl.save_graphs(out_path, g)

    print("Graph construction complete (bidirectional).")
    print(f"Users:  {num_users}")
    print(f"Movies: {num_movies}")
    print(f"Edges (rates):      {g.num_edges(('user','rates','movie'))}")
    print(f"Edges (rev_rates):  {g.num_edges(('movie','rev_rates','user'))}")
    print(f"Saved graph to: {out_path}")

    return g




# ======================================================
# 7. DATA LOADER (FOR TRAINING)
# ======================================================

def build_dataloader(pos_path, neg_path, batch_size=2048, shuffle=True):
    pos_df = pd.read_csv(pos_path)
    neg_df = pd.read_csv(neg_path)

    pos_df["label"] = 1
    neg_df["label"] = 0

    df = pd.concat([pos_df, neg_df], ignore_index=True)

    user_idx = torch.tensor(df["user_idx"].values, dtype=torch.long)
    movie_idx = torch.tensor(df["movie_idx"].values, dtype=torch.long)
    labels = torch.tensor(df["label"].values, dtype=torch.float32)

    dataset = TensorDataset(user_idx, movie_idx, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return loader, len(df)



# ======================================================
# 8. EVALUATION METRICS
# ======================================================

def precision_at_k(ranklist, k):
    """ranklist = list of 0/1 relevance sorted by predicted score descending."""
    return np.sum(ranklist[:k]) / k


def recall_at_k(ranklist, k):
    """Relevant items divided by total relevant items."""
    total_pos = np.sum(ranklist)
    if total_pos == 0:
        return 0.0
    return np.sum(ranklist[:k]) / total_pos


def ndcg_at_k(ranklist, k):
    """Discounted gain."""
    ranklist = np.array(ranklist)
    dcg = 0.0
    for i in range(k):
        if ranklist[i] == 1:
            dcg += 1.0 / np.log2(i + 2)

    # Ideal DCG
    ideal = np.sum(ranklist)
    if ideal == 0:
        return 0.0

    idcg = 0.0
    for i in range(int(ideal)):
        idcg += 1.0 / np.log2(i + 2)

    return dcg / idcg


# ======================================================
# 9. FULL-RANKING EVALUATION
# ======================================================

def evaluate_full_ranking(
    model,
    g,
    pos_path,
    k=10,
    device="cpu"
):
    """
    Full ranking evaluation:
        For each user in the validation or test set:
            - score ALL movies
            - sort by predicted score
            - compute P@K, R@K, NDCG@K

    Parameters
    ----------
    model : GNNRecommender
    g     : DGLGraph (train graph)
    pos_path : path to val_pos.csv or test_pos.csv
               should contain [user_idx, movie_idx]
    k : cutoff for metrics
    """

    df = pd.read_csv(pos_path)
    users = df["user_idx"].unique()

    print(f"Evaluating {len(users)} users with full ranking...")

    model.eval()
    g = g.to(device)

    # Compute embeddings ONCE
    with torch.no_grad():
        user_emb, movie_emb = model.encode(g)
        user_emb = user_emb.to(device)
        movie_emb = movie_emb.to(device)

    num_movies = movie_emb.shape[0]

    all_prec, all_rec, all_ndcg = [], [], []

    # Create mapping of user → their positive movies
    user_pos = {}
    for u, m in zip(df["user_idx"], df["movie_idx"]):
        user_pos.setdefault(u, []).append(m)

    for u in tqdm(users):
        # 1. Score ALL MOVIES for this user
        u_idx_tensor = torch.tensor([u], dtype=torch.long, device=device).repeat(num_movies)
        m_idx_tensor = torch.arange(num_movies, device=device)

        with torch.no_grad():
            logits = model.link_predictor(user_emb, movie_emb, u_idx_tensor, m_idx_tensor)
            scores = torch.sigmoid(logits).cpu().numpy()

        # 2. Build relevance vector for ranking
        rel = np.zeros(num_movies, dtype=np.int32)
        for m in user_pos[u]:
            rel[m] = 1

        # 3. Sort scores descending
        sorted_idx = np.argsort(-scores)
        ranked_rel = rel[sorted_idx]

        # 4. Compute metrics
        p = precision_at_k(ranked_rel, k)
        r = recall_at_k(ranked_rel, k)
        nd = ndcg_at_k(ranked_rel, k)

        all_prec.append(p)
        all_rec.append(r)
        all_ndcg.append(nd)

    # Final aggregated metrics
    return {
        "precision@k": float(np.mean(all_prec)),
        "recall@k": float(np.mean(all_rec)),
        "ndcg@k": float(np.mean(all_ndcg)),
        "num_users": int(len(users)),
    }


def recommend_for_user(model, g, user_idx, movies_df, k=10, device="cpu"):
    model.eval()
    g = g.to(device)

    with torch.no_grad():
        user_emb, movie_emb = model.encode(g)
        user_emb = user_emb.to(device)
        movie_emb = movie_emb.to(device)

    num_movies = movie_emb.shape[0]

    # Score all movies
    u_tensor = torch.tensor([user_idx], device=device).repeat(num_movies)
    m_tensor = torch.arange(num_movies, device=device)

    with torch.no_grad():
        logits = model.link_predictor(user_emb, movie_emb, u_tensor, m_tensor)
        scores = torch.sigmoid(logits).cpu().numpy()

    # Rank descending
    ranked = np.argsort(-scores)[:k]

    # Attach titles
    recs = movies_df.set_index("movie_idx").loc[ranked][["title"]]
    recs["score"] = scores[ranked]

    return recs.reset_index()


def compare_user_preferences(user_idx, val_pos_path, movies_df, recs):
    """Return which recommended movies were actually liked by the user."""
    df = pd.read_csv(val_pos_path)

    # Movies user actually interacted with in validation set
    true_movies = df[df["user_idx"] == user_idx]["movie_idx"].tolist()

    recs["relevant"] = recs["movie_idx"].apply(lambda x: 1 if x in true_movies else 0)
    return recs

def show_recommendations_with_scores(user_id, model, g, movies_df, k=10, device="cpu"):
    """
    Display top-K recommendations WITH predicted scores.
    Clean demo: does NOT show relevance labels.
    """

    model.eval()
    g = g.to(device)

    # Encode users + movies once
    with torch.no_grad():
        user_emb, movie_emb = model.encode(g)

    num_movies = movie_emb.shape[0]

    # Build score tensors
    u_tensor = torch.tensor([user_id], dtype=torch.long, device=device).repeat(num_movies)
    m_tensor = torch.arange(num_movies, dtype=torch.long, device=device)

    # Predict scores
    with torch.no_grad():
        logits = model.link_predictor(
            user_emb.to(device), 
            movie_emb.to(device), 
            u_tensor, 
            m_tensor
        )
        scores = torch.sigmoid(logits).cpu().numpy()

    # Pick top-K
    topk_idx = scores.argsort()[-k:][::-1]  # descending

    # Build result table
    recs = movies_df.loc[topk_idx, ["title", "genres"]].copy()
    recs.insert(0, "rank", range(1, k + 1))
    recs["score"] = scores[topk_idx].round(4)    # rounded for readability

    return recs.reset_index(drop=True)



# ======================================================
# FUNCTIONS FOR BONUS SECTION
# ======================================================

def load_genome_tags(raw_dir="data/raw/"):
    """
    Load MovieLens genome tags and relevance scores.

    Returns:
        tags_df: tagId -> tag string
        scores_df: movieId, tagId, relevance
    """
    tags_df = pd.read_csv(os.path.join(raw_dir, "genome_tags.csv"))
    scores_df = pd.read_csv(os.path.join(raw_dir, "genome_scores.csv"))

    expected_cols_tags = {"tagId", "tag"}
    expected_cols_scores = {"movieId", "tagId", "relevance"}

    if not expected_cols_tags.issubset(tags_df.columns):
        raise ValueError("genome-tags.csv missing required columns")

    if not expected_cols_scores.issubset(scores_df.columns):
        raise ValueError("genome-scores.csv missing required columns")

    return tags_df, scores_df



def build_movie_tag_edges(movies_df, genome_scores_df, relevance_threshold=0.8):
    """
    Build movie–tag edges using only movies present in the processed dataset.
    We drop genome-score rows for movies NOT included in movies.csv.
    """

    # 1. Keep only tags for our sampled movies
    valid_movieIds = set(movies_df["movieId"])
    genome_scores_df = genome_scores_df[genome_scores_df["movieId"].isin(valid_movieIds)].copy()

    # 2. Build tag mapping
    tag_ids = sorted(genome_scores_df["tagId"].unique())
    tag2idx = {tid: i for i, tid in enumerate(tag_ids)}

    # 3. Map movieId → movie_idx
    movieId_to_idx = dict(zip(movies_df["movieId"], movies_df["movie_idx"]))
    genome_scores_df["movie_idx"] = genome_scores_df["movieId"].map(movieId_to_idx)

    # 4. Filter edges by relevance threshold
    filtered = genome_scores_df[genome_scores_df["relevance"] >= relevance_threshold].copy()

    # 5. Create final edge dataframe
    filtered["tag_idx"] = filtered["tagId"].map(tag2idx)
    movie_tag_edges = filtered[["movie_idx", "tag_idx"]].dropna().copy()

    # Ensure correct types
    movie_tag_edges["movie_idx"] = movie_tag_edges["movie_idx"].astype(int)
    movie_tag_edges["tag_idx"] = movie_tag_edges["tag_idx"].astype(int)

    return movie_tag_edges, tag2idx




def build_hetero_graph(train_pos, movies_df, movie_tag_edges):
    """
    Build a heterogeneous graph with:
      • user
      • movie
      • tag
    Node types & relations:
      (user)  --rates-->   (movie)
      (movie) --rev_rates-> (user)
      (movie) --has_tag--> (tag)
      (tag)   --tag_of-->  (movie)
    """

    # --- Edge lists for user–movie ---
    src_users = torch.tensor(train_pos["user_idx"].values, dtype=torch.int64)
    dst_movies = torch.tensor(train_pos["movie_idx"].values, dtype=torch.int64)

    num_users = train_pos["user_idx"].max() + 1
    num_movies = movies_df["movie_idx"].max() + 1

    # --- Movie–Tag edges ---
    mt_src = torch.tensor(movie_tag_edges["movie_idx"].values, dtype=torch.int64)
    mt_dst = torch.tensor(movie_tag_edges["tag_idx"].values, dtype=torch.int64)

    num_tags = movie_tag_edges["tag_idx"].max() + 1

    # --- Build heterograph ---
    g = dgl.heterograph(
        {
            ("user", "rates", "movie"): (src_users, dst_movies),
            ("movie", "rev_rates", "user"): (dst_movies, src_users),

            ("movie", "has_tag", "tag"): (mt_src, mt_dst),
            ("tag", "tag_of", "movie"): (mt_dst, mt_src),
        },
        num_nodes_dict={
            "user": num_users,
            "movie": num_movies,
            "tag": num_tags,
        }
    )

    # --- Add movie features ---
    genre_cols = [c for c in movies_df.columns if c.startswith("genre_")]
    g.nodes["movie"].data["genre"] = torch.tensor(
        movies_df.sort_values("movie_idx")[genre_cols].values,
        dtype=torch.float32
    )

    print("\nHeterogeneous Graph Summary:")
    print(g)

    return g


