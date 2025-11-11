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

import dgl
import dgl.nn as dglnn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MultiLabelBinarizer

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
