"""
DGL.API.v2.py — Native API + Thin Wrapper Demonstration (MSML610 / MovieLens)

This script demonstrates the *native* DGL pieces we rely on (heterographs,
edge data, GraphSAGE) and the *thin wrapper* functions implemented in
`dgl_utils.py`. It is intentionally runnable on CPU, with configurable
subsampling for large datasets.

What it does:
  1) Loads MovieLens ratings (and movies.csv if available) or a tiny toy sample.
  2) Optionally subsamples ratings (max-edges) and filters movies to match.
  3) Builds a bipartite heterograph with edge weights (ratings).
  4) Adds movie-genre one-hot features.
  5) Splits edges into train/val/test using a temporal split on timestamps.
  6) Trains a small GraphSAGE link-prediction model for a configurable
     number of epochs (didactic CPU-friendly setup).
  7) Evaluates Precision@K / Recall@K and rating RMSE via a ridge regressor.
  8) Produces Top-N recommendations for a sample user.

Usage (CPU):
  python DGL.API.v2.py --ratings data/raw/rating.csv --movies data/raw/movie.csv \
                       --max-edges 200000 --epochs 40 --embed-dim 64 --k 10

If --ratings/--movies are omitted or files are missing, a tiny toy dataset
is used so the script can always run to completion.

"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import dgl  

import dgl_utils as du


# -----------------------------------------------------------------------------
# Args & logging
# -----------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DGL API tutorial runner")
    p.add_argument("--ratings", type=str, default="", help="Path to rating.csv")
    p.add_argument("--movies", type=str, default="", help="Path to movie.csv (optional)")
    p.add_argument("--max-edges", type=int, default=200_000, help="Max edges to sample from ratings for speed")
    p.add_argument("--epochs", type=int, default=40, help="Epochs for the didactic link-pred training")
    p.add_argument("--embed-dim", type=int, default=64, help="Embedding dimension")
    p.add_argument("--k", type=int, default=10, help="K for Precision@K / Recall@K and Top-N")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device (API demo works on CPU)")
    p.add_argument("--sample-user", type=int, default=0, help="User index (contiguous id) to show recs for")
    return p.parse_args()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _init_logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger("DGL.API")


# -----------------------------------------------------------------------------
# Data loading (real or toy)
# -----------------------------------------------------------------------------

def _load_or_toy(ratings_path: str, movies_path: str, max_edges: int, seed: int) -> Dict[str, pd.DataFrame]:
    """
    Load ratings (+ movies) if present; otherwise return a tiny toy dataset.
    """
    have_real = ratings_path and os.path.exists(ratings_path)
    if have_real:
        ratings = pd.read_csv(ratings_path)
        movies = None
        if movies_path and os.path.exists(movies_path):
            movies = pd.read_csv(movies_path)
        
        if max_edges and len(ratings) > max_edges:
            ratings = ratings.sample(n=max_edges, random_state=seed).reset_index(drop=True)
            if movies is not None:
                used_movie_ids = ratings["movieId"].unique()
                movies = movies[movies["movieId"].isin(used_movie_ids)].reset_index(drop=True)
        
        out = {"ratings": ratings}
        if movies is not None:
            out["movies"] = movies
        return out

    movies, ratings = du.load_dummy_data()
    return {"ratings": ratings, "movies": movies}


# -----------------------------------------------------------------------------
# Main driver
# -----------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    _set_seed(args.seed)
    log = _init_logger()
    log.info("Starting DGL API demo (device=%s)", args.device)

    # 1) Load data (real or toy)
    data = _load_or_toy(args.ratings, args.movies, args.max_edges, args.seed)
    ratings_df = data["ratings"]
    movies_df = data.get("movies", None)
    log.info("Ratings rows=%d; Movies rows=%s", len(ratings_df), "n/a" if movies_df is None else len(movies_df))

    # 2) Build heterograph 
    g, id_maps = du.create_hetero_graph_from_pandas(ratings_df, movies_df)
    log.info("Graph built: users=%d movies=%d edges=%d | etypes=%s", 
             g.num_nodes("user"), g.num_nodes("movie"), 
             g.num_edges(("user", "rates", "movie")), g.etypes)

    # 3) Add edge features (ratings, timestamps)
    du.add_edge_features(g, ratings_df, id_maps)
    log.info("Edge features added")

    # 4) Add movie node features 
    movie_feat_tensor = None
    if movies_df is not None:
        du.add_movie_node_features(g, movies_df, id_maps)
        if 'feat' in g.nodes['movie'].data:
            movie_feat_tensor = g.nodes['movie'].data['feat']
            log.info("Movie features: shape=%s | #genres=%d", 
                     tuple(movie_feat_tensor.shape), movie_feat_tensor.shape[1])
        else:
            log.warning("Movie features not added (no genres column?)")

    # 5) Edge splits
    splits = du.make_temporal_edge_splits(
        g,
        etype=("user", "rates", "movie"),
        val_frac=0.1,
        test_frac=0.1,
        ts_key="ts",
    )
    train_pairs = du.eids_to_pairs(g, splits["train_eids"])
    val_pairs = du.eids_to_pairs(g, splits["val_eids"])
    test_pairs = du.eids_to_pairs(g, splits["test_eids"])

    # Keep ratings per split for RMSE
    r_all = g.edges[('user', 'rates', 'movie')].data["rating"].numpy()
    train_ratings = [r_all[i] for i in splits["train_eids"].tolist()]
    val_ratings = [r_all[i] for i in splits["val_eids"].tolist()]
    test_ratings = [r_all[i] for i in splits["test_eids"].tolist()]

    log.info("Splits: train=%d val=%d test=%d", len(train_pairs), len(val_pairs), len(test_pairs))

    # 6) Train GraphSAGE link prediction 
    embs = du.train_link_prediction(
        g, splits,
        embed_dim=args.embed_dim,     
        epochs=args.epochs,           
        lr=3e-4,
        device=args.device,
        movie_feat_tensor=movie_feat_tensor
    )
    user_emb, movie_emb = embs["user_emb"], embs["movie_emb"]
    log.info("Embeddings: users=%s movies=%s", tuple(user_emb.shape), tuple(movie_emb.shape))

    # 7) Evaluate P@K / R@K on test edges
    metrics_k = du.evaluate_precision_recall_at_k(user_emb, movie_emb, test_pairs, k=args.k)
    log.info("P@%d=%.4f | R@%d=%.4f", args.k, metrics_k["precision@k"], args.k, metrics_k["recall@k"])

    # 8) RMSE via ridge regressor on frozen embeddings (train→fit, test→score)
    reg = du.fit_edge_regressor_ridge(user_emb, movie_emb, train_pairs, train_ratings, alpha=1.0)
    rmse_test = du.rmse_from_regressor(reg, user_emb, movie_emb, test_pairs, test_ratings)
    log.info("Rating RMSE (test)=%.4f", rmse_test)

    # 9) Top-N for a sample user 
    seen_map = du.build_user_seen_map(
        g, 
        splits["train_eids"], 
        etype=("user", "rates", "movie")  
    )
    topn_idx = du.recommend_topk_for_user(
        args.sample_user, 
        user_emb, 
        movie_emb,
        k=args.k,
        seen_items=seen_map.get(args.sample_user, set()),
        maps=id_maps  
    )
    
    title_lookup = {}
    if movies_df is not None:
        title_lookup = du.id_maps_to_title_lookup(movies_df, id_maps['movie'])  
    
    recs = [(mid, title_lookup.get(mid, f"movie_{mid}")) for mid in topn_idx]

    # 10) Print compact summary for your write-up
    summary = {
        "users": int(user_emb.shape[0]),
        "movies": int(movie_emb.shape[0]),
        "edges": int(g.num_edges(("user", "rates", "movie"))),
        f"P@{args.k}": round(metrics_k["precision@k"], 4),
        f"R@{args.k}": round(metrics_k["recall@k"], 4),
        "RMSE": round(rmse_test, 4),
        "sample_user": args.sample_user,
        "topN_titles": [t for (_, t) in recs],
    }
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()