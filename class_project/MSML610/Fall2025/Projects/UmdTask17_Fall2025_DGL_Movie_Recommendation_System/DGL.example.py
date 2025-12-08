"""
DGL.example.py — Full End-to-End Graph-Based Movie Recommendation System (MSML610)

This script demonstrates a complete graph neural network-based recommendation system
using DGL. It implements the full pipeline from raw MovieLens data to trained model
and evaluation, following the Example notebook workflow.

What it does:
  1) Loads raw MovieLens data (ratings.csv, movies.csv, genome_tags.csv and genome_scores.csv)
  2) Samples a subset of users for computational efficiency
  3) Preprocesses data (ID mapping, genre features)
  4) Creates temporal train/val/test splits with negative sampling
  5) Builds a DGL heterogeneous bipartite graph
  6) Defines and trains a GraphSAGE-based GNN model
  7) Evaluates using full-ranking metrics (Precision@K, Recall@K, NDCG@K)
  8) Generates personalized recommendations for sample users

Usage:
  python DGL.example.py --raw-dir data/raw/ --n-users 15000 --epochs 20 --k 10
  python DGL.example.py --raw-dir data/raw/ --n-users 15000 --epochs 20 --k 10 --include-tags

If --raw-dir is omitted or files are missing, the script will exit with an error
(as this is the full example, not a toy demo).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import dgl
from dgl.nn.pytorch import SAGEConv, HeteroGraphConv
from torch.utils.data import DataLoader, TensorDataset

import dgl_utils as du


# -----------------------------------------------------------------------------
# Model Definitions (from Example notebook)
# -----------------------------------------------------------------------------

class BipartiteGraphSAGEEncoder(nn.Module):
    """
    Encoder for a user–movie bipartite heterograph with bidirectional edges.
    Users: learnable embeddings
    Movies: projected from genre features
    """
    
    def __init__(
        self,
        num_users: int,
        movie_in_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.num_users = num_users
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        
        # User embedding table
        self.user_embedding = nn.Embedding(num_users, hidden_dim)
        
        # Project movie genre features
        self.movie_input_proj = nn.Linear(movie_in_dim, hidden_dim)
        
        # GraphSAGE layers
        self.convs = nn.ModuleList()
        self.bns_user = nn.ModuleList()
        self.bns_movie = nn.ModuleList()
        
        in_dim = hidden_dim
        for layer in range(num_layers):
            out_dim_layer = hidden_dim if layer < num_layers - 1 else out_dim
            
            conv = HeteroGraphConv(
                {
                    "rates": SAGEConv(
                        in_feats=in_dim,
                        out_feats=out_dim_layer,
                        aggregator_type="mean",
                    ),
                    "rev_rates": SAGEConv(
                        in_feats=in_dim,
                        out_feats=out_dim_layer,
                        aggregator_type="mean",
                    ),
                },
                aggregate="sum",
            )
            self.convs.append(conv)
            self.bns_user.append(nn.BatchNorm1d(out_dim_layer))
            self.bns_movie.append(nn.BatchNorm1d(out_dim_layer))
            in_dim = out_dim_layer
        
        self.activation = nn.LeakyReLU(0.1)
        self.dropout_layer = nn.Dropout(dropout)
        
        if hidden_dim != out_dim:
            self.user_out_proj = nn.Linear(hidden_dim, out_dim)
        else:
            self.user_out_proj = nn.Identity()
    
    def forward(self, g: dgl.DGLHeteroGraph):
        """Returns user_emb [num_users, out_dim], movie_emb [num_movies, out_dim]"""
        device = next(self.parameters()).device
        g = g.to(device)
        
        h_user = self.user_embedding.weight
        movie_genre = g.nodes["movie"].data["genre"].to(device)
        h_movie = self.movie_input_proj(movie_genre)
        
        for conv, bn_u, bn_m in zip(self.convs, self.bns_user, self.bns_movie):
            h_dict = {"user": h_user, "movie": h_movie}
            h_new = conv(g, h_dict)
            
            h_user_new = bn_u(h_new["user"])
            h_movie_new = bn_m(h_new["movie"])
            
            h_user_new = self.activation(h_user_new)
            h_movie_new = self.activation(h_movie_new)
            
            h_user_new = self.dropout_layer(h_user_new)
            h_movie_new = self.dropout_layer(h_movie_new)
            
            if h_user_new.shape == h_user.shape:
                h_user = h_user + h_user_new
            else:
                h_user = h_user_new
            
            if h_movie_new.shape == h_movie.shape:
                h_movie = h_movie + h_movie_new
            else:
                h_movie = h_movie_new
        
        h_user_out = self.user_out_proj(h_user)
        return h_user_out, h_movie


class MLPLinkPredictor(nn.Module):
    """Link prediction head operating on concatenated user & movie embeddings."""
    
    def __init__(
        self,
        emb_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * emb_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, user_emb, movie_emb, user_idx, movie_idx):
        u = user_emb[user_idx]
        m = movie_emb[movie_idx]
        x = torch.cat([u, m], dim=-1)
        logits = self.mlp(x).squeeze(-1)
        return logits


class GNNRecommender(nn.Module):
    """Full model: GraphSAGE encoder + link prediction head."""
    
    def __init__(
        self,
        num_users: int,
        movie_feat_dim: int,
        encoder_hidden_dim: int = 128,
        encoder_out_dim: int = 128,
        encoder_layers: int = 2,
        encoder_dropout: float = 0.2,
        lp_hidden_dim: int = 256,
        lp_dropout: float = 0.3,
    ):
        super().__init__()
        
        self.encoder = BipartiteGraphSAGEEncoder(
            num_users=num_users,
            movie_in_dim=movie_feat_dim,
            hidden_dim=encoder_hidden_dim,
            out_dim=encoder_out_dim,
            num_layers=encoder_layers,
            dropout=encoder_dropout,
        )
        
        self.link_predictor = MLPLinkPredictor(
            emb_dim=encoder_out_dim,
            hidden_dim=lp_hidden_dim,
            dropout=lp_dropout,
        )
    
    def encode(self, g: dgl.DGLHeteroGraph):
        """Returns user_emb [num_users, d], movie_emb [num_movies, d]"""
        return self.encoder(g)
    
    def forward(self, g, user_idx, movie_idx):
        """Returns logits (before sigmoid) for BCEWithLogitsLoss"""
        user_emb, movie_emb = self.encode(g)
        logits = self.link_predictor(user_emb, movie_emb, user_idx, movie_idx)
        return logits

class HeteroGraphSAGEEncoder(nn.Module):
    """
    Encoder for a 3-type heterograph:
      user, movie, tag
    with 4 edge types:
      user --rates--> movie
      movie --rev_rates--> user
      movie --has_tag--> tag
      tag   --tag_of--> movie
    """
    def __init__(self, num_users, num_movies, num_tags, movie_feat_dim,
                 hidden_dim=128, out_dim=128, num_layers=2, dropout=0.2):
        super().__init__()

        # Embeddings for user and tag nodes
        self.user_embedding = nn.Embedding(num_users, hidden_dim)
        self.tag_embedding = nn.Embedding(num_tags, hidden_dim)

        # Project movie feature vectors
        self.movie_proj = nn.Linear(movie_feat_dim, hidden_dim)

        # Create hetero GraphSAGE layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroGraphConv(
                {
                    "rates":    SAGEConv(hidden_dim, hidden_dim, "mean"),
                    "rev_rates": SAGEConv(hidden_dim, hidden_dim, "mean"),
                    "has_tag":  SAGEConv(hidden_dim, hidden_dim, "mean"),
                    "tag_of":   SAGEConv(hidden_dim, hidden_dim, "mean"),
                },
                aggregate="sum",
            )
            self.layers.append(conv)

        self.dropout = nn.Dropout(dropout)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, g):
        h = {
            "user": self.user_embedding.weight,
            "movie": self.movie_proj(g.nodes["movie"].data["genre"]),
            "tag": self.tag_embedding.weight,
        }

        for conv in self.layers:
            h_new = conv(g, h)
            h = {ntype: self.act(self.dropout(h_new[ntype])) for ntype in h_new}

        return h["user"], h["movie"]


class HeterogeneousGNNRecommender(nn.Module):
    """
    Recommender for heterogeneous graph with tags.
    Uses HeteroGraphSAGEEncoder + MLP link predictor.
    """
    def __init__(
        self,
        num_users: int,
        num_movies: int,
        num_tags: int,
        movie_feat_dim: int,
        encoder_hidden_dim: int = 128,
        encoder_out_dim: int = 128,
        encoder_layers: int = 2,
        encoder_dropout: float = 0.2,
        lp_hidden_dim: int = 256,
        lp_dropout: float = 0.3,
    ):
        super().__init__()
        
        self.encoder = HeteroGraphSAGEEncoder(
            num_users=num_users,
            num_movies=num_movies,
            num_tags=num_tags,
            movie_feat_dim=movie_feat_dim,
            hidden_dim=encoder_hidden_dim,
            out_dim=encoder_out_dim,
            num_layers=encoder_layers,
            dropout=encoder_dropout,
        )

        self.link_predictor = MLPLinkPredictor(
            emb_dim=encoder_out_dim,
            hidden_dim=lp_hidden_dim,
            dropout=lp_dropout,
        )

    def encode(self, g):
        """Returns user_emb [num_users, d], movie_emb [num_movies, d]"""
        return self.encoder(g)

    def forward(self, g, user_idx, movie_idx):
        """Returns logits (before sigmoid) for BCEWithLogitsLoss"""
        user_emb, movie_emb = self.encode(g)
        logits = self.link_predictor(user_emb, movie_emb, user_idx, movie_idx)
        return logits
# -----------------------------------------------------------------------------
# Args & logging
# -----------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DGL Example: Full recommendation system")
    p.add_argument("--raw-dir", type=str, default="data/raw/", 
                   help="Directory containing rating.csv, movie.csv, genome_tags.csv, and genome_scores.csv")
    p.add_argument("--n-users", type=int, default=15000,
                   help="Number of users to sample")
    p.add_argument("--min-ratings-per-user", type=int, default=20,
                   help="Minimum ratings per user for sampling")
    p.add_argument("--epochs", type=int, default=20,
                   help="Training epochs")
    p.add_argument("--batch-size", type=int, default=2048,
                   help="Batch size for training")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Learning rate")
    p.add_argument("--k", type=int, default=10,
                   help="K for Precision@K / Recall@K / NDCG@K")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed")
    p.add_argument("--device", type=str, default="cpu",
                   choices=["cpu", "cuda"], help="Device")
    p.add_argument("--sample-user", type=int, default=12,
                   help="User index to show recommendations for")
    p.add_argument("--include-tags", action="store_true",
               help="Include tag nodes in graph (heterogeneous with 3 node types)")
    p.add_argument("--relevance-threshold", type=float, default=0.8,
               help="Minimum relevance score for movie-tag edges (default: 0.8)")
    return p.parse_args()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _init_logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger("DGL.Example")


# -----------------------------------------------------------------------------
# Main driver
# -----------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    _set_seed(args.seed)
    log = _init_logger()
    log.info("Starting DGL Example pipeline (device=%s)", args.device)
    
    # 1) Load raw MovieLens data
    log.info("Step 1: Loading raw MovieLens data from %s", args.raw_dir)
    ratings_raw, movies_raw = du.load_movielens_data(args.raw_dir)
    log.info("Loaded: %d ratings, %d movies", len(ratings_raw), len(movies_raw))
    
    # 2) Sample users
    log.info("Step 2: Sampling %d users (min_ratings=%d)", 
             args.n_users, args.min_ratings_per_user)
    ratings_sampled, movies_sampled = du.sample_users(
        ratings_raw,
        movies_raw,
        n_users=args.n_users,
        min_ratings_per_user=args.min_ratings_per_user,
        out_dir="data/processed/"
    )
    log.info("Sampled: %d users, %d movies, %d ratings",
             ratings_sampled['userId'].nunique(),
             movies_sampled['movieId'].nunique(),
             len(ratings_sampled))
    
    # 3) Preprocess
    log.info("Step 3: Preprocessing (ID mapping, genre features)")
    du.preprocess_and_save(ratings_sampled, movies_sampled, out_dir="data/processed/")
    log.info("Preprocessing complete")
    
    # 4) Make splits
    log.info("Step 4: Creating train/val/test splits with negative sampling")
    split_dict = du.make_splits(
        ratings_path="data/processed/ratings.csv",
        out_dir="data/processed/",
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2,
        num_neg_per_pos=1.0,
    )
    log.info("Splits: train=%d pos/%d neg, val=%d pos/%d neg, test=%d pos/%d neg",
             len(split_dict["train_pos"]), len(split_dict["train_neg"]),
             len(split_dict["val_pos"]), len(split_dict["val_neg"]),
             len(split_dict["test_pos"]), len(split_dict["test_neg"]))
    
    # 5) Build graph
    log.info("Step 5: Building DGL heterogeneous graph")
    if args.include_tags:
        # Load tags and build tag edges
        log.info("Step 5.1: Loading genome tags and building tag edges")
        tags_df, scores_df = du.load_genome_tags(args.raw_dir)
        movies_df = pd.read_csv("data/processed/movies.csv")
        movie_tag_edges, tagId_to_idx = du.build_movie_tag_edges(
            movies_df=movies_df,
            genome_scores_path=os.path.join(args.raw_dir, "genome_scores.csv"),
            relevance_threshold=args.relevance_threshold
        )
        log.info("Tag edges: %d movie-tag connections", len(movie_tag_edges))
        
        # Build graph with tags
        train_pos = pd.read_csv("data/processed/train_pos.csv")
        g = du.build_hetero_graph(
            train_pos=train_pos,
            movies_df=movies_df,
            movie_tag_edges=movie_tag_edges
        )
    else:
        g = du.build_graph(
            ratings_path="data/processed/train_pos.csv",
            movies_path="data/processed/movies.csv",
            out_path="data/graphs/train_graph.bin"
        )
        movie_tag_edges = None
    # 6) Define model
    log.info("Step 6: Initializing GNN model")
    device = torch.device(args.device)
    movie_feat_dim = g.nodes["movie"].data["genre"].shape[1]

    if args.include_tags:
        num_tags = g.num_nodes("tag")
        model = HeterogeneousGNNRecommender(
            num_users=g.num_nodes("user"),
            num_movies=g.num_nodes("movie"),
            num_tags=num_tags,
            movie_feat_dim=movie_feat_dim,
        ).to(device)
    else:
        model = GNNRecommender(
            num_users=g.num_nodes("user"),
            movie_feat_dim=movie_feat_dim,
        ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()
    
    # 7) Build dataloader
    log.info("Step 7: Building training dataloader")
    train_loader, total_samples = du.build_dataloader(
        "data/processed/train_pos.csv",
        "data/processed/train_neg.csv",
        batch_size=args.batch_size,
        shuffle=True
    )
    log.info("Training samples: %d", total_samples)
    
    # 8) Train
    log.info("Step 8: Training model (%d epochs)", args.epochs)
    epoch_losses = []
    for ep in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        for user_idx, movie_idx, labels in train_loader:
            user_idx = user_idx.to(device)
            movie_idx = movie_idx.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            logits = model(g, user_idx, movie_idx)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * len(labels)
        
        epoch_loss /= total_samples
        epoch_losses.append(epoch_loss)
        log.info("Epoch %d/%d — Loss = %.4f", ep, args.epochs, epoch_loss)
    
    # 9) Evaluate
    log.info("Step 9: Evaluating on validation set (full ranking)")
    metrics = du.evaluate_full_ranking(
        model=model,
        g=g,
        pos_path="data/processed/val_pos.csv",
        k=args.k,
        device=args.device
    )
    log.info("Validation metrics: P@%d=%.4f, R@%d=%.4f, NDCG@%d=%.4f",
             args.k, metrics["precision@k"],
             args.k, metrics["recall@k"],
             args.k, metrics["ndcg@k"])
    
    # 10) Generate recommendations
    log.info("Step 10: Generating recommendations for user %d", args.sample_user)
    movies_df = pd.read_csv("data/processed/movies.csv")
    recs = du.show_recommendations_with_scores(
        user_id=args.sample_user,
        model=model,
        g=g,
        movies_df=movies_df,
        k=args.k,
        device=args.device
    )
    
    # 11) Print summary
    summary = {
    "users": int(g.num_nodes("user")),
    "movies": int(g.num_nodes("movie")),
    "edges": int(g.num_edges(("user", "rates", "movie"))),
    "training_samples": total_samples,
    "epochs": args.epochs,
    f"P@{args.k}": round(metrics["precision@k"], 4),
    f"R@{args.k}": round(metrics["recall@k"], 4),
    f"NDCG@{args.k}": round(metrics["ndcg@k"], 4),
    "sample_user": args.sample_user,
    "topN_recommendations": recs["title"].tolist(),
    }
    if args.include_tags:
        summary["tags"] = int(g.num_nodes("tag"))
        summary["movie_tag_edges"] = int(g.num_edges(("movie", "has_tag", "tag")))
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\n=== Top-%d Recommendations for User %d ===" % (args.k, args.sample_user))
    print(recs.to_string(index=False))


if __name__ == "__main__":
    main()