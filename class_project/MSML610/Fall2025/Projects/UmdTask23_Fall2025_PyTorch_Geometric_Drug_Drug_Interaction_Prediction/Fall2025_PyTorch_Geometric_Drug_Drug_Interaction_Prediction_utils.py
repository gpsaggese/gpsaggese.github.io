"""
Phase 1 utility stubs for the DDI project.
Actual implementations will be added in later phases.
"""

import torch
from torch import nn
from torch_geometric.data import Batch


class DrugGNN(nn.Module):
    """
    Placeholder GNN encoder.
    In Phase 2, this will use GCNConv or GATConv layers.
    """

    def __init__(self, in_channels=10, hidden_channels=32, out_channels=32):
        super().__init__()
        self.embed = nn.Linear(in_channels, out_channels)

    def forward(self, batch: Batch) -> torch.Tensor:
        x = batch.x  # node features
        # dummy pooling: mean over node features for each graph
        graph_emb = []
        for idx in range(batch.num_graphs):
            mask = (batch.batch == idx)
            graph_emb.append(self.embed(x[mask].mean(dim=0)))
        return torch.stack(graph_emb, dim=0)


class DDIInteractionModel(nn.Module):
    """
    Placeholder pairwise classifier.
    Combines two drug embeddings and predicts interaction probability.
    """

    def __init__(self, emb_dim=32, hidden_dim=32):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(2 * emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, emb1, emb2):
        pair_emb = torch.cat([emb1, emb2], dim=-1)
        return self.classifier(pair_emb)


def train_one_epoch(model, optimizer, batch_iter, device):
    """
    Training loop placeholder.
    """
    return {"loss": None, "num_batches": 0}


def evaluate_model(model, batch_iter, device):
    """
    Evaluation loop placeholder.
    """
    return {"roc_auc": None, "loss": None}
