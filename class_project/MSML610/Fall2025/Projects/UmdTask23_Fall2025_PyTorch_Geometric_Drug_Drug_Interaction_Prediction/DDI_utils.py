# Drug_utils.py
# Helpers for Drug–Drug Interaction (DDI) prediction using PyTorch Geometric (PyG).
# I wrote this as a small “utility layer” so my notebook stays readable:
# the notebook calls these functions/classes instead of repeating boilerplate.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_mean_pool

try:
    from rdkit import Chem
except Exception:
    Chem = None


@dataclass
class TrainConfig:
    """
    Basic training configuration.
    I keep it minimal so it's easy to tweak learning rate, epochs, and device in one place.
    """
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 10
    batch_size: int = 64
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class DrugGATEncoder(nn.Module):
    """
    Encodes a single molecular graph into a fixed-size embedding.

    Input:
      - x: node features, shape [num_nodes, in_dim]
      - edge_index: COO edges, shape [2, num_edges]
      - batch (optional): maps each node -> graph id, shape [num_nodes]

    Output:
      - graph embedding, shape [num_graphs_in_batch, out_dim]
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, heads: int = 2):
        super().__init__()

        # First GAT layer expands to hidden_dim * heads when concat=True
        self.gat1 = GATConv(in_dim, hidden_dim, heads=heads, concat=True)

        # Second GAT layer produces the final out_dim embedding per node
        self.gat2 = GATConv(hidden_dim * heads, out_dim, heads=1, concat=False)

        self.act = nn.ELU()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # If batch is missing, we treat it as a single-graph input.
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        h = self.act(self.gat1(x, edge_index))
        h = self.gat2(h, edge_index)

        # Pool node embeddings into a graph embedding
        g = global_mean_pool(h, batch)
        return g


class DrugInteractionModel(nn.Module):
    """
    Pair model for DDI prediction.

    It encodes Drug A and Drug B using the same encoder, concatenates embeddings,
    and predicts a single interaction logit (for BCEWithLogitsLoss).
    """

    def __init__(self, encoder: nn.Module, emb_dim: int):
        super().__init__()
        self.encoder = encoder

        # Simple MLP head: [zA, zB] -> logit
        self.mlp = nn.Sequential(
            nn.Linear(2 * emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, 1),
        )

    def forward(self, drugA: Data, drugB: Data) -> torch.Tensor:
        # drugA and drugB are PyG Data/Batch objects
        zA = self.encoder(drugA.x, drugA.edge_index, getattr(drugA, "batch", None))
        zB = self.encoder(drugB.x, drugB.edge_index, getattr(drugB, "batch", None))

        z = torch.cat([zA, zB], dim=-1)          # shape [num_graphs, 2*emb_dim]
        logits = self.mlp(z).squeeze(-1)         # shape [num_graphs]
        return logits


def train_one_epoch(model: nn.Module, loader, optimizer, device: str) -> float:
    """
    Trains for one epoch.

    Expected batch format:
      - batch.drugA: PyG Batch for drug A graphs
      - batch.drugB: PyG Batch for drug B graphs
      - batch.y: tensor of labels, shape [batch_size] with values 0/1

    Returns:
      - average loss for the epoch
    """
    model.train()
    loss_fn = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        optimizer.zero_grad()

        drugA = batch.drugA.to(device)
        drugB = batch.drugB.to(device)
        y = batch.y.float().to(device)

        logits = model(drugA, drugB)
        loss = loss_fn(logits, y)

        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        n_batches += 1

    return total_loss / max(1, n_batches)


@torch.no_grad()
def predict_proba(model: nn.Module, loader, device: str):
    """
    Runs inference and returns:
      - y_true: concatenated labels
      - y_prob: concatenated predicted probabilities
    """
    model.eval()

    all_y = []
    all_p = []

    for batch in loader:
        drugA = batch.drugA.to(device)
        drugB = batch.drugB.to(device)

        logits = model(drugA, drugB)
        probs = torch.sigmoid(logits).detach().cpu()

        all_y.append(batch.y.detach().cpu())
        all_p.append(probs)

    y_true = torch.cat(all_y) if len(all_y) else torch.tensor([])
    y_prob = torch.cat(all_p) if len(all_p) else torch.tensor([])

    return y_true, y_prob


def smiles_to_pyg_graph(smiles: str) -> Data:
    """
    Converts a SMILES string into a PyTorch Geometric graph.

    This is intentionally a simple baseline graph builder:
    node features are:
      [atomic_num, degree, formal_charge, is_aromatic]

    Edges are undirected (each bond is added in both directions).
    """
    if Chem is None:
        raise ImportError("RDKit not found. Install with: conda install -c conda-forge rdkit -y")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    x_list = []
    for atom in mol.GetAtoms():
        x_list.append([
            float(atom.GetAtomicNum()),
            float(atom.GetDegree()),
            float(atom.GetFormalCharge()),
            1.0 if atom.GetIsAromatic() else 0.0,
        ])
    x = torch.tensor(x_list, dtype=torch.float)

    edges = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edges.append([i, j])
        edges.append([j, i])

    if len(edges) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index)


if __name__ == "__main__":
    print("Drug_utils.py loaded successfully.")
    print("torch version:", torch.__version__)
