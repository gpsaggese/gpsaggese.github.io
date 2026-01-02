import dataclasses
from dataclasses import dataclass
from typing import Dict, Optional

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import SAGEConv, HeteroGraphConv


# TODO: adjust if your transaction node type has a different name in the graph
TX_NODE_TYPE = "transaction"


@dataclass
class GNNConfig:
    input_dim: int
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    tx_node_type: str = TX_NODE_TYPE


class FraudHeteroGNN(nn.Module):
    """
    Heterogeneous GraphSAGE-style model for fraud detection on a DGL heterograph.

    - We assume we have node features only for the transaction node type.
    - Other node types get learnable embeddings (initialized randomly).
    - Message passing happens over all canonical edge types.
    - We output a single logit per transaction node (binary fraud / not fraud).
    """

    def __init__(self, graph: dgl.DGLHeteroGraph, cfg: GNNConfig):
        super().__init__()

        self.cfg = cfg
        self.tx_node_type = cfg.tx_node_type
        self.hidden_dim = cfg.hidden_dim
        self.input_dim = cfg.input_dim
        self.num_layers = cfg.num_layers
        self.dropout = cfg.dropout

        if self.tx_node_type not in graph.ntypes:
            raise ValueError(
                f"Transaction node type '{self.tx_node_type}' not found in graph.ntypes={graph.ntypes}"
            )

        # Embeddings for non-transaction node types
        self.embeddings = nn.ModuleDict()
        for ntype in graph.ntypes:
            if ntype == self.tx_node_type:
                continue
            num_nodes = graph.num_nodes(ntype)
            # We give them same dimensionality as transaction input features
            self.embeddings[ntype] = nn.Embedding(num_nodes, self.input_dim)

        # Build heterogeneous GraphSAGE convs per relation
        self.convs = nn.ModuleList()
        for layer_idx in range(self.num_layers):
            in_dim = self.input_dim if layer_idx == 0 else self.hidden_dim
            conv_dict = {}
            for canonical_etype in graph.canonical_etypes:
                src_ntype, _, dst_ntype = canonical_etype
                conv_dict[canonical_etype] = SAGEConv(
                    in_feats=in_dim,
                    out_feats=self.hidden_dim,
                    aggregator_type="mean",
                )
            self.convs.append(HeteroGraphConv(conv_dict, aggregate="sum"))

        # MLP head for transaction node logits
        self.tx_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, graph: dgl.DGLHeteroGraph, tx_feats: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        graph : dgl.DGLHeteroGraph
            The heterograph used for message passing.
        tx_feats : Tensor [num_tx_nodes, input_dim]
            Features for transaction nodes in canonical node ID order.

        Returns
        -------
        logits : Tensor [num_tx_nodes]
            Raw logits for fraud classification (use BCEWithLogitsLoss).
        """
        # Initialize node features for all node types
        h: Dict[str, torch.Tensor] = {}
        for ntype in graph.ntypes:
            if ntype == self.tx_node_type:
                h[ntype] = tx_feats
            else:
                h[ntype] = self.embeddings[ntype].weight

        # Heterogeneous GraphSAGE layers
        for conv in self.convs:
            h = conv(graph, h)
            for ntype in h:
                h[ntype] = F.relu(h[ntype])
                h[ntype] = F.dropout(h[ntype], p=self.dropout, training=self.training)

        tx_h = h[self.tx_node_type]  # [num_tx_nodes, hidden_dim]
        logits = self.tx_head(tx_h).squeeze(-1)  # [num_tx_nodes]
        return logits


def build_model(graph: dgl.DGLHeteroGraph, input_dim: int, cfg_overrides: Optional[Dict] = None) -> FraudHeteroGNN:
    """
    Helper to build a model with a GNNConfig.

    Parameters
    ----------
    graph : dgl.DGLHeteroGraph
    input_dim : int
        Dimensionality of transaction node features.
    cfg_overrides : dict, optional
        Any config fields to override (e.g., {"hidden_dim": 256}).

    Returns
    -------
    model : FraudHeteroGNN
    """
    cfg = GNNConfig(input_dim=input_dim)
    if cfg_overrides:
        for k, v in cfg_overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    return FraudHeteroGNN(graph, cfg)
