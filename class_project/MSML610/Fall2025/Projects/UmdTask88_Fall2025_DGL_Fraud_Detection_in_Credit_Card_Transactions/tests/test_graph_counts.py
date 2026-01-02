# tests/test_graph_counts.py
import os, torch, yaml
def test_graph_counts():
    cfg = yaml.safe_load(open("configs/default.yaml"))
    path = cfg["graph"]["save_path"]
    assert os.path.exists(path), "graph file missing; run `make graph` first"
    g = torch.load(path)
    assert g["transaction"].num_nodes > 0
    assert g["account"].num_nodes > 0
    assert g["transaction","owns","account"].edge_index.shape[1] == g["transaction"].num_nodes
