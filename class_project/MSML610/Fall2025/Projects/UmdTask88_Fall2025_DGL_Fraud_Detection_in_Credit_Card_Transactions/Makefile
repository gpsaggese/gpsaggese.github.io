PYTHON ?= python

CONFIG = configs/default.yaml

DATA_RAW_DIR = data/raw
DATA_PROCESSED_DIR = data/processed
ARTIFACTS_DIR = data/artifacts

.PHONY: data features graph train_tabular train_gnn test clean

data:
	$(PYTHON) -m src.data.load_data --config $(CONFIG) $(if $(MAX_ROWS),--max-rows $(MAX_ROWS),) $(if $(SAMPLE_FRAC),--sample-frac $(SAMPLE_FRAC),)

features:
	$(PYTHON) -m src.features.make_features --config $(CONFIG)

graph:
	$(PYTHON) -m src.graph.construct_hetero_graph --config $(CONFIG)

train_tabular:
	$(PYTHON) -m src.models.tabular_baselines --config $(CONFIG)

train_tabular_lgbm:
	$(PYTHON) -m src.models.tabular_baselines --config $(CONFIG) --model lgbm

train_gnn:
	$(PYTHON) -m src.train.train_gnn --config $(CONFIG) --epochs 5 --lr 0.001 --hidden-dim 128 --num-layers 2 --dropout 0.2

test:
	pytest -q

clean:
	rm -f $(ARTIFACTS_DIR)/*.json
	rm -f $(ARTIFACTS_DIR)/*.pt
