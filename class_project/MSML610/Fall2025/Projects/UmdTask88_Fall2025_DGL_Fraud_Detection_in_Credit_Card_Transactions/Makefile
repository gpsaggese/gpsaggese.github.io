# Makefile (top-level)
CONFIG := configs/default.yaml
PY := python

.PHONY: data features graph train_tabular all clean

data:
	$(PY) -m src.data.load_data --config $(CONFIG)

features: data
	$(PY) -m src.features.make_features --config $(CONFIG)

graph: features
	$(PY) -m src.graph.construct_hetero_graph --config $(CONFIG)

train_tabular: features
	$(PY) -m src.models.tabular_baselines --config $(CONFIG)

all: data features graph train_tabular

clean:
	rm -rf data/processed/*.parquet data/artifacts/*
