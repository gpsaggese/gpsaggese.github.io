.PHONY: setup clean

# Define virtual environment directory
VENV := venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

setup: $(VENV)
	$(PIP) install -r requirements.txt

$(VENV):
	python3 -m venv $(VENV)

clean:
	rm -rf $(VENV)
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete

help:
	@echo "Makefile commands:"
	@echo "  setup  - Create virtual environment and install dependencies"
	@echo "  clean  - Remove virtual environment and compiled files"
	@echo "  help   - Show this help message"