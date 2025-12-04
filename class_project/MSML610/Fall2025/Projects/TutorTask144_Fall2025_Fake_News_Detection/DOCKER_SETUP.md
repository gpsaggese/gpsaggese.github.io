# Docker Setup Guide - Fake News Detection Project

## Overview

This guide explains how to build and run the entire Fake News Detection project using Docker. The Docker setup provides:

- **Isolated Environment**: Complete Python environment with all dependencies
- **Easy Deployment**: Single command to execute the entire project
- **Reproducibility**: Guaranteed consistent results across machines
- **Scalability**: Easy to scale across multiple containers
- **Persistence**: Data, models, and logs stored in volumes

---

## Prerequisites

Before you begin, ensure you have:

1. **Docker** (version 20.10 or higher)
2. **Docker Compose** (version 1.29 or higher)
3. **Git** (for cloning the repository)

---

## Quick Start

### 1. Build the Docker Image

```bash
docker-compose build
```

### 2. Run the Project

```bash
docker-compose up
```

This executes the entire project including:
- Accuracy tests
- K-fold cross-validation
- BERT training pipeline
- LSTM training pipeline
- Enhanced training pipeline
- All model evaluations

---

## Available Commands

Run specific tasks with:

```bash
docker-compose run --rm fake-news-app [COMMAND]
```

**Commands:**
- `all` - Run all tests and training (default)
- `test` - Run accuracy tests only
- `bert` - Run BERT training only
- `lstm` - Run LSTM training only
- `cv` - Run k-fold cross-validation only
- `enhanced` - Run enhanced training pipeline
- `eval` - Run all evaluations
- `mcp` - Start MCP server
- `bash` - Interactive bash shell

---

## Volume Mounts

Results are persisted in:
- `./data/` - Datasets
- `./models/` - Trained models
- `./output/` - Results
- `./logs/` - Execution logs

---

## Port Mappings

- `8000` - API Server
- `5000` - MCP Server

---

## Docker Architecture

```
Dockerfile (Multi-stage build)
├── Stage 1: Base image (Python 3.11)
├── Stage 2: Python dependencies
└── Stage 3: Final application

docker-compose.yml
├── fake-news-app (main service)
└── jupyter (optional, use with --profile jupyter)

docker-entrypoint.sh
└── Orchestrates all project pipelines
```

---

## Common Usage

### Full Execution
```bash
docker-compose up
```

### Run in Background
```bash
docker-compose up -d
docker-compose logs -f
```

### Stop Running Container
```bash
docker-compose down
```

### Interactive Shell
```bash
docker-compose run --rm fake-news-app bash
```

### View Logs
```bash
docker-compose logs -f fake-news-app
tail -f logs/*.log
```

---

## Environment Variables

Customize execution with environment variables:

```bash
docker-compose run -e LOG_LEVEL=DEBUG fake-news-app test
```

---

## Troubleshooting

### Port Already in Use
```bash
# Use different port in docker-compose.yml
ports:
  - "8001:8000"
```

### Out of Memory
```bash
# Increase Docker memory limit
# Docker Desktop > Preferences > Resources > Memory
# Or modify docker-compose.yml resource limits
```

### NLTK Data Issues
```bash
docker-compose exec fake-news-app python3 -m nltk.downloader punkt punkt_tab
```

---

## Files Included

- `Dockerfile` - Multi-stage Docker image definition
- `docker-compose.yml` - Service orchestration
- `.dockerignore` - Files to exclude from build
- `docker-entrypoint.sh` - Startup script
- `DOCKER_SETUP.md` - This documentation

---

## Next Steps

1. Build: `docker-compose build`
2. Run: `docker-compose up`
3. Check logs: `tail -f logs/*.log`
4. View results: `cat accuracy_test_results.json`

Enjoy containerized execution!
