# Running Fake News Detection Project with Docker

## Overview

This document provides complete instructions for building and running the entire Fake News Detection project using Docker. With Docker, you can execute the complete pipeline with a single command.

---

## Quick Start (3 Commands)

```bash
docker-compose build
docker-compose up
tail -f logs/*.log
```

---

## Prerequisites

### System Requirements

**Minimum:**
- Docker 20.10+
- Docker Compose 1.29+
- 4 GB RAM
- 20 GB disk space

**Recommended:**
- Docker 24.0+
- 8 GB RAM
- 4+ CPU cores
- 50 GB disk space

### Installation

macOS:
```bash
brew install docker docker-compose
```

Linux:
```bash
sudo apt-get install docker.io docker-compose
```

Windows:
Download Docker Desktop from https://www.docker.com/products/docker-desktop

---

## Step-by-Step Execution

### Step 1: Build Docker Image

```bash
docker-compose build
```

Time: 5-10 minutes (first time)

### Step 2: Run Project

```bash
docker-compose up
```

Automatically executes:
- Accuracy Tests (~2-5 min)
- Cross-Validation (~30-60 min)
- BERT Training (~45-90 min)
- LSTM Training (~45-90 min)
- Enhanced Training (~60-120 min)
- Model Evaluations (~15-30 min)

Total: 2-4 hours

### Step 3: View Results

In another terminal:
```bash
docker-compose logs -f
tail -f logs/*.log
cat accuracy_test_results.json
ls -la models/ output/
```

---

## Execution Modes

Run everything:
```bash
docker-compose up
```

Run specific task:
```bash
docker-compose run --rm fake-news-app test       # Accuracy tests
docker-compose run --rm fake-news-app bert       # BERT training
docker-compose run --rm fake-news-app lstm       # LSTM training
docker-compose run --rm fake-news-app cv         # Cross-validation
docker-compose run --rm fake-news-app enhanced   # Enhanced training
docker-compose run --rm fake-news-app eval       # Evaluations
docker-compose run --rm fake-news-app bash       # Interactive shell
```

---

## Data Persistence

Results are saved in host directories:
- ./data/ - Input datasets
- ./models/ - Trained models
- ./output/ - Results
- ./logs/ - Logs

Data survives container restart!

---

## Port Mappings

- 8000: API Server
- 5000: MCP Server
- 8888: Jupyter Lab (optional)

---

## Common Commands

```bash
docker-compose build           # Build image
docker-compose up              # Run everything
docker-compose up -d           # Run in background
docker-compose logs -f         # View logs
docker-compose down            # Stop container
docker-compose ps              # Check status
docker-compose exec ... bash   # Execute command
docker stats fake-news-detection  # Resource usage
```

---

## Troubleshooting

### Cannot connect to Docker daemon
Start Docker:
- macOS: open /Applications/Docker.app
- Linux: sudo systemctl start docker

### Port already in use
Edit docker-compose.yml, change ports: ["8001:8000"]

### Out of memory
Increase Docker memory (Docker Desktop > Preferences > Resources)

### Build is slow
Use BuildKit: DOCKER_BUILDKIT=1 docker-compose build

---

## Docker Files Included

1. **Dockerfile** (1.4 KB)
   - Multi-stage build
   - Python 3.11 slim
   - All dependencies

2. **docker-compose.yml** (2.1 KB)
   - Service orchestration
   - Volume mounts
   - Health checks

3. **.dockerignore** (931 B)
   - Build optimization

4. **docker-entrypoint.sh** (7.4 KB)
   - Startup script
   - 9 execution modes

5. **Documentation** (26 KB)
   - DOCKER_SETUP.md
   - DOCKER_QUICK_START.md
   - DOCKER_SUMMARY.md
   - RUN_IN_DOCKER.txt

---

## Features

✅ Single command execution
✅ Complete environment (all dependencies pre-installed)
✅ Data persistence (results survive restart)
✅ Flexible execution modes
✅ Comprehensive logging
✅ Production ready
✅ Easy deployment

---

## Next Steps

1. Ensure Docker is running
2. Run: docker-compose build
3. Run: docker-compose up
4. Monitor: docker-compose logs -f
5. View results: cat accuracy_test_results.json

---

## Support

For detailed instructions:
- DOCKER_QUICK_START.md - Quick reference
- DOCKER_SETUP.md - Comprehensive guide
- DOCKER_SUMMARY.md - Implementation details
- RUN_IN_DOCKER.txt - Step-by-step guide

---

Your entire fake news detection project is ready to run!

**Start with:** docker-compose build && docker-compose up
