# Docker Quick Start Guide

## TL;DR - Three Commands to Run Everything

```bash
# 1. Build the Docker image
docker-compose build

# 2. Run the entire project
docker-compose up

# 3. View results (in another terminal)
tail -f logs/*.log
cat accuracy_test_results.json
```

---

## What Gets Executed?

When you run `docker-compose up`, the following happens automatically:

1. **Accuracy Tests** - Validates all models work correctly
2. **Cross-Validation** - K-fold CV for model evaluation
3. **BERT Training** - Trains BERT models on LIAR dataset
4. **LSTM Training** - Trains LSTM models
5. **Enhanced Training** - Runs advanced preprocessing + data augmentation
6. **All Evaluations** - Comprehensive model comparison

**Expected Duration:** 2-4 hours (depends on dataset)

---

## Common Commands

### Start Everything
```bash
docker-compose up
```

### Run in Background
```bash
docker-compose up -d
docker-compose logs -f  # Monitor logs
```

### Stop Everything
```bash
docker-compose down
```

### Run Specific Task
```bash
# Accuracy tests only
docker-compose run --rm fake-news-app test

# BERT training only
docker-compose run --rm fake-news-app bert

# LSTM training only
docker-compose run --rm fake-news-app lstm

# Enhanced training
docker-compose run --rm fake-news-app enhanced

# Interactive shell
docker-compose run --rm fake-news-app bash
```

### View Results
```bash
# Test results
cat accuracy_test_results.json

# Recent logs
tail -100 logs/*.log

# All available logs
ls -la logs/
```

---

## Accessing Services

### API Server (Port 8000)
```bash
curl http://localhost:8000
```

### MCP Server (Port 5000)
Start MCP server:
```bash
docker-compose run --rm fake-news-app mcp
```

Then use Python client:
```python
from MCP.client import FakeNewsMCPClient
import asyncio

async def main():
    async with FakeNewsMCPClient() as client:
        result = await client.predict("News text here...")
        print(result)

asyncio.run(main())
```

---

## Docker File Structure

```
Dockerfile              ← Multi-stage build configuration
docker-compose.yml      ← Service orchestration
.dockerignore          ← Exclude files from build
docker-entrypoint.sh   ← Startup script
DOCKER_SETUP.md        ← Full documentation
DOCKER_QUICK_START.md  ← This file
```

---

## Troubleshooting

### Problem: "Cannot connect to Docker daemon"
```bash
# Start Docker
sudo systemctl start docker  # Linux
open /Applications/Docker.app  # macOS
```

### Problem: "Port 8000 already in use"
```bash
# Edit docker-compose.yml
# Change: ports: ["8000:8000"]
# To:     ports: ["8001:8000"]
```

### Problem: "Out of memory"
```bash
# Docker Desktop > Preferences > Resources > Memory (increase to 4GB+)
# Or check available system memory
docker stats
```

### Problem: Building is slow
```bash
# Use Docker BuildKit for faster builds
DOCKER_BUILDKIT=1 docker-compose build
```

---

## Docker Image Details

- **Base Image:** Python 3.11 slim
- **Dependencies:** All from requirements.txt
- **Size:** ~2-3 GB
- **Build Time:** 5-10 minutes (first time)
- **Runtime:** 2-4 hours (full execution)

---

## Environment Variables

Customize execution:
```bash
docker-compose run -e LOG_LEVEL=DEBUG fake-news-app test
docker-compose run -e BATCH_SIZE=64 fake-news-app enhanced
```

---

## Next Steps

1. **Build image:** `docker-compose build`
2. **Run project:** `docker-compose up`
3. **Monitor progress:** Watch the logs in real-time
4. **Check results:** `cat accuracy_test_results.json`

That's it! Your entire fake news detection project will run automatically inside Docker.

---

## Support

For more details, see:
- `DOCKER_SETUP.md` - Comprehensive Docker guide
- `README.md` - Main project documentation
- `MCP_ARCHITECTURE.md` - MCP system details
