# Running TFX Pipeline in Docker via WSL2/Ubuntu

This guide helps you run the House Price TFX Pipeline using Docker in WSL2 with Ubuntu.

## Why WSL2 + Docker?

✅ Better compatibility with TFX and Linux-based tools
✅ Faster performance than Docker Desktop on Windows
✅ Native Linux environment for data science workflows
✅ Easier debugging and troubleshooting

---

## Prerequisites

### 1. Verify WSL2 Installation

```powershell
# In Windows PowerShell (as Administrator)
wsl --list --verbose
```

Expected output:
```
  NAME            STATE           VERSION
* Ubuntu          Running         2
```

If VERSION is 1, upgrade to WSL2:
```powershell
wsl --set-version Ubuntu 2
```

### 2. Install Docker in WSL2/Ubuntu

**Open Ubuntu terminal** and run:

```bash
# Update package list
sudo apt-get update

# Install prerequisites
sudo apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

# Add Docker's official GPG key
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# Set up stable repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Install docker-compose (standalone)
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

### 3. Configure Docker (No Sudo Required)

```bash
# Add your user to docker group
sudo usermod -aG docker $USER

# Start Docker service
sudo service docker start

# IMPORTANT: Log out and log back in to WSL for group changes to take effect
exit
```

Re-open Ubuntu terminal and verify:
```bash
docker --version
docker-compose --version
```

### 4. Enable Docker Auto-Start (Optional)

Add to `~/.bashrc`:
```bash
# Start Docker service if not running
if ! service docker status > /dev/null 2>&1; then
    sudo service docker start
fi
```

Then:
```bash
source ~/.bashrc
```

---

## Accessing Your Project in WSL2

### Method 1: Navigate to Windows Files (Recommended)

WSL2 can access Windows files via `/mnt/`:

```bash
# Navigate to your project
cd /mnt/c/Masters/Coursework/Fall\ 25/MSML610-AML/final_project/umd_classes/class_project/MSML610/Fall2025/Projects/UmdTask56_Fall2025_TFX_Predicting_House_Prices_with_Feature_Engineering
```

**Tip:** Create an alias for easy access:
```bash
# Add to ~/.bashrc
echo 'alias tfx-project="cd /mnt/c/Masters/Coursework/Fall\ 25/MSML610-AML/final_project/umd_classes/class_project/MSML610/Fall2025/Projects/UmdTask56_Fall2025_TFX_Predicting_House_Prices_with_Feature_Engineering"' >> ~/.bashrc
source ~/.bashrc

# Now just type:
tfx-project
```

### Method 2: Copy Project to WSL Home (Alternative)

```bash
# Copy project to WSL filesystem (faster performance)
cp -r /mnt/c/Masters/Coursework/Fall\ 25/MSML610-AML/final_project/umd_classes/class_project/MSML610/Fall2025/Projects/UmdTask56_Fall2025_TFX_Predicting_House_Prices_with_Feature_Engineering ~/tfx-project

cd ~/tfx-project
```

---

## Running the Pipeline in WSL2

### Quick Start (Recommended)

```bash
# Navigate to project
cd /mnt/c/Masters/Coursework/Fall\ 25/MSML610-AML/final_project/umd_classes/class_project/MSML610/Fall2025/Projects/UmdTask56_Fall2025_TFX_Predicting_House_Prices_with_Feature_Engineering

# Make script executable
chmod +x docker/run_pipeline.sh

# Run full pipeline
cd docker
./run_pipeline.sh full
```

### Step-by-Step

```bash
# Navigate to project root
cd /mnt/c/Masters/.../UmdTask56_Fall2025_TFX_Predicting_House_Prices_with_Feature_Engineering

# Make script executable
chmod +x docker/run_pipeline.sh

# Build Docker image
cd docker
./run_pipeline.sh build

# Start container
./run_pipeline.sh start

# Run pipeline
./run_pipeline.sh run

# Enter container (optional)
./run_pipeline.sh bash

# Stop container when done
./run_pipeline.sh stop
```

---

## Alternative: Direct Docker Commands

If you prefer manual control:

```bash
# Navigate to project root
cd /mnt/c/Masters/.../UmdTask56_Fall2025_TFX_Predicting_House_Prices_with_Feature_Engineering

# Build image
docker-compose -f docker/docker-compose.yml build

# Start container
docker-compose -f docker/docker-compose.yml up -d

# Check container is running
docker ps

# Run pipeline
docker exec house-price-tfx python scripts/api.py

# View logs
docker logs house-price-tfx

# Enter container
docker exec -it house-price-tfx bash

# Stop container
docker-compose -f docker/docker-compose.yml down
```

---

## Expected Output

```bash
$ ./run_pipeline.sh full

========================================================================
  House Price TFX Pipeline - Docker Runner
========================================================================

Project root: /mnt/c/Masters/.../UmdTask56_Fall2025_TFX_Predicting_House_Prices_with_Feature_Engineering

Building Docker image...
[+] Building 245.3s (14/14) FINISHED
 => [internal] load build definition from Dockerfile
 => => transferring dockerfile: 1.2kB
 ...
 => exporting to image
 => => exporting layers
 => => writing image sha256:...
 => => naming to docker.io/library/house-price-tfx:latest
[OK] Docker image built successfully!

Starting Docker container...
[+] Running 1/1
 ⠿ Container house-price-tfx  Started
[OK] Container started!

Running TFX pipeline inside container...
================================================================================
House Price Prediction TFX Pipeline
================================================================================
Pipeline Name: house_price_prediction_pipeline
Pipeline Root: /app/pipeline_outputs/house_price_prediction_pipeline
...

Creating TFX pipeline...
Components: CsvExampleGen, StatisticsGen, SchemaGen (Phase 2)

Running pipeline with LocalDagRunner...
This may take a few minutes...

WARNING:absl:... (TFX warnings are normal)
INFO:absl:Running CsvExampleGen...
INFO:absl:Running StatisticsGen...
INFO:absl:Running SchemaGen...

================================================================================
Pipeline execution completed successfully!
================================================================================

Outputs saved to: /app/pipeline_outputs/house_price_prediction_pipeline
Metadata saved to: /app/pipeline_outputs/metadata/house_price_prediction_pipeline/metadata.db

Next steps:
  - Check pipeline_outputs/ for generated artifacts
  - Review schema in pipeline_outputs/.../SchemaGen/...
  - Proceed to Phase 3 for feature engineering

========================================================================
```

---

## Verifying Pipeline Outputs

### From WSL Terminal

```bash
# List generated artifacts
ls -la pipeline_outputs/house_price_prediction_pipeline/

# Expected structure:
# CsvExampleGen/
# StatisticsGen/
# SchemaGen/
```

### Detailed Exploration

```bash
# View schema (generated by SchemaGen)
find pipeline_outputs -name "schema.pbtxt" -exec cat {} \;

# Check example files (TFRecords)
ls -lh pipeline_outputs/house_price_prediction_pipeline/CsvExampleGen/

# View statistics
ls -la pipeline_outputs/house_price_prediction_pipeline/StatisticsGen/
```

### Access from Windows

The outputs are also accessible from Windows at:
```
C:\Masters\Coursework\Fall 25\MSML610-AML\final_project\...\pipeline_outputs\
```

You can view them in VS Code or Windows Explorer!

---

## Working with Jupyter in WSL2

### Start Jupyter Lab

```bash
# Enter container
docker exec -it house-price-tfx bash

# Inside container, start Jupyter
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

### Access from Windows Browser

1. Copy the token from output:
   ```
   http://127.0.0.1:8888/lab?token=abc123...
   ```

2. Open in Windows browser:
   ```
   http://localhost:8888/lab?token=abc123...
   ```

It just works! WSL2 automatically forwards ports to Windows.

---

## Troubleshooting

### Issue: "Cannot connect to Docker daemon"

**Solution:**
```bash
# Start Docker service
sudo service docker start

# Verify Docker is running
docker ps
```

### Issue: Permission denied

**Solution:**
```bash
# Add user to docker group
sudo usermod -aG docker $USER

# Log out and back in
exit
# Then reopen Ubuntu terminal
```

### Issue: "docker-compose: command not found"

**Solution:**
```bash
# Use docker compose (with space) instead
docker compose version

# Or create alias
echo 'alias docker-compose="docker compose"' >> ~/.bashrc
source ~/.bashrc
```

### Issue: Slow file access from /mnt/c

**Symptom:** Build is very slow

**Solution:** Copy project to WSL filesystem:
```bash
# Copy to home directory
cp -r /mnt/c/Masters/.../UmdTask56_Fall2025_TFX_Predicting_House_Prices_with_Feature_Engineering ~/tfx-project
cd ~/tfx-project

# Then build
cd docker
./run_pipeline.sh full
```

### Issue: Out of memory

**Solution:** Increase WSL2 memory limit

Create/edit `%USERPROFILE%\.wslconfig` (in Windows):
```ini
[wsl2]
memory=8GB
processors=4
```

Then restart WSL:
```powershell
# In Windows PowerShell
wsl --shutdown
```

Reopen Ubuntu terminal.

---

## Performance Tips

### 1. Use WSL Filesystem for Better Performance

```bash
# Work directly in WSL filesystem
cp -r /mnt/c/.../UmdTask56_Fall2025_TFX_Predicting_House_Prices_with_Feature_Engineering ~/tfx-project
cd ~/tfx-project
```

### 2. Enable BuildKit for Faster Builds

```bash
# Add to ~/.bashrc
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1
```

### 3. Use Volume Caching

Already configured in docker-compose.yml!

---

## Development Workflow in WSL2

### Recommended Setup

1. **Edit code in Windows** (VS Code, PyCharm)
2. **Run/test in WSL2 Docker** (terminal commands)
3. **Files sync automatically** (via /mnt/)

### VS Code Integration

Install "Remote - WSL" extension:
1. Open VS Code
2. Install "Remote - WSL" extension
3. Press F1, type "WSL: Open Folder in WSL"
4. Navigate to project

Now you can:
- Edit files in VS Code (Windows)
- Files are in WSL filesystem
- Run terminal commands in WSL
- Everything is fast!

---

## Quick Reference Commands

```bash
# Navigate to project (create alias first)
tfx-project

# Build and run everything
cd docker && ./run_pipeline.sh full

# Check container status
docker ps

# View logs
docker logs -f house-price-tfx

# Enter container
docker exec -it house-price-tfx bash

# Run data exploration
docker exec house-price-tfx python scripts/phase2_data_exploration.py

# Stop everything
cd docker && ./run_pipeline.sh stop

# Clean up everything
docker system prune -a
```

---

## Next Steps

1. **Run the pipeline:**
   ```bash
   cd /mnt/c/Masters/.../UmdTask56_Fall2025_TFX_Predicting_House_Prices_with_Feature_Engineering/docker
   ./run_pipeline.sh full
   ```

2. **Verify outputs:**
   ```bash
   ls -la ../pipeline_outputs/house_price_prediction_pipeline/
   ```

3. **Explore in Jupyter:**
   ```bash
   docker exec -it house-price-tfx jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
   ```
   Open http://localhost:8888 in Windows browser

4. **Review schema:**
   ```bash
   find ../pipeline_outputs -name "schema.pbtxt" -exec cat {} \;
   ```

---

## Advantages of WSL2 + Docker

✅ **Native Linux environment** - Better for TFX/TensorFlow
✅ **Performance** - Faster than Docker Desktop on Windows
✅ **Compatibility** - All Linux tools work natively
✅ **Port forwarding** - Access from Windows browser automatically
✅ **File access** - Edit in Windows, run in Linux
✅ **Resource control** - Fine-tune memory/CPU via .wslconfig

---

**Ready to start?**

```bash
# Open Ubuntu terminal
# Navigate to project
cd /mnt/c/Masters/Coursework/Fall\ 25/MSML610-AML/final_project/umd_classes/class_project/MSML610/Fall2025/Projects/UmdTask56_Fall2025_TFX_Predicting_House_Prices_with_Feature_Engineering

# Make script executable
chmod +x docker/run_pipeline.sh

# Run the pipeline!
cd docker
./run_pipeline.sh full
```

Good luck! 🚀🐧
