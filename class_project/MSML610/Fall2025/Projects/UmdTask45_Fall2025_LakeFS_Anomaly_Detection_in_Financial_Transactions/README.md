# 🛡️ LakeFS - Anomaly Detection in Financial Transactions
**Author:** Ojasvi Maddala
**UID:** 121129154
**Course:** MSML610 

---

## 📖 1. Project Overview
This project demonstrates a **Data-Centric MLOps pipeline** for detecting credit card fraud. Instead of a static script, this project uses **LakeFS** to provide Git-like version control for our data and models.

**The Core Feature:** An automated "Tournament" that trains 8 different algorithms (including XGBoost, Random Forest, and Neural Networks). For every model trained, the system:
1.  Creates a **new, isolated branch** in LakeFS (e.g., `exp-xgb`).
2.  Uploads the trained model metrics.
3.  Generates and uploads visualization artifacts (Confusion Matrix, ROC Curve) to that specific branch.

### 📊 Dataset Context
The project utilizes the **Credit Card Fraud Detection** dataset.
* **Content:** Transactions made by credit cards in September 2013 by European cardholders.
* **Privacy:** Features V1, V2, ... V28 are the principal components obtained with PCA (for confidentiality).
* **Imbalance:** The dataset is highly unbalanced; the positive class (frauds) accounts for only 0.172% of all transactions.
* **Strategy:** This project employs **SMOTE** (Synthetic Minority Over-sampling Technique) to handle this imbalance during training.

---

## 💻 2. System Prerequisites
Before running any commands, ensure your host machine is set up.

### **Operating System**
* **Windows Users:** Must use **WSL2** (Windows Subsystem for Linux) running Ubuntu 22.04 or 24.04.
* **Mac/Linux:** Supported natively.

### **Software Requirements**
1.  **Docker Desktop:** Installed and running.
    * *Windows Check:* Ensure "Use WSL 2 based engine" is enabled in Docker Settings.
2.  **Git:** Installed inside your terminal environment (`sudo apt install git`).

---

## 🏗️ 3. One-Time Initial Setup
**Perform these steps ONLY ONCE on a new machine to configure credentials and the environment.**

### Step 1: Configure Secure Access (SSH & PAT)
The class scripts require specific secure credentials to verify your identity.

1.  **Generate SSH Key:**
    ```bash
    ssh-keygen -t ed25519 -C "your_email@example.com" -f ~/.ssh/id_rsa.causify-ai.github
    ```
2.  **Create Personal Access Token (PAT):**
    * Generate a classic PAT on GitHub with `repo` scopes.
    * Save it locally:
    ```bash
    echo "YOUR_GITHUB_TOKEN_HERE" > ~/.ssh/github_pat.causify-ai.txt
    ```

### Step 2: Clone the Repository
Clone the class repository recursively to download the project code and all helper submodules.
```bash
git clone --recursive git@github.com:gpsaggese-org/umd_classes.git ~/src/umd_classes1
````

### Step 3: Build the Thin Client Environment

We must build the Python virtual environment that orchestrates the Docker containers.

1.  Navigate to the repository:
    ```bash
    cd ~/src/umd_classes1
    ```
2.  **Install Dependencies:** (If on a fresh Linux/WSL install)
    ```bash
    sudo apt update
    sudo apt install awscli python3-pip
    ```
3.  **Run the Build Script:**
    ```bash
    ./dev_scripts_umd_classes/thin_client/build.py
    ```
    *This script creates a virtual environment and installs necessary libraries.*

### Step 4: Verify Installation

1.  Activate the environment:
    ```bash
    source dev_scripts_umd_classes/thin_client/setenv.sh
    ```
2.  Test Docker connection:
    ```bash
    docker pull hello-world
    ```

-----

## ⚙️ 4. Daily Workflow (Running the Project)

**Do this every time you want to work on the project.**

This project requires **two separate terminal windows** running simultaneously.

### **Terminal 1: Start the LakeFS Server**

This terminal runs the data versioning engine (LakeFS) and the database.

1.  Navigate to the project folder:
    ```bash
    cd ~/src/umd_classes1/class_project/MSML610/Fall2025/Projects/UmdTask45_Fall2025_LakeFS_Anomaly_Detection_in_Financial_Transactions/
    ```
2.  Start the server:
    ```bash
    docker-compose -f docker-compose.lakefs.yaml up
    ```
    *Wait until you see "Server started". **Keep this terminal open.***

### **Terminal 2: Start the Jupyter Environment**

This terminal runs the coding environment where the notebooks live.

1.  Navigate to the tutorial folder containing the launch script:
    ```bash
    cd ~/src/umd_classes1/class_project/instructions/tutorial_template/tutorial_github_data605_style
    ```
2.  Launch the container:
    ```bash
    ./docker_jupyter.sh
    ```
3.  **Access Jupyter:**
      * The terminal will print a URL containing a token (e.g., `http://127.0.0.1:8888/?token=...`).
      * Copy and paste this URL into your browser.

-----

## 🔧 5. Configuration (First Time Only)

Since LakeFS runs in a fresh container, you must initialize it before running the notebooks.

### Step 1: Create Admin User

1.  Go to **[http://localhost:8000/setup](https://www.google.com/search?q=http://localhost:8000/setup)**.
2.  **Email:** Enter any email (e.g., `admin@test.com`).
3.  **Password:** Create a password.
4.  **Click "Setup".**
5.  **CRITICAL:** Copy the **Access Key ID** and **Secret Access Key**. You will need these for the notebook.

### Step 2: Create the Repository

1.  On the LakeFS dashboard, click **"Create Repository"**.
2.  **Repository Name:** `creditcard-fraud`
3.  **Storage Namespace:** `local://creditcard-fraud`
4.  Click **"Create Repository"**.

-----

## 🧪 6. Running the Experiments

### Step 1: Connect Notebook to LakeFS

1.  In Jupyter, open **`LakeFS_Fraud.example.ipynb`**.
2.  Find the credential cell (near the top).
3.  **Paste the Keys:** Replace `YOUR_ACCESS_KEY` and `YOUR_SECRET_KEY` with the keys you generated in Section 5.

### Step 2: Run the Pipeline

Run the notebook cells in order:

1.  **Ingestion:** Loads `creditcard.csv`, applies SMOTE, and commits the "Golden Dataset" to `main`.
2.  **Tournament:** Loops through 8 models (XGBoost, RF, etc.).
      * *Mechanism:* For each model, it creates a branch (e.g., `exp-xgb`), trains the model, and uploads artifacts.
3.  **Leaderboard:** Displays the final F1 scores comparing all models.

### Step 3: Verify Versioning

1.  Go to the **LakeFS UI** (`http://localhost:8000`).
2.  Click on the **`creditcard-fraud`** repo.
3.  **Change Branch:** Switch from `main` to `exp-xgb` (or any experiment branch).
4.  Navigate to `results/viz/` to see the isolated artifact images.

-----

## 🔄 7. Resuming Work (Important)

If you stop the server and want to come back later, you **do not** need to repeat Phase A or Section 5.

1.  **Open Terminal 1:** Run the LakeFS start command (`docker-compose ... up`).
2.  **Open Terminal 2:** Run the Jupyter script (`./docker_jupyter.sh`).
3.  **Log In:** Your previous Admin credentials and Repository data will still be there, provided you stopped the container gracefully.

-----

## 🛑 Cleanup

To stop the environment safely:

1.  In **Terminal 1** (LakeFS), press `Ctrl + C`.
2.  In **Terminal 2** (Jupyter), press `Ctrl + C`.
