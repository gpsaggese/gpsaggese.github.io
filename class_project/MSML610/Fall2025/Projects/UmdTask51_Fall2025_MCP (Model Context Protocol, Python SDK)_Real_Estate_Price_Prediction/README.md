# UmdTask51: Real Estate Price Prediction with MCP

## Project Objective

The goal of this project is to predict house prices using the "House Sales in King County, USA" dataset. We will use an XGBoost regression model and leverage the **Model Context Protocol (MCP)** to manage the model training lifecycle, track experiments, and log parameters during hyperparameter tuning.

## Dataset

* **Name:** House Sales in King County, USA
* **Source:** [https://www.kaggle.com/datasets/harlfoxem/housesalesprediction](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction)
* **Local Path:** Assumed to be in this directory as `kc_house_data.csv`.

## Core Technologies

* **Model:** XGBoost (`xgboost`)
* **Experiment Tracking:** Model Context Protocol (`mcp`)
* **Data Handling:** `pandas`, `numpy`, `scikit-learn`
* **Environment:** Docker, Jupyter

## File Structure

As per the course guidelines, the project separates logic from presentation:

* **`MCP.API.md / .ipynb`**: A technical tutorial on how to use the `mcp` library itself, with simple examples.
* **`MCP.example.md / .ipynb`**: The main notebook for our real estate project, showing the end-to-end story.
* **`MCP_utils.py`**: Contains all Python functions for model training and hyperparameter tuning that use `mcp`.
* **`Dockerfile`**: Defines the Docker environment to run the project.

## Project Setup and Execution Guide

### 1\. Initial Setup (Data & Files)

Before touching Docker, ensure your project directory has all the necessary components.

  * **Project Root Directory:** This is where your `Dockerfile`, Python files (`MCP_utils.py`), and Markdown/Notebook files (`MCP.API.ipynb`, `MCP.example.ipynb`) are located.
  * **Download the Dataset:**
      * Navigate to the dataset source: [https://www.kaggle.com/datasets/harlfoxem/housesalesprediction](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction)
      * Download the dataset file.
      * **Crucially, rename the file** and place it directly in your project root directory with the name:
        ```
        kc_house_data.csv
        ```
  * **Verify Files:** Your root directory should contain at least:
      * `Dockerfile`
      * `kc_house_data.csv`
      * `MCP.API.ipynb`
      * `MCP.example.ipynb`
      * `MCP_utils.py`

### 2\. Docker Environment

We will now build the project image and launch the JupyterLab container.

#### A. Build the Docker Image

This step compiles your environment, installs dependencies (like `pandas`, `xgboost`, and `mcp`), and copies your data and code into the image.

1.  Open your terminal and navigate to the project root directory.
2.  Run the build command:
    ```bash
    docker build -t mcp-real-est .
    ```
      * *The `-t mcp-real-est` tags the image, and the `.` specifies the current directory as the build context.*

#### B. Run the JupyterLab Container

This starts the container and maps the internal Jupyter port (`8888`) to a port on your local machine.

1.  In the same terminal, run the container:
    ```bash
    docker run -p 8888:8888 mcp-real-est
    ```
      * *The container will start, and the Jupyter server will launch, printing connection details and a URL to the terminal.*

### 3\. Execute the Notebooks

Follow these steps to access JupyterLab and run the project logic.

#### A. Access JupyterLab

1.  Copy the URL printed in your terminal (it will look something like `http://127.0.0.1:8888/lab?token=...`).
2.  Paste the URL into your web browser. You should now see the JupyterLab interface, with all your project files listed on the left.

#### B. Run the API Tutorial

1.  Click on **`MCP.API.ipynb`** to open it.
2.  Review the cells to understand how the **Model Context Protocol (MCP)** works.
3.  **Run each cell** sequentially to execute the API examples. This confirms your `mcp` setup is functional.

#### C. Run the Main Project

1.  Click on **`MCP.example.ipynb`** to open it. This notebook contains the end-to-end real estate price prediction project logic, including data loading, pre-processing, XGBoost model training, hyperparameter tuning, and logging experiments with MCP.
2.  **Run each cell** sequentially from top to bottom.
      * The notebook will use the `kc_house_data.csv` file you placed in the root.
      * The `MCP_utils.py` functions will be imported and executed.
      * You should observe the **MCP** logging key metrics and parameters for each experiment run.

-----

### Next Step: Stopping the Container

Once you are done, remember to stop and clean up the running container:

1.  Go back to your terminal where the container is running (if you didn't run it detached). Press **`Ctrl + C`** to stop the process.
2.  Alternatively, if you started it detached (with `-d` not used here, but common), you would use:
    ```bash
    docker stop <container_name_or_id>
    ```
