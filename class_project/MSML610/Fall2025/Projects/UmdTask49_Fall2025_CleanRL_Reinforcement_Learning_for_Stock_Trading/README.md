# CleanRL Reinforcement Learning for Stock Trading

This project implements an ensemble reinforcement learning strategy for stock trading, combining LSTM-based uncertainty forecasting, NLP-based news interpretation, and CleanRL agents (PPO/SAC).

## Project Structure

### Documentation

- **`CleanRL.example.md`**: The core documentation explaining the "Why" and "How" of the ensemble architecture.
- **`CleanRL.API.md`**: Details on the integration with CleanRL, including the custom environment and modified agent scripts.
- **`rl_utils.md`**: Breakdown of the data utility scripts in `rl_utils/`.

### Code

- **`CleanRL.example.ipynb`**: The main executable notebook. It walks through the entire pipeline: data fetching, model training, and backtesting.
- **`rl_env.py`**: Defines the `SignalTesterEnv`, a custom OpenAI Gym environment that the RL agents interact with.
- **`rl_utils/`**: A package containing helper scripts for data fetching (`data.py`, `data_handler.py`), news processing (`news_handler.py`), and technical indicators (`indicators.py`).
- **`CleanRL_API/`**: Contains the single-file implementations of PPO and SAC algorithms, adapted from [CleanRL](https://docs.cleanrl.dev/).

## How to Run

### Prerequisites

1.  **API Keys**: a `.env` file with **valid** Alpaca and Polygon.io API keys keys provided in the project root. (see `rl_utils.md` for details). (Note: for security puposes these keys will be invalidated after 30days ensure to regenerate your own keys for long term testing)

### Running the Strategy

The entire workflow is orchestrated through docker scripts from the thin build.

open terminal and navigate to with current folder as source, then run:

```
./docker_scripts/docker_build.sh
```

the above command should take 2-5 min and will build the docker contianer. to run jupyer and expore jupyter_server run:

```
./docker_scripts/docker_jupyter.sh
```

now you can navigate to http://localhost:8888/ to access the jupyter server.\
for the CleanRL.example.ipynb specifically open:
http://localhost:8888/notebooks/UmdTask49_Fall2025_CleanRL_Reinforcement_Learning_for_Stock_Trading/CleanRL.example.ipynb

run the notebook end to end to see the whole workflow. refer to CleanRL.example.md and CleanRL.API.md for in-depth workflow details

NOTE: if this docker error occurs:

```
docker: Error response from daemon: Conflict. The container name "/umd_msml610_image" is already in use by container "96921a576685f7153bb03af79142c1cab7fb1c259242d55a9bec987113244003". You have to remove (or rename) that container to be able to reuse that name.
```

please delete the umd_msml610_image in your docker and run docker build command again
