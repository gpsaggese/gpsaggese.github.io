<!-- toc -->

- [Building a Chat Interface for Jupyter with LangChain](#building-a-chat-interface-for-jupyter-with-langchain)
  * [What you'll learn](#what-youll-learn)
  * [Project files](#project-files)
  * [Docker: build + run JupyterLab](#docker-build--run-jupyterlab)
  * [Running the notebook](#running-the-notebook)
  * [Provider configuration](#provider-configuration)

<!-- tocstop -->

# Building a Chat Interface for Jupyter with LangChain

This folder demonstrates how to build a chat interface for Jupyter notebooks using LangChain and ipywidgets, with an optional JupyterLab extension for production use.

## What you'll learn

- Building a basic chat UI with ipywidgets
- Integrating LangChain for AI-powered responses
- Managing conversation state and memory
- Making chat context-aware of notebook variables
- Handling async operations in Jupyter
- Building a production JupyterLab extension

## Project files

- `jupyter_chat_extension.ipynb`: complete tutorial notebook with runnable code
- `blog_jupyter_chat_langchain.md`: comprehensive guide covering both widget-based and extension approaches
- `jupyterlab_context_chat/`: full JupyterLab extension (sidebar chat with notebook context)
- `requirements.txt`: Python dependencies
- `Dockerfile`: installs deps for JupyterLab
- `docker-compose.yml`: starts JupyterLab with the repo mounted at `/app`
- `.env.example`: template for environment variables (copy to `.env`)

## Two approaches

This tutorial shows two ways to add chat to Jupyter:

### 1. In-Notebook Chat (ipywidgets)
Quick and easy - add chat directly in your notebook using `ipywidgets`. See `jupyter_chat_extension.ipynb`.

Pros:
- Simple to implement
- No extension installation needed  
- Works in any Jupyter environment

Cons:
- Chat lives inside notebook cells
- Limited UI customization
- Restarts with kernel

### 2. JupyterLab Extension (recommended for production)
A proper JupyterLab sidebar extension with backend endpoint. See `jupyterlab_context_chat/`.

Pros:
- Professional sidebar UI
- Persistent across notebooks
- Server-side LLM calls (API keys stay secure)
- Captures notebook context automatically
- Distributable as Python package

Cons:
- Requires extension installation
- More complex setup

For detailed comparison and full tutorial, see [blog_jupyter_chat_langchain.md](blog_jupyter_chat_langchain.md).

Extension setup: [jupyterlab_context_chat/SETUP.md](jupyterlab_context_chat/SETUP.md)

## Docker: build + run JupyterLab

From this directory:

```bash
cd tutorials/Jupyter_Extension_Langchain
cp .env.example .env
# Edit `.env` and set your API key.
docker compose up --build
```

Then open:

- `http://localhost:8888/lab`

Notes:
- The compose file disables the Jupyter token/password for convenience. Don't use this on an untrusted network.
- The repo is mounted into the container (`/app`), so edits on your host are reflected instantly.

## Running the notebook

Open `jupyter_chat_extension.ipynb` and run it top-to-bottom.

Notes:
- Running this notebook will call your configured LLM provider and may incur costs
- The notebook demonstrates progressively more advanced chat features
- Context-aware examples require creating variables in the notebook first

## Provider configuration

Set env vars in `.env` (loaded via `docker-compose.yml`) and restart the container. See `.env.example` for the supported variables.
