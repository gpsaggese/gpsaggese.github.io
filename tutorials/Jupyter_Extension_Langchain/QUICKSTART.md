# Quick Start Guide

This tutorial shows you how to build a chat interface for Jupyter notebooks using LangChain.

## Files Created

- README.md - Full documentation and overview
- blog_jupyter_chat_langchain.md - Detailed blog post explaining the concepts
- jupyter_chat_extension.ipynb - Step-by-step tutorial notebook
- requirements.txt - Python dependencies
- Dockerfile - Container setup
- docker-compose.yml - Easy Docker deployment
- .env.example - Environment variables template

## Quick Start

### Option 1: Run locally

```bash
cd tutorials/Jupyter_Extension_Langchain
pip install -r requirements.txt
export OPENAI_API_KEY="your-key"
jupyter lab
# Open jupyter_chat_extension.ipynb
```

### Option 2: Docker

```bash
cd tutorials/Jupyter_Extension_Langchain
cp .env.example .env
# Edit .env and add your API key
docker compose up --build
# Navigate to http://localhost:8888
```

### Option 3: Just read

- Read [blog_jupyter_chat_langchain.md](blog_jupyter_chat_langchain.md) for the full story
- Check [README.md](README.md) for architecture and details

## What You'll Build

A simple chat interface inside Jupyter that:
- Uses ipywidgets for the UI
- Connects to LangChain for AI responses
- Can see and reference notebook variables
- Handles async operations properly
- Takes less than 100 lines of code

## Resources

- LangChain docs: https://python.langchain.com/
- ipywidgets docs: https://ipywidgets.readthedocs.io/

