# Learn Autogen in 60 Minutes

## Summary
- This tutorial's goal is to show you in 60 minutes
  - The basic API of Autogen (an open source framework for building agentic AI
    systems)
  - Concrete examples of using Autogen to build agents that can debate and reason
    investment strategies based on market data and SEC filings

## Autogen in 30 seconds
- AutoGen is an open-source framework from Microsoft for building agentic AI
  systems made of multiple collaborating agents.
- Provides layered APIs for message-passing agents, tool usage, and
  human-in-the-loop workflows.
- Enables autonomous planning, reasoning, and execution of tasks such as coding,
  browsing, and data processing.
- Supports multiple languages (including Python), with extensible integrations
  for LLM providers using OpenAI-style clients.
- Designed to help developers quickly prototype and deploy applications ranging
  from simple assistants to coordinated teams of specialized AI agents.

## Official References
- [AutoGen A framework for building AI agents and applications](microsoft.github.io/autogen/)
- [GitHub repo](https://github.com/microsoft/autogen)

## Tutorial Content
- This tutorial includes all the code, notebooks, and Docker containers in
  `tutorials/Autogen/...`
- The file `tutorials/Autogen/README.md` contains instructions to 
- A Docker system to build and run the environment using our standardize
  approach
- `autogen.API.ipynb`: Tutorial notebook focusing on API configurations and
  basic agent setup.
- `autogen.example.ipynb`: Contains advanced example of how to use Autogen
  covering end-to-end agentic workflow.
- `autogen_utils.py`: Contains the utilility functions required by
  `autogen.example.ipynb`

# `autogen.example.ipynb`
- This notebook provides a practical, end-to-end example of using **AutoGen** to
  demonstrate a complete agentic workflow

## Part 1: Dynamic Market Debate & Live Data
- Fetches real-time stock data from Yahoo Finance.
- Bull and Bear strategist agents debate market trends.
- Selector agent dynamically decides which expert to call at each step.
- Generates stock charts and financial summaries.

## Part 2: SEC Filings & Quantitative RAG Analysis (Extension of Part 1)
- Pulls 10-K filings from SEC EDGAR and cleans them.
- Embeds documents into a **ChromaDB** vector database.
- Senior Quant Analyst agent queries the database to extract revenue splits,
  risk factors, and other insights.
- Quant Runtime agent executes Python code locally to transform raw tables into
  structured visualizations.

- Part 2 extends Part 1 by combining live market data with deep, structured
  analysis of SEC filings, showing how multiple agents collaborate, leverage
  private databases via RAG, and produce actionable insights in a single
  integrated workflow.

- This example shows **how multiple agents collaborate**, use live data, leverage
  private databases via RAG, and produce actionable insights in a single
  integrated workflow.
