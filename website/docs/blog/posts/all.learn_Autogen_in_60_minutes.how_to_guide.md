---
title: "Autogen in 60 Minutes"
authors:
  - PranavShashidhara
  - gpsaggese
date: 2026-02-21
description:
categories:
  - AI Research
  - Software Engineering
---

TL;DR: Learn how to build agentic AI systems using AutoGen in 60 minutes with
hands-on examples including live market data debates and SEC filing analysis.

<!-- more -->

This tutorial's goal is to show you in 60 minutes:

  - The basic API of AutoGen (an open-source framework for building agentic AI
    systems)
  - Concrete examples of using AutoGen to build agents that can debate and reason
    investment strategies based on market data and SEC filings

## AutoGen in 30 Seconds

AutoGen is an open-source framework from Microsoft for building agentic AI
systems made of multiple collaborating agents.

Key capabilities:

- **Layered APIs**: Message-passing agents, tool usage, and human-in-the-loop
  workflows
- **Autonomous planning**: Reasoning and execution of tasks such as coding,
  browsing, and data processing
- **Multi-language support**: Python and other languages, with extensible
  integrations for LLM providers using OpenAI-style clients
- **Rapid prototyping**: Helps developers quickly build applications from simple
  assistants to coordinated teams of specialized AI agents

## Official References

- [AutoGen: A framework for building AI agents and applications](https://microsoft.github.io/autogen/)
- [GitHub repo](https://github.com/microsoft/autogen)

## Tutorial Content

This tutorial includes all the code, notebooks, and Docker containers in
`tutorials/Autogen/`.

- `tutorials/Autogen/README.md`: Instructions and setup for the tutorial
  environment
- A Docker system to build and run the environment using our standardized
  approach
- `autogen.API.ipynb`: Tutorial notebook focusing on API configurations and
  basic agent setup
- `autogen.example.ipynb`: Advanced end-to-end agentic workflow example
- `autogen_utils.py`: Utility functions required by `autogen.example.ipynb`

## `autogen.example.ipynb`

This notebook provides a practical, end-to-end example of using AutoGen to
demonstrate a complete agentic workflow.

### Part 1: Dynamic Market Debate and Live Data

- Fetches real-time stock data from Yahoo Finance
- Bull and Bear strategist agents debate market trends
- Selector agent dynamically decides which expert to call at each step
- Generates stock charts and financial summaries

### Part 2: SEC Filings and Quantitative RAG Analysis

Part 2 extends Part 1 by combining live market data with deep, structured
analysis of SEC filings.

- Pulls 10-K filings from SEC EDGAR and cleans them
- Embeds documents into a **ChromaDB** vector database
- Senior Quant Analyst agent queries the database to extract revenue splits,
  risk factors, and other insights
- Quant Runtime agent executes Python code locally to transform raw tables into
  structured visualizations

This example shows how multiple agents collaborate, use live data, leverage
private databases via RAG, and produce actionable insights in a single
integrated workflow.
