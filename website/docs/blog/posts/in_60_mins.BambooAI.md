---
title: "BambooAI in 60 Minutes"
draft: false
authors:
    - aver81
    - gpsaggese
date: 2026-04-03
description:
categories:
    - AI Research
    - Software Engineering
---

TL;DR: Learn how to build intelligent data analysis systems with BambooAI in 60
minutes with hands-on examples covering multi-agent workflows, dataframe
analysis, and dynamic code generation.

<!-- more -->

## Tutorial in 30 Seconds

BambooAI is a flexible framework for building intelligent systems made of
multiple collaborating LLM agents that can reason over data and execute code.

Key capabilities:

- **Multi-agent orchestration**: Expert Selector, Code Generator, Error
  Corrector, and other specialized agents working together
- **Data-aware reasoning**: Dataframe Inspector and Analyst agents that
  understand and explore tabular data
- **Dynamic code generation and execution**: Planner and Code Generator agents
  that create and run Python code
- **Multi-provider support**: Works with OpenAI, Anthropic Claude, Google
  Gemini, Mistral, DeepSeek, and more
- **Extensible architecture**: Easy to add custom agents and integrate with
  external tools like Google Search

This tutorial's goal is to show you in 60 minutes:

- The basic API of BambooAI (a framework for building multi-agent AI systems
  that can reason over data)
- Concrete examples of using BambooAI to build specialized agents that analyze
  dataframes, generate code, and execute complex workflows

## Official References

- [BambooAI GitHub Repository](https://github.com/bamboo-ai/bamboo)
- [BambooAI Documentation](https://bamboo-ai.github.io/)

## Tutorial Content

This tutorial includes all the code, notebooks, and Docker containers in
[tutorials/BambooAI](https://github.com/gpsaggese/gpsaggese.github.io/tree/master/tutorials/BambooAI)

- [`README.md`](https://github.com/gpsaggese/gpsaggese.github.io/blob/master/tutorials/BambooAI/README.md): Instructions and setup for the tutorial environment
- A Docker system to build and run the environment using our standardized
  approach
- [`bambooai.API.ipynb`](https://github.com/gpsaggese/gpsaggese.github.io/blob/master/tutorials/BambooAI/bambooai.API.ipynb): Tutorial notebook focusing on fundamental classes,
  methods, and API configurations
- [`bambooai.example.ipynb`](https://github.com/gpsaggese/gpsaggese.github.io/blob/master/tutorials/BambooAI/bambooai.example.ipynb): Complete real-world application workflow using BambooAI
    - Data exploration: Using Dataframe Inspector and Analyst agents to
      understand your data
    - Dynamic planning: Planner agent that decomposes complex analysis tasks
      into steps
    - Code generation: Code Generator agent that writes and proposes solutions
    - Error handling: Error Corrector agent that fixes and improves generated
      code
- [`bambooai.example.py`](https://github.com/gpsaggese/gpsaggese.github.io/blob/master/tutorials/BambooAI/bambooai.example.py): Stand-alone script version of the example for quick
  reference or automation
- [`bambooai_utils.py`](https://github.com/gpsaggese/gpsaggese.github.io/blob/master/tutorials/BambooAI/bambooai_utils.py): Utility functions required by the example notebooks
