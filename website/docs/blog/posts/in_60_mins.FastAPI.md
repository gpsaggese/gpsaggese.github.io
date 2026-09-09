---
title: "FastAPI in 60 Minutes"
draft: false
authors:
    - gpsaggese
date: 2026-08-10
description:
categories:
    - Python
    - Software Engineering
---

TL;DR: Learn how to build a REST API with FastAPI and uvicorn in 60 minutes with
hands-on examples covering request validation, dependency injection, and a
complete Book Catalog API served over real HTTP

<!-- more -->

## Tutorial in 30 Seconds
**FastAPI** is a modern Python framework for building APIs, using standard type
hints to drive request validation, serialization, and interactive
documentation, and **uvicorn** is the ASGI server that runs it

Key capabilities:

- **Automatic validation**: `pydantic` models validate request and response
  bodies from type hints alone, with no separate schema to maintain
- **Interactive docs**: Swagger UI and ReDoc are generated automatically from
  the same type hints, always in sync with the code
- **Async native**: first-class support for `async def` endpoints, so a single
  worker serves many concurrent requests
- **Dependency injection**: `Depends()` shares parameter parsing and validation
  logic across endpoints
- **High performance**: built on `Starlette` and `pydantic`, close to Node.js
  and Go for I/O-bound workloads

This tutorial's goal is to show you in 60 minutes:

- The core API of FastAPI: path and query parameters, request bodies, dependency
  injection, and error handling
- How to run a FastAPI app with uvicorn, both in-process with `TestClient` and
  behind a real server
- A complete example: a Book Catalog REST API exercised with real HTTP requests
  over the network

## Official References
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [FastAPI GitHub repo](https://github.com/fastapi/fastapi)
- [uvicorn Documentation](https://www.uvicorn.org/)
- [uvicorn GitHub repo](https://github.com/encode/uvicorn)

## Tutorial Content
This tutorial includes all the code, notebooks, and Docker containers in
[tutorials/fastapi](https://github.com/gpsaggese/gpsaggese.github.io/tree/master/tutorials/fastapi)

- [`README.md`](https://github.com/gpsaggese/gpsaggese.github.io/blob/master/tutorials/fastapi/README.md):
  Instructions and setup for the tutorial environment
- A Docker system to build and run the environment using our standardized
  approach
- [`fastapi.API.ipynb`](https://github.com/gpsaggese/gpsaggese.github.io/blob/master/tutorials/fastapi/fastapi.API.ipynb):
  Tutorial notebook focusing on core FastAPI building blocks
  - Path and query parameters, request body validation with `pydantic` models
  - Dependency injection with `Depends()`
  - Error handling with `HTTPException`
  - The automatic `/docs`, `/redoc`, and `/openapi.json`
- [`fastapi.example.ipynb`](https://github.com/gpsaggese/gpsaggese.github.io/blob/master/tutorials/fastapi/fastapi.example.ipynb):
  End-to-end Book Catalog API example
  - Runs a real `uvicorn` server on a background thread
  - Exercises the API with real HTTP requests via `httpx`
  - Covers create, read, update, delete, filtering, and error responses
- [`fastapi_utils.py`](https://github.com/gpsaggese/gpsaggese.github.io/blob/master/tutorials/fastapi/fastapi_utils.py):
  Utility functions and models shared by both notebooks
