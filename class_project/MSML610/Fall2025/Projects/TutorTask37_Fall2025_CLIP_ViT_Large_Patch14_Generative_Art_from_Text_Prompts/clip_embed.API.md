# clip_embed.API.ipynb Documentation

## Overview

This notebook demonstrates how to set up and implement a FastAPI service for generating CLIP embeddings from images and text. It provides a step-by-step guide for creating a RESTful API that can be used to generate embeddings for both image and text inputs.

## Contents

### Imports and Setup

The notebook begins by importing the necessary libraries:
- `io` for handling byte streams
- `torch` for PyTorch operations
- `FastAPI`, `File`, `Form`, `UploadFile` from FastAPI for building the API
- `JSONResponse` for API responses
- `PIL.Image` for image processing
- `CLIPModel` and `CLIPProcessor` from transformers for the CLIP model

### FastAPI Application Initialization

The notebook sets up a FastAPI application with the title "CLIP Embedding API" and initializes the CLIP model:
- Model: `openai/clip-vit-base-patch32`
- Processor: `openai/clip-vit-base-patch32`
- Device selection: Automatically uses CUDA if available, otherwise falls back to CPU

### API Endpoints

The notebook implements three main API endpoints:

#### 1. POST /embed/image
This endpoint accepts an uploaded image file and returns its CLIP embedding vector.

**Functionality:**
- Accepts an image file via file upload
- Converts the image to RGB format
- Processes the image through the CLIP processor
- Generates image embeddings using the CLIP model
- Returns a JSON response containing the image filename and the embedding vector (as a list)

**Error Handling:**
- Catches exceptions and returns appropriate error messages with status code 500

#### 2. POST /embed/text
This endpoint accepts a text string and returns its CLIP embedding vector.

**Functionality:**
- Accepts text input via form data
- Processes the text through the CLIP processor with padding
- Generates text embeddings using the CLIP model
- Returns a JSON response containing the input text and the embedding vector (as a list)

**Error Handling:**
- Catches exceptions and returns appropriate error messages with status code 500

#### 3. GET /
This is a root endpoint that provides a simple health check for the API.

**Functionality:**
- Returns a JSON response with a status message indicating the API is running

## Purpose

This notebook serves as a reference implementation for:
- Setting up a FastAPI service for CLIP embedding generation
- Creating RESTful endpoints for multimodal embedding generation
- Handling file uploads and form data in FastAPI
- Implementing proper error handling in API endpoints
- Deploying CLIP models in a production-ready API service

## Usage

The code in this notebook can be used directly in the `clip_embed_API.py` file, which is the actual implementation used in the Docker container. The notebook provides an interactive way to understand and test the API endpoints before deployment.

## Key Features

- Asynchronous request handling for better performance
- Automatic device detection (GPU/CPU)
- Proper error handling and status codes
- JSON responses for easy integration
- Support for both image and text inputs

