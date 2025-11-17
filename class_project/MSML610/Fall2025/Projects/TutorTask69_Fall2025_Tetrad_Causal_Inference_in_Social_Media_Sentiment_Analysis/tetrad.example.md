# Causal Inference in Social Media Sentiment Analysis with Tetrad

<!-- toc -->
- [Introduction](#introduction)
- [Core Functionality](#core-functionality)
- [Example Description](#example-description)
    * [1. Twitter Sentiment Data Loading and Analysis](#1-twitter-sentiment-data-loading-and-analysis)
    * [2. Twitter Sentiment Data Preprocessing](#2-twitter-sentiment-data-preprocessing)
    * [3. Historical Stock Price Data Loading and Analysis](#3-historical-stock-price-data-loading-and-analysis)
    * [4. Stock Price Data Preprocessing](#4-stock-price-data-preprocessing)
    * [5. Tetrad Analysis](#5-tetrad-analysis)
<!-- tocstop -->

## Introduction

Tetrad is a Java library of various interfaces for exploring causal explanations of data. With the py-tetrad python interface, we can access all of Tetrad's functionality, from creating graphs of relationships within data to simulating data from constructed models. This example demonstrates how to use Tetrad via the py-tetrad interface to analyze the relationship between historical stock price data and Twitter sentiment data.

In this example, we will:
- Load and preprocess Twitter sentiment data
- Load and preprocess historical stock price data
- Use Tetrad to graph relationships within the data

## Core Functionality
#TODO: A brief explanation of the Tetrad functions used and the available models

## Example Description
### 1. Twitter Sentiment Data Loading and Analysis
- **Source**: Twitter Sentiment Data from [emad12/stock_tweets_sentiment](https://huggingface.co/datasets/emad12/stock_tweets_sentiment)
    - Post Date
    - Tweet text
    - Sentiment
    - Referenced ticker symbol
- **DataFrame Example**:
    - Using Pandas DataFrame:
    | post_date  | text                               | sentiment | ticker_symbol |
    |------------|------------------------------------|-----------|---------------|
    | 2015-08-26 | $AMZN Reversal that quick from B&R | 0         | AMZN          |

### 2. Twitter Sentiment Data Preprocessing
- **Operations**
    1. **Non-uniform date formats**
        - Convert Unix timestamps to YYYY-MM-DD
        - Convert long-form date strings to YYYY-MM-DD
        - Keep existing YYYY-MM-DD dates as-is
    2. **Drop all low-incidence and incorrectly formatted ticker symbols**
        - Many ticker symbol entries are formatted as full company names
        - Many ticker symbol entries have low counts
- **Purpose**: Data consistency for later comparisons between records.

### 3. Historical Stock Price Data Loading and Analysis
- **Source**: Historical Stock Price Data from [no-ry/world-stock-prices-daily-updating](https://huggingface.co/datasets/no-ry/world-stock-prices-daily-updating)
    - Date
    - Ticker symbol
    - Open
    - High
    - Low
    - Close
    - Volume
- **DataFrame Example**:
    - Using Pandas DataFrame:
    | Date | Open | High | Low | Close | Volume | Ticker|
    | --- | --- | --- | --- | --- | --- | --- | 
    | 2025-07-03 00:00:00-04:00 | 212.145004 | 214.649994 | 211.810104 | 213.550003 | 34,697,317 | AAPL |

### 4. Stock Price Data Preprocessing
- **Operations**
    1. **Date Formatting**
        - Convert dates to YYYY-MM-DD for consistency with other dataset
    2. **Drop all ticker symbols except those in sentiment dataset**
        - Drop these because they will not be used
    3. **Drop partial duplicate records**
        - There are many instances of records with duplicate Date and Ticker combinations
        - Drop the duplicate with lower volume to ensure the dataset is complete
    4. **Add Delta column**
        - Calculated as Close - Open
        - Key insights may be gained from the difference
- **Purpose**: Refine the dataset to align with the sentiment dataset and remove unwanted records and columns.

### 5. Tetrad Analysis
#TODO