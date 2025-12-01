# Causal Inference in Social Media Sentiment Analysis with Tetrad

<!-- toc -->
- [Introduction](#introduction)
- [Core Functionality](#core-functionality)
- [Example Description](#example-description)
    * [1. Twitter Sentiment Data Loading and Analysis](#1-twitter-sentiment-data-loading-and-analysis)
    * [2. Twitter Sentiment Data Preprocessing](#2-twitter-sentiment-data-preprocessing)
    * [3. Historical Stock Price Data Loading and Analysis](#3-historical-stock-price-data-loading-and-analysis)
    * [4. Stock Price Data Preprocessing](#4-stock-price-data-preprocessing)
    * [5. Normalizing the Dataset and Creating Features](#5-normalizing-the-dataset-and-creating-features)
    * [6. Setting Prior Knowledge](#6-setting-prior-knowledge)
    * [7. Running the FGES Search Algorithm](#7-running-the-fges-search-algorithm)
    * [8. Running the FCI Search Algorithm](#8-running-the-fci-search-algorithm)
    * [9. Simulation and Sensitivity Analysis](#9-simulation-and-sensitivity-analysis)
    * [10. IMaGES as an Alternative Algorithm](#10-images-as-an-alternative-algorithm)
- [Conclusion](#conclusion)
<!-- tocstop -->

## Introduction

Tetrad is a Java library of various interfaces for exploring causal explanations of data. With the py-tetrad python interface and JPype, we can access all of Tetrad's functionality, from creating graphs of relationships within data to simulating data from constructed models. This example demonstrates how to use Tetrad via the py-tetrad interface to analyze the relationship between historical stock price data and Twitter sentiment data.

In this example, we will:
- Load and preprocess Twitter sentiment data
- Load and preprocess historical stock price data
- Use py-tetrad to apply the FGES algorithm to create a Complete Partially Directed Acyclic Graph (CPDAG) 
- Use py-tetrad to apply the FCI algorithm to create a Partial Ancestral Graph (PAG) and evaluate causal relationships within the data
- Use py-tetrad to train a Causal Perceptron Network to simulate data and assess the robustness of the model
- Use JPype to call Tetrad functionality to use the IMaGES algorithm to build an alternative CPDAG using several different stock datasets

## Core Functionality
This example will make use of the TetradSearch function to run the FGES and FCI algorithms. These two have been selected to evaluate a CPDAG and PAG, respectively, because both are specifically designed to work with continuous data. We will make use of the Knowledge object to apply prior domain knowledge to the system. TetradSearch also allows for statistical tests to be specified for assessing the independence of parameters; in this example we apply the Fisher Z and SEM BIC tests. Then, we will use the generated CPDAG to train a Causal Perceptron Network from which we will simulate data to assess the robustness of the model. Lastly, we will employ the IMaGES algorithm to create an alternative CPDAG using data from multiple different stocks.

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
- **Purpose**: Refine the dataset to align with the sentiment dataset and remove unwanted records and columns.

### 5. Normalizing the Dataset and Creating Features
Since each ticker symbol's data has its own magnitude for features like Volume and Stock Price, they will need to be split by ticker symbol and separately normalized. 

We will also compute an additional "Daily_Stock_Close_Delta" feature to capture the percentage change of a stock price during a single trading day by comparing its opening price to its closing price. This allows us to capture any impact that a given stock's movement has on sentiment, or vice versa, for each day. Similarly, the features "Daily_Stock_Low_Delta" and "Daily_Stock_High_Delta" capture a percentage change relative to the opening price for the lowest and highest prices on a day, respectively.

Additionally, we will create the "EMA_Delta" feature to compare a 5-day exponential moving average against the current day's opening price. This will provide a small window of context that may influence sentiment.

### 6. Setting Prior Knowledge
Tetrad's Knowledge object allows us to incorporate some element of domain knowledge into the model. In this example, the "EMA_Delta" feature uses data that occurs prior to the rest of the features. As such, we can define knowledge tiers such that the "EMA_Delta" feature may influence the other features but cannot itself be influenced by those features. We can also set a required edge from "EMA_Delta" to "Sentiment" since we can safely assume that the recent performance of a stock would have some influence on peoples' opinion of it.

### 7. Running the FGES Search Algorithm
Now that all of the setup has been done, we can plug in the data and knowledge to the TetradSearch algorithm. TetradSearch also requires that a score function be indicated; in this case we will be using both SEM BIC and Fisher Z. The former allows us to compare the strength of edges while building the DAG, the latter helps to determine independence between features, and both are suited for use on continuous data like that which we are using here. We set the time lag to 1 to add a temporal element to the model.

### 8. Running the FCI Search Algorithm
We can set up the FCI search algorithm similarly to the FGES algorithm above. The FCI algorithm outputs a Partial Ancestral Graph (PAG), representing a "set of causal Bayesian networks that cannot be distinguished by the algorithm." For instance, "EMA_Delta --> Sentiment" indicates that EMA_Delta is a direct or indirect cause of Sentiment but it does not rule out the possibility of an unmeasured confounder. 

As another example, "Stock_Volume:1 <-> Stock_Volume" indicates that there exists some unmeasured variable that is a cause of both Stock_Volume:1 and Stock_Volume and that neither causes the other. This intuitevely makes sense as other external events such as news regarding the stock or major price movements would more likely be influencers of trading volume. 

### 9. Simulation and Sensitivity Analysis
To evaluate the robustness of the model, we can train a CausalPerceptronNetwork provided by py-tetrad to simulate data from a graph. Using the CPDAG from the FGES algorithm, we'll generate a new dataset to then attempt to reconstruct the original CPDAG. In this instance, the reconstructed graph is rather poor and has a significantly lower score than the original graph.

### 10. IMaGES as an Alternative Algorithm
The IMaGES algorithm is able to construct multiple DAGs from several similarly-structured datasets to then form one CPDAG. Because IMaGES has slightly less py-tetrad support, we cannot use the TetradSearch object and instead must construct the parameters manually. The resulting graph edges can be directed or undirected, with undirected edges meaning that there are competing directions among the separate individual DAGs. While these do not provide much information about the underlying model, the directed edges indicate agreement among the DAGs and, in this case, inform us that the Daily_Stock_High_Delta is a causal influence on the Daily_Stock_Close_Delta.

## Conclusion
From running these algorithms, we can see that they produce plausible causal graphs modeling the causal effects of social media sentiment on stock market movements. The results of the FGES algorithm, with its only edges into Sentiment coming from Tweet_Volume, indicate that there is little definitive causal effect of sentiment on stock price movements or vice versa in this dataset. The FCI algorithm's PAG results in a similar conlusion. Moreover, this model is not particularly robust as the CPDAG reconstructed from simulated data is both scored as a weaker model and bears little resemblance to the original graph. Finally, the graph produced by the IMaGES algorithm indicates many interconnected nodes but very few directed edges, meaning that the constituent DAGs are not in agreement on the direction of those edges and again indicating a relatively weak model from this data. 

Despite this, Tetrad and its Python package py-tetrad provide access to a suite of search algorithms for analyzing graphical causal models. Incorporating JPype as well allows one to access the whole of Tetrad's functionality.