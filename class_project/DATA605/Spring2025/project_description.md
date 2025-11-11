# **Class project assignment**

### **ActiveCampaign**

**Title**: Analyze Email Campaign Performance Over Time with ActiveCampaign  
**Difficulty**: 1 (easy)

**Description**  
This project introduces students to working with ActiveCampaign’s API to extract email campaign data, analyze time-based trends, and visualize results using Python.

**Describe Technology**

- **ActiveCampaign**: A customer experience automation platform used for email marketing, CRM, and automation. Its API allows programmatic access to campaign metrics (e.g., open rates, click-through rates) and user data.  
- **Key Features**:  
  - Retrieve historical campaign performance data (time-stamped).  
  - Track user engagement metrics (opens, clicks, unsubscribes).  
  - Automate data extraction for time series analysis.  
    

**Describe the Project**

1. **Set Up API Access**:  
   - Create an ActiveCampaign trial account and generate API credentials.  
   - Use Python’s `requests` library or the `activecampaign` Python client to connect to the API.  
2.   
3. **Extract Time Series Data**:  
   - Fetch email campaign metrics (e.g., daily opens, clicks) for the past 30 days.  
   - Store the data in a pandas DataFrame with timestamps.  
4. **Clean and Analyze Data**:  
   - Handle missing values and outliers.  
   - Calculate trends (e.g., weekly engagement patterns) using moving averages.  
5. **Visualize Results**:  
   - Plot time series trends with `matplotlib` or `seaborn`.  
   - Use `statsmodels` to forecast future engagement (e.g., ARIMA model).

**Useful Resources**

- [ActiveCampaign API Documentation](https://developers.activecampaign.com/reference)  
- [Pandas Time Series Guide](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html)  
- [Matplotlib Tutorials](https://matplotlib.org/stable/tutorials/index.html)


**Is it Free?**

- ActiveCampaign requires a paid account, but a 14-day free trial is available. API access is included in all plans.

**Python Libraries / Bindings**

- **activecampaign-python**: Official Python client for ActiveCampaign’s API. Install via `pip install activecampaign`.  
- **pandas**: For data manipulation and time series analysis.  
- **matplotlib/seaborn**: For visualization.  
- **statsmodels** (optional): For advanced time series modeling.  
- **requests**: For direct API calls if not using the client.

### 

### **Allms**

**Title**: Real-time Bitcoin Sentiment Analysis and Predictive Modeling with allms

**Difficulty**: Difficult

**Description**

- **Technology Overview:**  
  allms is an open-source library that provides a unified interface for interacting with multiple Large Language Models (LLMs). It simplifies tasks like sentiment analysis and topic modeling by integrating with various LLM providers or local models. In this project, allms will process real-time Bitcoin-related text data for sentiment analysis and predictive modeling.  
    
- **Project Details:**  
  This project builds a system to:  
    
  - **Ingest Data:** Fetch real-time Bitcoin-related data from Twitter, Reddit, and news APIs (e.g., Twitter Streaming API, PRAW for Reddit, NewsAPI).  
  - **Process with allms:** Use allms to connect to an LLM (e.g., GPT-3 or a fine-tuned model) to perform sentiment analysis and topic modeling on the text data.  
  - **Time Series Analysis:** Aggregate sentiment scores and topic frequencies into time series data for trend analysis.  
  - **Predictive Modeling:** Develop a model (e.g., LSTM or Prophet) to forecast Bitcoin prices based on sentiment and topics.  
  - **Visualization:** Create a real-time dashboard using Dash or Streamlit to display sentiment, topics, and price predictions.  
  - **Scalability:** Optimize for high data volumes using cloud services like AWS Lambda.


  This project demands real-time data handling, advanced NLP, time series forecasting, and system optimization, making it a complex and time-intensive endeavor.

**Useful Resources**

- [altlms GitHub](https://github.com/allegro/allms)  
- [Twitter Streaming API](https://developer.twitter.com/en/docs/twitter-api/tweets/filtered-stream/introduction)  
- [PRAW Documentation](https://praw.readthedocs.io/en/stable/)  
- [NewsAPI](https://newsapi.org/docs)  
- [CoinGecko API](https://www.coingecko.com/en/api) (Bitcoin price data)  
- [Dash Documentation](https://dash.plotly.com/)  
- [Streamlit Documentation](https://docs.streamlit.io/)

**Is it Free?**

- **allms:** Yes, open-source.  
- **APIs:** Free tiers available (Twitter, Reddit, NewsAPI, CoinGecko) with limitations.  
- **Cloud Services:** AWS Lambda has a free tier; costs may apply with heavy use.

**Python Libraries**

- `allms`: `pip install allms`  
- `tweepy`: `pip install tweepy` (Twitter API)  
- `praw`: `pip install praw` (Reddit API)  
- `requests`: `pip install requests` (API calls)  
- `pandas`: `pip install pandas` (data manipulation)  
- `scikit-learn` or `tensorflow`: `pip install scikit-learn` or `pip install tensorflow` (modeling)  
- `dash` or `streamlit`: `pip install dash` or `pip install streamlit` (dashboard)  
- `boto3`: `pip install boto3` (AWS integration)

### **Altair**

**Title**: Interactive Bitcoin Market Dashboard with Vega-Altair

**Difficulty**: Difficult

**Description**  
Create a real-time interactive visualization system for Bitcoin market data using Vega-Altair's declarative visualization capabilities. The project combines live data streaming with complex financial visualizations, implementing a self-updating dashboard that shows price trends, volatility heatmaps, and on-chain metrics through declarative JSON specifications.

**Describe Technology**

- **Vega-Altair**: Declarative visualization library that:  
  1. Creates interactive charts through concise Python syntax  
  2. Supports complex visual encodings (e.g., layered charts, interactive selections)  
  3. Generates Vega-Lite specifications for web-native rendering  
  4. Enables client-side interactivity without callback functions  
  5. Integrates with Jupyter and web frameworks through JSON output

**Describe the Project**

1. **Real-Time Visualization Pipeline**:  
   - Ingest WebSocket data from Coinbase Pro/Binance  
   - Implement windowed transforms for:  
     - 15-minute candlestick aggregates  
     - Relative Strength Index (RSI) calculations  
     - Miner reserve vs price correlation  
   - Use Altair's `transform_*` methods instead of pre-processing

2. **Advanced Visual Features**:  
   - Create a layered chart with:  
     - Price line (primary axis)  
     - Volume bars (secondary axis)  
     - Bollinger Band confidence intervals  
   - Add interactive elements:  
     - Brush selection for time range focus  
     - Crosshair tooltip with multiple axis values  
     - Legend-driven series toggling  
-   
3. **Dashboard System**:  
   - Build a 3-panel view using Altair's `hconcat`/`vconcat`:  
     1. Time series with technical indicators  
     2. Volatility surface heatmap (time vs window size)  
     3. Mempool transaction size distribution  
   - Implement shared selections across panels  
-   
2. **Deployment Architecture**:  
   - Serve visualizations through FastAPI/Starlette with:  
     - Server-sent events for real-time updates  
     - Vega-Embed for web rendering  
     - Persistent view states through URL parameters  
   - Create a monitoring system that:  
     - Detects chart rendering errors  
     - Auto-adjusts bin sizes based on data density

   

**Useful Resources**

- [Vega-Altair Documentation](https://altair-viz.github.io)  
- [Vega-Altair Interactive Examples](https://github.com/altair-viz/altair_notebooks)  
- [Cryptocurrency Market Data Best Practices](https://www.kaiko.com/insights)


**Is it free?**  
Yes \- Vega-Altair is open-source (BSD-3). Exchange APIs have rate-limited free tiers.

**Python Libraries / Bindings**

- `altair`: Core visualization engine  
- `websockets`/`aiohttp`: Real-time data ingestion  
- `pandas`: Windowed transformations  
- `fastapi`/`starlette`: Dashboard serving  
- `jinja2`: Template rendering for web views


### **AWS Athena**

* **Title**: Analyze Bitcoin price trends using AWS Athena  
* **Difficulty**: 1 (Easy)  
* **Description**  
  * **AWS Athena** is an interactive query service that makes it easy to analyze data directly in Amazon S3 using standard SQL. Athena is serverless, so there is no need to manage infrastructure, and it can query large datasets with ease. It supports data formats such as CSV, JSON, Parquet, ORC, and Avro. Athena automatically integrates with AWS Glue for data cataloging, making it a powerful tool for big data analytics.  
  * **The project** involves using AWS Athena to analyze Bitcoin price data in real time. You will first collect Bitcoin price data from a public API like CoinGecko or CryptoCompare and store it in an S3 bucket in JSON format. The next step is to create a Glue Data Catalog for this data, allowing Athena to query it. After setting up the catalog, you will write SQL queries in Athena to perform time series analysis, such as calculating moving averages or identifying price trends over specific time intervals. Finally, you will present the results of your analysis in a structured format, such as CSV or Parquet, which can be further processed or visualized. This project gives students a hands-on introduction to serverless analytics and time series analysis using SQL, making it a practical and efficient way to process large-scale data.  
* **Useful resources**  
  * **Is it free?**  
    * Athena charges based on the amount of data scanned by your queries. It’s cost-effective for smaller datasets, but be mindful of query optimization to reduce costs. AWS offers a free tier with limited usage.  
  * **Python libraries / bindings**

    * **boto3**: The official AWS SDK for Python allows you to interact with Athena and other AWS services programmatically. You can use boto3 to submit queries, retrieve results, and manage your S3 bucket. You can install it via `pip install boto3`.  
    * **pandas**: Used to process and visualize the query results from Athena in Python. It's helpful for analyzing data and generating reports. Install it via `pip install pandas`.  
    * **awswrangler**: A library that simplifies interaction with AWS services, particularly for querying Athena and working with data stored in S3. It supports data frames and integrates with pandas. Install via `pip install awswrangler`.  
    * **SQL**: The query language you will use within Athena to interact with your data. You will write SQL queries for time series analysis, filtering, and aggregating the Bitcoin price data.  
  * **References**  
    * **boto3 (AWS SDK for Python)**: [Boto3 Documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)  
    * **awswrangler**: [AWS Wrangler Documentation](https://aws-sdk-pandas.readthedocs.io/en/stable/)  
    * **AWS Athena**: [AWS Athena Documentation](https://docs.aws.amazon.com/athena/latest/ug/what-is-athena.html)  
    * **pandas**: [Pandas Documentation](https://pandas.pydata.org/pandas-docs/stable/)

### **Amazon Aurora**

**Title**: Analyzing Bitcoin Trends with Amazon Aurora

**Difficulty**: Easy

**Description**  
Amazon Aurora is a fully managed relational database engine provided by Amazon Web Services (AWS), compatible with both MySQL and PostgreSQL databases. It combines the speed and availability of high-end commercial databases with the simplicity and cost-effectiveness of open-source databases. This project involves using Amazon Aurora to store, manage, and analyze Bitcoin price data as part of a real-time data system. The project will provide hands-on experience in setting up a database in Amazon Aurora, ingesting Bitcoin data, and performing basic time series analysis in Python.

**Describe technology**

- Amazon Aurora is designed for mission-critical workloads and delivers an enhanced performance and availability.  
- Offers built-in automation for high availability with recovery features and up to 15 low-latency read replicas.  
- Provides self-healing storage, which automatically scales up to 128 terabytes per database instance.  
- Easily integrates with other AWS services like Lambda, S3, and more for seamless data processing and analysis.

**Describe the project**

- Students will begin by setting up an Amazon Aurora database instance with MySQL compatibility.  
- Using the Python `requests` package, students will write a simple script to fetch real-time Bitcoin price data from an API like CoinGecko.  
- The script will then insert the fetched data into a table within the Amazon Aurora database.  
- Perform time series analysis by querying this data, such as calculating the average daily price and identifying trends over specific periods.  
- Visualization of the data trends can be achieved using Python libraries like Matplotlib or Seaborn.

**Useful resources**

- [Amazon Aurora Documentation](https://docs.aws.amazon.com/AmazonRDS/latest/AuroraUserGuide/Welcome.html)  
- [CoinGecko API Documentation](https://www.coingecko.com/en/api)  
- [AWS Python Boto3 Documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)

**Is it free?**  
You need to create an AWS account. Amazon Aurora offers a free tier, which includes 750 hours of usage per month for a limited time, enabling students to explore Aurora without incurring costs, within certain limitations.

**Python libraries / bindings**

- **Boto3**: The official Amazon Web Services (AWS) SDK for Python, which will be used to interact with and manage your Amazon Aurora instance. You can install it using `pip install boto3`.  
- **MySQL Connector/Python**: For connecting to the Aurora MySQL database. Install it using `pip install mysql-connector-python`.  
- **Requests**: A simple HTTP library in Python to fetch data from APIs (e.g., CoinGecko) with `pip install requests`.  
- **Matplotlib/Seaborn**: For visualization purposes. They can be installed using `pip install matplotlib seaborn`.

### **Amazon Batch**

**Title**: Real-Time Bitcoin Data Processing with Amazon Batch=  
**Difficulty**: Difficult

**Description**  
Amazon Batch is a fully managed batch processing service that enables developers, scientists, and engineers to run hundreds to thousands of batch computing jobs efficiently. It allows users to efficiently manage their compute resources while letting Amazon Batch handle job scheduling, compute environment scaling, and resource provisioning. This project focuses on using Amazon Batch to ingest and process real-time Bitcoin data for complex analytics, specifically performing time series analysis.

**Describe technology**

- **Amazon Batch**: A service designed to handle large-scale batch computing at any scale. It automates job scheduling and execution on compute resources such as EC2 instances or Fargate containers. Key components include job definitions, compute environments, and job queues.  
- **Job Queues**: Define the order in which jobs run. You can set priorities and policies for your jobs.  
- **Job Definitions**: Specify how jobs should be run, including Docker container properties, resource requirements, IAM roles, and environment variables.  
- **Compute Environments**: Specify AWS resources that Amazon Batch can use. Can be on-demand EC2 instances, Spot Instances, or a mixture of both.

**Describe the project**

1. **Data Ingestion**:  
     
   - Set up a process using a Python script to gather live Bitcoin price data from an API (such as CoinGecko) at regular intervals.  
   - Store incoming data in an S3 bucket for persistent storage and further processing.

   

2. **Amazon Batch Setup**:  
     
   - Design and create a job definition that describes the time series analysis tasks, specifying necessary resources and execution settings.  
   - Configure a compute environment to leverage scalable EC2 instances for handling peak and off-peak workloads efficiently.  
   - Set up a job queue to manage job prioritization and resource allocation for high efficiency.

   

3. **Time Series Analysis**:  
     
   - Implement a Python-based analysis using libraries such as pandas and statsmodels to detect patterns, trends, and perform predictions on Bitcoin price fluctuations.  
   - Develop algorithms to process data in intervals, such as calculating moving averages, ARIMA model predictions, and identifying volatility based on historical and real-time data.

   

4. **Results Processing**:  
     
   - Store results of the analysis back in the S3 bucket for archival and historical analysis.  
   - Develop scripts to generate visualizations (using libraries like matplotlib or seaborn) to intuitively present time series analysis findings.

   

5. **Automation and Scaling**:  
     
   - Automate the scheduling and scaling of analysis tasks using job queues to adapt to variable loads, ensuring efficient use of compute resources.

**Useful resources**

- [Amazon Batch Documentation](https://docs.aws.amazon.com/batch/index.html)  
- [CoinGecko API Documentation](https://www.coingecko.com/en/api)  
- [Python pandas Documentation](https://pandas.pydata.org/docs/)  
- [statsmodels Documentation](https://www.statsmodels.org/stable/index.html)

**Is it free?**  
Amazon Batch is part of AWS services, which offers a free tier but requires an AWS account. Costs may incur based on compute resources (EC2 instances) used beyond the free tier limits.

**Python libraries / bindings**

- `boto3`: The official AWS SDK for Python; used to interact with AWS services, including Amazon Batch.  
- `pandas`: For data manipulation and analysis in Python; essential for processing and analyzing JSON or CSV data.  
- `statsmodels`: A Python module for statistical modeling that provides tools for time series analysis, such as ARIMA models.  
- `matplotlib` / `seaborn`: Visualization libraries in Python; useful for generating plots and graphs from the analyzed data.

### 

### **Amazon CloudFormation**

**Title**: Real-Time Bitcoin Price Analysis Pipeline with AWS CloudFormation  
**Difficulty**: 3 (Difficult)

**Description**  
This project challenges students to build an automated AWS pipeline for ingesting, processing, and analyzing real-time Bitcoin price data using AWS CloudFormation. The pipeline includes machine learning integration for price trend prediction and alerting.

**Describe Technology**

- **AWS CloudFormation**: Infrastructure-as-Code (IaC) service for deploying complex architectures.  
- **AWS Services**:  
  - **Kinesis Data Streams**: Real-time Bitcoin price ingestion  
  - **SageMaker**: ML model training/predictions  
  - **Lambda (Python)**: Data processing and alert logic  
  - **CloudWatch**: Monitoring and triggers  
- **Key Features**:  
  - End-to-end automation of financial data pipeline  
  - ML integration for predictive analytics  
  - Real-time alerting system


**Describe the Project:**

1. **Design the Infrastructure**:  
   - Create a CloudFormation template to deploy:  
     - Kinesis stream for live price data ingestion  
     - Lambda functions for data cleaning/transformation  
     - SageMaker endpoint for LSTM/Prophet price predictions  
     - S3 buckets for raw data and model artifacts  
     - SNS topic for price alert notifications

   

2. **Implement Python Logic**:  
   - Build a Python scraper to feed live Bitcoin prices to Kinesis (CoinGecko API)  
   - Develop Lambda functions to:  
     - Calculate moving averages/RSI indicators  
     - Train/update ML models using historical data  
     - Compare predictions vs actual prices  
   - Create CloudWatch-triggered retraining workflow

   

3. **Advanced Features**:  
   - Implement anomaly detection for sudden price swings  
   - Deploy automated trading signals (e.g., "BUY/SELL" alerts via SNS)  
   - Optimize costs using spot instances for SageMaker training  
     

**Useful Resources**

- [CoinGecko API Documentation](https://www.coingecko.com/en/api)  
- [AWS SageMaker Python SDK](https://sagemaker.readthedocs.io/)  
- [Time Series Forecasting with TensorFlow](https://www.tensorflow.org/tutorials/structured_data/time_series)


**Is it Free?**

- CloudFormation is free, but Kinesis/SageMaker/Lambda incur costs. Free tier limits apply.  
- CoinGecko API: Free plan (50 calls/min)


**Python Libraries / Bindings**

- **boto3**: AWS service interactions  
- **yfinance/pycoingecko**: Bitcoin price data fetching  
- **tensorflow/pytorch**: ML model development  
- **pandas-ta**: Technical indicator calculations  
- **fastapi** (optional): Prediction endpoint

### **Amazon Data Pipeline**

**Title**: Real-Time Bitcoin Data Processing with Amazon Data Pipeline

**Difficulty**: Medium

**Description**  
Amazon Data Pipeline is a web service that assists in automating the movement and transformation of data. With its easy-to-use interface, you can define a data processing workflow or pipeline that involves multiple stages, handling tasks like scheduling, dependency tracking, retry policies, and more. For data scientists and engineers, it provides the tools necessary to create data-driven workflows at scale.

**Describe technology**

- **Purpose and Functionality**: Amazon Data Pipeline is used to manage the flow of data between compute and storage services inside and outside of AWS. It orchestrates and automates the data-driven workflows and handles complex scheduling and dependency management.  
- **Key Components**:  
  - **Pipeline Definition**: The JSON specification to define the inputs, outputs, and activity logic.  
  - **Tasks**: Units of work performed during execution.  
  - **Schedules and Preconditions**: Define timing and dependencies for task execution.  
  - **Resources**: AWS resources like EC2 instances or S3 buckets involved in the pipeline.

**Describe the project**  
In this project, students will build a data pipeline to ingest real-time Bitcoin price data, process it to perform basic time series analysis, and store the results for further analysis.

- **Steps**:  
  - **Data Ingestion**: Use Amazon Data Pipeline to fetch Bitcoin prices from a public API, such as CoinGecko, at regular intervals.  
  - **Data Storage**: Store the ingested data in an AWS S3 bucket as a CSV or JSON file for easy access.  
  - **Data Processing**: Write a Python script using basic libraries such as Pandas and NumPy to perform time series analysis. This can involve calculating moving averages or identifying price trends over specific periods.  
  - **Data Transformation**: Use Amazon Data Pipeline to run the Python script on an EC2 instance, process the data, and store the results back in S3.  
  - **Schedule and Automate**: Set up the pipeline to run the data collection and processing activities on a regular schedule.

**Useful resources**

- [Amazon Data Pipeline Documentation](https://docs.aws.amazon.com/datapipeline/latest/DeveloperGuide/what-is-datapipeline.html)  
- [CoinGecko API Documentation](https://www.coingecko.com/en/api)  
- [AWS S3](https://docs.aws.amazon.com/AmazonS3/latest/user-guide/what-is-s3.html)  
- [Pandas Documentation](https://pandas.pydata.org/pandas-docs/stable/)  
- [NumPy Documentation](https://numpy.org/doc/)

**Is it free?**  
Amazon Data Pipeline is a managed service. While AWS offers a free tier that includes limited access to many of its services, usage of Data Pipeline incurs costs after the free-tier limits. Always check the latest pricing on the AWS Pricing page.

**Python libraries / bindings**

- **boto3**: The AWS SDK for Python, used to interact with AWS services programmatically, including managing the data pipeline. Install using `pip install boto3`.  
- **Pandas**: A library providing data manipulation and analysis tools for Python. Used here for time series analysis. Install using `pip install pandas`.  
- **NumPy**: A package for scientific computing with Python, helpful for performing mathematical functions used in time series analysis. Install using `pip install numpy`.

### **Amazon DynamoDB**

**Title**: Real-Time Bitcoin Data Processing with Amazon DynamoDB  
**Difficulty**: 

**Description**  
Amazon DynamoDB is a fully managed NoSQL database service provided by Amazon Web Services (AWS). It's designed to provide consistent, single-digit millisecond latency at any scale, making it well-suited for applications requiring a reliable, fast, and scalable database. In this project, students will leverage Amazon DynamoDB to ingest, store, and analyze real-time Bitcoin price data. This medium-difficulty project will take approximately ten days to complete, offering students hands-on experience with setting up a DynamoDB table, ingesting data using a real-time streaming service, and performing basic time series analysis on the data.

**Describe technology**

- **NoSQL Database**: Amazon DynamoDB is a key-value and document database optimized for speed and scalability.  
- **Managed Service**: No need for provisioning, patching, or operating hardware. Offers automatic scaling and built-in security.  
- **Streams**: DynamoDB Streams provide an ordered flow of changes to the table, enabling real-time data replication and event-driven processing.  
- **Integrations**: Easily integrate with AWS Lambda for serverless compute possibilities or Amazon Kinesis for streaming analytics.

**Describe the project**

- **Objective**: Create a real-time data ingestion system for Bitcoin price tracking, storing the data in DynamoDB, and performing initial time series analysis.  
- **Setup DynamoDB Table**: Create a DynamoDB table designed to hold Bitcoin price data, structured with a primary key of timestamp and additional attributes for price and metadata.  
- **Data Ingestion**: Utilize a Python script to fetch Bitcoin price data from a public API, such as CoinGecko, and insert this data into the DynamoDB table in real-time.  
- **Real-time Processing**: Implement DynamoDB Streams to detect and analyze changes in the stored data; use AWS Lambda to trigger instant computations, such as calculating moving averages or detecting price anomalies.  
- **Time Series Analysis**: Using data stored in DynamoDB, perform a basic time series analysis such as identifying trends or creating visualizations that help understand price movements over time.  
- **Final Output**: Compile insights into a report or presentation that showcases data processing steps, findings from time series analysis, and suggestions for further exploration.

**Useful resources**

- [Amazon DynamoDB Documentation](https://docs.aws.amazon.com/dynamodb/index.html)  
- [DynamoDB Streams Overview](https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/Streams.html)  
- [AWS Lambda FAQs](https://aws.amazon.com/lambda/faqs/)  
- [Python Boto3 Documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)

**Is it free?**  
You need to create an AWS account. DynamoDB offers a free tier that includes 25 GB of storage and enough read/write capacity for small to medium workloads. However, exceeding this tier or using additional services will incur charges.

**Python libraries / bindings**

- **boto3**: The official AWS SDK for Python. Install it via `pip install boto3`. Use boto3 to interact with DynamoDB and manage database operations programmatically.  
- **AWS Lambda with Python**: For writing serverless functions to process real-time data events. AWS provides guidance and support for setting up Lambda functions in Python.  
- **Requests**: A simple HTTP library for Python, used to fetch real-time Bitcoin data from public APIs. Install it via `pip install requests`.

### **Amazon ElasticSearch/OpenSearch**

**Title**: Bitcoin Price Trend Analysis and News Correlation with AWS OpenSearch  
**Difficulty**: 2 (Medium)

**Description**:  
Students will build a system to analyze Bitcoin price trends using AWS OpenSearch for time series analysis **and** incorporate semantic search/RAG techniques to correlate price anomalies with cryptocurrency news articles.  
**Describe Technology**:

- **AWS OpenSearch Service**: Validated for time series use cases via:  
  - Native support for `date_histogram` aggregations and time-based indexing.  
  - Built-in [Anomaly Detection](https://opensearch.org/docs/latest/monitoring-plugins/ad/index/) for automated pattern recognition.  
  - Hybrid use case: Combines time series analysis (price data) with RAG/semantic search (news articles).

**Describe the Project**:

1. **Time Series Pipeline**:  
   - Use Python to fetch real-time Bitcoin price data (e.g., CoinGecko API) and ingest into OpenSearch with timestamped indices.  
   - Perform time series aggregations (e.g., volatility analysis, moving averages) using OpenSearch DSL.  
2. **RAG Integration**:  
   - Scrape/news-API Bitcoin-related articles (e.g., Reddit, CryptoNews) and index them in OpenSearch.  
   - Use OpenSearch's semantic search to retrieve news snippets during priAmazon DynamoDB ce spikes/drops detected via anomaly analysis.  
3. **Correlation Analysis**:  
   - Build a dashboard showing Bitcoin price trends \+ annotated news events (e.g., "Regulation announcement → 12% price drop").

   

**Useful Resources**:

- OpenSearch [Time Series Documentation](https://opensearch.org/docs/latest/search-plugins/timeseries/)  
- AWS Guide: [Combining Time Series and Text Data](https://docs.aws.amazon.com/opensearch-service/latest/developerguide/time-series.html)  
- [Bitcoin Historical Data API](https://www.coingecko.com/en/api)


**Is it free?**:  
AWS Free Tier covers small-scale testing. News APIs may have free tiers (e.g., Reddit API).

**Python Libraries / Bindings**:

- Core: `opensearch-py`, `pandas`, `requests`  
- Data: `coingecko-api` (Python wrapper), `beautifulsoup4` (news scraping)  
- Visualization: `matplotlib`, OpenSearch Dashboards


### **Amazon ElasticSearch/OpenSearch**

**Title: Bitcoin Price Trend Analysis and News Correlation with AWS OpenSearch**  
**Difficulty: 2 (Medium)**

**Description:**  
**Students will build a system to analyze Bitcoin price trends using AWS OpenSearch for time series analysis and incorporate semantic search/RAG techniques to correlate price anomalies with cryptocurrency news articles.**  
**Describe Technology:**

- **AWS OpenSearch Service: Validated for time series use cases via:**  
  - **Native support for date\_histogram aggregations and time-based indexing.**  
  - **Built-in Anomaly Detection for automated pattern recognition.**  
  - **Hybrid use case: Combines time series analysis (price data) with RAG/semantic search (news articles).**

**Describe the Project:**

4. **Time Series Pipeline:**  
   - **Use Python to fetch real-time Bitcoin price data (e.g., CoinGecko API) and ingest into OpenSearch with timestamped indices.**  
   - **Perform time series aggregations (e.g., volatility analysis, moving averages) using OpenSearch DSL.**  
5. **RAG Integration:**  
   - **Scrape/news-API Bitcoin-related articles (e.g., Reddit, CryptoNews) and index them in OpenSearch.**  
   - **Use OpenSearch's semantic search to retrieve news snippets during priAmazon DynamoDB ce spikes/drops detected via anomaly analysis.**  
6. **Correlation Analysis:**  
   - **Build a dashboard showing Bitcoin price trends \+ annotated news events (e.g., "Regulation announcement → 12% price drop").**

   

**Useful Resources:**

- **OpenSearch Time Series Documentation**  
- **AWS Guide: Combining Time Series and Text Data**  
- **Bitcoin Historical Data API**


**Is it free?:**  
**AWS Free Tier covers small-scale testing. News APIs may have free tiers (e.g., Reddit API).**

**Python Libraries / Bindings:**

- **Core: opensearch-py, pandas, requests**  
- **Data: coingecko-api (Python wrapper), beautifulsoup4 (news scraping)**  
- **Visualization: matplotlib, OpenSearch Dashboards**


### **Amazon EMR**

**Title**:c Real-Time Bitcoin Price Analysis Using Amazon EMR

**Difficulty**: Medium

**Description:** Amazon Elastic MapReduce (EMR) is a cloud-native big data platform for processing vast amounts of data quickly and cost-effectively. It simplifies running large-scale data frameworks like Apache Spark, Hadoop, and other related applications in an easily scalable and managed environment. The project centers around using Amazon EMR to perform real-time processing of Bitcoin price data, highlighting time-series analysis capabilities.

**Describe Technology:**

- **Amazon EMR**: This managed cluster platform simplifies running big data applications across rapidly scalable and secure infrastructure. By integrating with other AWS services, EMR provides a powerful environment for data processing, analysis, and storage.  
  - **Core Components**: In the context of this project, Apache Spark will be the central framework for processing data, given its real-time data handling capabilities.  
  - **Scalability & Cost-Effectiveness**: Easily adjust resources to fit workload needs, thereby managing costs effectively.  
  - **Integration with AWS Services**: Works seamlessly with Amazon S3 for data storage, Amazon RDS for databases, and other data sources.

**Describe the Project:**

- **Objective**: Develop a real-time system leveraging Amazon EMR to ingest Bitcoin prices from a public API, process this data in real-time, and perform time-series analysis.  
- **Tasks**:  
  1. **Data Ingestion**: Use an API like CoinGecko or CryptoCompare to fetch real-time Bitcoin prices.  
  2. **Cluster Setup**: Launch and configure an EMR cluster with Apache Spark for real-time data processing.  
  3. **Data Processing**: Write a PySpark application that transforms and analyzes the incoming Bitcoin data. Key operations include data cleaning, filtering by specific time intervals, and aggregating data for summary statistics.  
  4. **Time-Series Analysis**: Implement basic time-series analyses, such as calculating moving averages or analyzing price fluctuations over given time windows.  
  5. **Data Storage**: Store raw and analyzed data in Amazon S3 in a format suitable for further analysis or visualization.  
  6. **Automation and Scaling**: Use EMR steps and Spark jobs to automate the workflow and scale according to the data volume.

**Useful Resources**:

- [Amazon EMR Documentation](https://docs.aws.amazon.com/emr/)  
- [Apache Spark Documentation](https://spark.apache.org/docs/latest/)  
- [CoinGecko API Documentation](https://www.coingecko.com/en/api)  
- [AWS Python SDK (Boto3) Documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)

**Is it free?** You need to create an AWS account. Amazon EMR is not free, and the cost depends on the resources you use (e.g., number of nodes, duration of running the cluster). However, you can use AWS's Free Tier for some preliminary steps or testing, but EMR usually incurs additional costs.

**Python Libraries / Bindings**:

- **Boto3**: The AWS SDK for Python, crucial for managing AWS services programmatically, including the setup and control of EMR clusters.  
- **PySpark**: Use PySpark to write distributed data processing applications within the Amazon EMR ecosystem.  
- **Pandas & NumPy**: Helpful for local processing or testing of data before scaling on EMR.  
- **Requests**: To fetch data from APIs like CoinGecko for ingestion into the EMR pipeline.

### 

### **AWS Glue**

- Title: Ingest bitcoin prices using AWS Glue  
- Description  
  - AWS Glue is a fully managed extract, transform, and load (ETL) service provided by Amazon Web Services (AWS). It helps you prepare and transform data for analytics, machine learning, and other data processing tasks. Understand AWS Glue's core concepts, including data catalogs, crawlers, and jobs  
  - An easy project that can be completed in one week involves using AWS Glue to ingest Bitcoin price data from a public API, such as CoinGecko, and store it in an S3 bucket for further analysis. The first step of the project is to set up an AWS Glue crawler to automatically discover and catalog the incoming JSON data. Students will configure the crawler to run on a schedule to continuously fetch new Bitcoin price data from the API every few minutes. They will store the data in an S3 bucket and create a Glue Data Catalog to manage the dataset’s schema. Then, a simple AWS Glue job can be written using Python (with PySpark) to process and transform the raw data, such as filtering the data for specific time intervals or adding new fields, like the price change over time. Finally, the processed data can be stored back in S3 in a more structured format (e.g., Parquet) for easy querying. This project gives students hands-on experience with AWS Glue and basic ETL processes while learning how to handle real-time data ingestion.  
- Difficulty: Easy  
- Useful links  
  - Is it free?  
    - You need to create an AWS account. AWS Glue offers a free tier, but it comes with some limitations.  
  - Python libraries / bindings  
    - To use AWS Glue with Python, the primary resource you'll need is boto3, the official AWS SDK for Python, which allows you to interact programmatically with AWS Glue services, such as creating and managing Glue jobs, crawlers, and data catalogs. You can install boto3 using pip install boto3. For processing and transforming data within AWS Glue jobs, you'll use PySpark, which is the Python API for Apache Spark. AWS Glue automatically provides a managed Spark environment, so you don't need to install PySpark locally; you simply write your ETL scripts using PySpark within Glue's job editor. Additionally, AWS Glue provides its own Python library, AWS Glue Python Library, which includes utilities for managing Glue jobs, handling data transformations, and interacting with the Glue Data Catalog. These resources together enable you to build, schedule, and execute ETL jobs for large-scale data processing in a serverless environment.  
    - **boto3 (AWS SDK for Python)**: The official Python SDK for interacting with AWS services, including AWS Glue. It provides tools to create, manage, and monitor Glue jobs, crawlers, and data catalogs.  [AWS boto3 Documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)  
    - **PySpark**: A Python API for Apache Spark, used for writing distributed data processing scripts in AWS Glue. PySpark is key for data transformation within Glue jobs.  [PySpark Documentation](https://spark.apache.org/docs/latest/api/python/)  
    - **AWS Glue Python Library**: A specialized library in AWS Glue for managing and transforming data, including utilities for Glue-specific tasks such as working with the Glue Data Catalog and executing ETL jobs. [AWS Glue Python API Reference](https://docs.aws.amazon.com/glue/latest/dg/aws-glue-programming-python.html)

### **Amazon IoT Analytics**

**Title**: Predictive Maintenance System Using AWS IoT Analytics for Time Series Analysis  
**Difficulty**: 2 (Medium)

**Description**  
This project involves building a predictive maintenance system for industrial IoT devices using AWS IoT Analytics. Students will simulate sensor data, ingest it into AWS IoT Analytics, perform time series analysis, and predict equipment failure using Python.

**Describe Technology**

- **AWS IoT Analytics**: A managed service for cleaning, transforming, and analyzing IoT data at scale.  
  - Features: Data ingestion, preprocessing (e.g., filtering anomalies), storage in queryable datasets, and integration with Jupyter Notebooks for analysis.  
  - Use cases: Industrial IoT monitoring, real-time analytics, and predictive maintenance.

**Describe the Project**

1. **Simulate IoT Sensor Data**: Use Python to generate synthetic time series data (e.g., temperature, vibration) from industrial machines.  
2. **AWS IoT Analytics Pipeline**:  
   - Ingest data into AWS IoT Core and route it to AWS IoT Analytics.  
   - Preprocess data (e.g., remove outliers, normalize values) using AWS IoT Analytics pipelines.  
   - Store cleaned data in a dataset.  
3. **Time Series Analysis**:  
   - Use Python (via a Jupyter Notebook in AWS IoT Analytics) to analyze trends, seasonality, and anomalies.  
   - Build a forecasting model (e.g., ARIMA or Prophet) to predict equipment failure.  
4. **Visualization**: Plot results using Matplotlib/Seaborn to show predicted vs. actual values.  
   

**Useful Resources**

- [AWS IoT Analytics Documentation](https://docs.aws.amazon.com/iotanalytics/)  
- [Time Series Forecasting with Prophet](https://facebook.github.io/prophet/docs/quick_start.html)  
- [AWS IoT Analytics Tutorial](https://aws.amazon.com/getting-started/hands-on/analyze-iot-data/)

**Is it free?**

- AWS IoT Analytics has a free tier for 12 months, but costs may apply for high-volume usage.

**Python Libraries / Bindings**

- `boto3` (AWS SDK for Python) to interact with AWS services.  
- `pandas` for data manipulation.  
- `matplotlib`/`seaborn` for visualization.  
- `prophet` or `statsmodels` for time series forecasting.  
- `numpy` for numerical operations.

### **Amazon Kinesis**

**Title**: Real-Time Bitcoin Price Anomaly Detection Using Amazon Kinesis  
**Difficulty**: 3 (Difficult)

**Description**  
This project involves building a real-time data pipeline to monitor Bitcoin prices, detect anomalies (e.g., sudden price spikes/drops), and trigger alerts using Amazon Kinesis. Students will stream live cryptocurrency data, process it with Kinesis, apply machine learning for anomaly detection, and visualize trends.

**Describe Technology**

- **Amazon Kinesis**: A suite for real-time data streaming and analytics.  
  1. **Kinesis Data Streams**: Ingests high-throughput Bitcoin price data from APIs.  
  2. **Kinesis Data Analytics**: Processes streaming data in real time (e.g., calculating moving averages).  
  3. **Kinesis Data Firehose**: Stores results in Amazon S3 or Redshift for historical analysis.  
  4. Use cases: Financial monitoring, algorithmic trading, fraud detection.


**Describe the Project**

1. **Stream Bitcoin Price Data**:  
   - Use Python (`ccxt` or `requests`) to fetch real-time Bitcoin prices from APIs like CoinGecko, Binance, or Coinbase.  
   - Stream data (price, volume, timestamp) to Kinesis Data Streams using `boto3`.  
2. **Real-Time Data Processing**:  
   - Use **Kinesis Data Analytics** (Apache Flink or SQL) to:  
     - Compute rolling averages and volatility metrics over 1-minute windows.  
     - Flag potential anomalies (e.g., prices deviating \>3σ from the mean).  
3. **Machine Learning Integration**:  
   - Deploy a pre-trained anomaly detection model (e.g., Isolation Forest, Autoencoder) via AWS Lambda.  
   - Trigger Lambda to score incoming data and send alerts (e.g., Amazon SNS) for severe anomalies.  
4. **Storage & Visualization**:  
   - Use Kinesis Data Firehose to archive raw and processed data in Amazon S3.  
   - Build a real-time dashboard with `plotly-dash` or AWS QuickSight to display price trends and anomalies.

**Useful Resources**

- [Amazon Kinesis Developer Guide](https://docs.aws.amazon.com/kinesis/)  
- [CoinGecko API Documentation](https://www.coingecko.com/en/api)  
- [Isolation Forest Anomaly Detection Tutorial](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)  
- [AWS Lambda with Python](https://docs.aws.amazon.com/lambda/latest/dg/python-handler.html)

**Is it free?**

- Kinesis offers a limited free tier (2 shards/month). Costs scale with data volume and shard usage.

**Python Libraries / Bindings**

- `boto3` (AWS SDK) to interact with Kinesis, Lambda, and S3.  
- `ccxt` or `requests` for fetching cryptocurrency data.  
- `pandas`/`numpy` for data transformations.  
- `scikit-learn` or `tensorflow` for anomaly detection models.  
- `plotly-dash` for dashboarding.  
- 

### **Amazon Lambda**

Title: Real-Time Bitcoin Price Processing with Amazon Lambda

**Difficulty**: Difficult

**Description**

Amazon Lambda, commonly referred to as AWS Lambda, is a serverless computing service that allows you to run code without provisioning or managing servers. It automatically scales your applications by executing code in response to triggers from various AWS services or direct invocations. This project focuses on utilizing AWS Lambda to ingest and process real-time Bitcoin price data for time series analysis.

**Describe Technology**

- **Serverless Computing**: AWS Lambda handles all of the capacity scaling, patching, and general infrastructure maintenance, allowing developers to focus solely on code features.  
- **Event-driven Architecture**: AWS Lambda can be triggered by a variety of event sources, such as changes in a data stream, messages on a queue, or API requests.  
- **Pay-as-you-use Pricing**: With Lambda, costs are incurred based only on the number of requests and the execution time utilized.  
- **Flexible Runtime Support**: Write your lambda functions in several languages, including Python, Java, Go, and more.  
- **Integrations with AWS Services**: Lambda can natively interact with multiple other AWS services, such as S3 for storage, DynamoDB for databases, and Kinesis for real-time data streams.

**Describe the Project**

- **Objective**: Implement a real-time data ingestion and processing system to analyze Bitcoin price fluctuations using AWS Lambda.  
- **Data Source**: Use the WebSocket API from cryptocurrency exchange platforms such as Coinbase Pro or Binance to receive real-time Bitcoin price data.  
- **System Architecture**:  
  - **Data Ingestion**: Develop a Lambda function triggered by a scheduled Amazon EventBridge rule to initiate a WebSocket connection.  
  - **Data Processing**: Another Lambda function will process incoming Bitcoin price data in near real-time. This function will extract, filter, and transform the data to derive meaningful insights, such as percentage price changes over predefined time intervals.  
  - **Data Storage**: Persist transformed data in Amazon S3 in a time-series-optimized format (e.g., Parquet/JSON) for further analysis.  
  - **Time Series Analysis**: Perform basic statistical computations or visualizations directly within AWS Lambda or using additionally integrated services like AWS QuickSight.  
- **Outcome**: A functioning distributed system utilizing AWS Lambda to inform users of significant Bitcoin price trends promptly.

**Useful Resources**

- [AWS Lambda Documentation](https://docs.aws.amazon.com/lambda/latest/dg/welcome.html)  
- [Amazon EventBridge Documentation](https://docs.aws.amazon.com/eventbridge/latest/userguide/what-is-amazon-eventbridge.html)  
- [Amazon S3 Documentation](https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html)  
- [AWS QuickSight Documentation](https://docs.aws.amazon.com/quicksight/latest/userguide/welcome.html)

**Is it Free?**

AWS Lambda provides a generous free tier that includes 1 million free requests and 400,000 GB-seconds of compute time per month. However, beyond these limits, charges will apply. AWS S3 and other integrated AWS services may also incur costs beyond their free tiers.

**Python Libraries / Bindings**

- **boto3**: The AWS SDK for Python, allowing interaction with AWS services including Lambda, S3, and EventBridge services.  
    
  - Installation: `pip install boto3`  
  - Documentation: [boto3 Documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)


- **Websockets**: Used for establishing WebSocket connections to receive live Bitcoin pricing data.  
    
  - Installation: `pip install websockets`  
  - Documentation: [Websockets Documentation](https://websockets.readthedocs.io/en/stable/)


- **Pandas**: For data manipulation and analysis, specifically useful for handling time series data.  
    
  - Installation: `pip install pandas`  
  - Documentation: [Pandas Documentation](https://pandas.pydata.org/docs/)


- **NumPy**: Provides support for large multi-dimensional arrays and matrices, and a collection of mathematical functions to operate on these arrays.  
    
  - Installation: `pip install numpy`  
  - Documentation: [NumPy Documentation](https://numpy.org/doc/stable/)

### **Amazon QuickSight**

**Title**: Real-time Bitcoin Analytics with Amazon QuickSight

**Difficulty**: 1 (easy)

**Description** Amazon QuickSight is a business analytics service that enables you to deliver insights to everyone in your organization. As a fast, cloud-powered BI service, it allows you to easily create and publish interactive dashboards that include machine learning insights. This project involves using Amazon QuickSight to visualize real-time Bitcoin price data.

**Describe Technology**

- **Overview**: Amazon QuickSight is a scalable, serverless BI service that integrates with other AWS services to offer interactive data visualization and insights. It features an easy-to-use interface and various visualization options, such as line graphs, pie charts, and geographical maps.  
- **Data Integration and Preparation**: Users can connect QuickSight to a variety of data sources, including AWS Data Lakes, S3, Athena, and RDS. QuickSight offers capabilities for data transformation and preparation through features like SPICE (Super-fast, Parallel In-memory Calculation Engine) for faster analytics performance.  
- **Analytics and Visualization**: QuickSight supports machine learning insights, anomaly detection, and forecasts within its visualizations, helping users derive deeper insights directly from their data.

**Describe the Project**

- **Objective**: Create a real-time analytics dashboard to visualize Bitcoin price data using Amazon QuickSight.  
- **Data Source**: Fetch real-time Bitcoin price data from a public API, such as CoinGecko.  
- **Data Storage**: Store the gathered data in an AWS S3 Bucket, which will serve as the data source for QuickSight.  
- **Data Visualization**: Use Amazon QuickSight to connect to the S3 bucket, and set up data visualizations like time series line charts to display price changes over time.  
- **Analysis and Insights**: Leverage QuickSight’s built-in analytics capabilities to add anomaly detection and forecast the future price of Bitcoin based on historical data. This allows for deeper insights and an interactive experience.  
- **Expected Outcomes**: A functioning QuickSight dashboard that automatically updates with real-time data and displays meaningful insights and trends regarding Bitcoin prices over time.

**Useful Resources**

- [Amazon QuickSight Documentation](https://docs.aws.amazon.com/quicksight/index.html)  
- [QuickSight Learning Path](https://aws.amazon.com/training/path-analytics/)  
- [CoinGecko API Documentation](https://www.coingecko.com/en/api/documentation)

**Is it Free?** Amazon QuickSight offers a free trial, but there are costs associated with its continued use after the trial period. Additional AWS costs may apply for S3 storage and other services used.

**Python Libraries / Bindings**

- **Boto3**: To interact with AWS services and manage S3 for data storage.  
  - Install using `pip install boto3`.  
  - [Boto3 Documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)  
- **Requests**: For fetching data from APIs, such as CoinGecko.  
  - Install using `pip install requests`.

### **Amazon RDS**

**Title**: Real-Time Bitcoin Data Processing using Amazon RDS

**Difficulty**: Medium

**Description**: Amazon RDS (Relational Database Service) is a managed SQL database service provided by Amazon Web Services. It facilitates setting up, operating, and scaling a relational database in the cloud. RDS offers support for several database engines, including MySQL, PostgreSQL, Oracle, Microsoft SQL Server, and Amazon Aurora, making it versatile for a variety of applications. In this project, students will utilize Amazon RDS to store and process real-time Bitcoin price data. The focus will be on understanding how RDS works, setting up a database instance, creating tables, and implementing a Python-based solution to ingest and analyze Bitcoin data.

**Describe technology**:

- **Amazon RDS Features**:  
    
  - Fully managed database service with automated backups, software patching, and hardware provisioning.  
  - Supports a variety of database engines.  
  - Offers scalability with the ability to adjust compute and storage capacity.  
  - Provides built-in security features, including encryption at rest and in transit.  
  - Allows for monitoring and managing database operations using AWS Management Console and AWS CLI.


- **Core Concepts**:  
    
  - **Database Instance**: The primary database environment consisting of resources needed to run a database.  
  - **Security Groups**: Controls access to the database instance.  
  - **Parameter Groups**: Allows customization of database parameters.

**Describe the project**:

- **Objective**: To ingest, store, and analyze real-time Bitcoin prices using Amazon RDS, with a focus on time series analysis.  
    
- **Steps**:  
    
  1. **Set Up an Amazon RDS Instance**:  
       
     - Create an RDS instance with a supported database engine like PostgreSQL or MySQL.  
     - Configure security and parameter groups for optimal access and performance.

     

  2. **Database Design**:  
       
     - Design schema for storing real-time Bitcoin price data.  
     - Implement tables to store raw data, processed data, and results of time series analysis.

     

  3. **Data Ingestion**:  
       
     - Utilize Python to fetch data from a Bitcoin price API (e.g., CoinGecko).  
     - Write scripts to insert real-time data into the RDS instance using Python's database connectivity libraries.

     

  4. **Time Series Analysis**:  
       
     - Implement basic time series analysis methods using Python packages such as Pandas and NumPy.  
     - Analyze trends and patterns in Bitcoin price changes over time.

     

  5. **Visualize Results**:  
       
     - Use Matplotlib or similar libraries to visualize the results of the time series analysis.  
     - Create informative charts that illustrate data trends and insights.

**Useful resources**:

- [Amazon RDS Documentation](https://aws.amazon.com/documentation/rds/)  
- [AWS Python SDK (Boto3) Documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)  
- [Pandas Documentation](https://pandas.pydata.org/docs/)  
- [CoinGecko API](https://www.coingecko.com/en/api)

**Is it free?**: Amazon RDS offers a free tier that allows you to run a single database instance for free, with some limitations. However, additional usage beyond the free tier may incur costs.

**Python libraries / bindings**:

- **Boto3**: The official AWS SDK for Python, used for interfacing with AWS services, including Amazon RDS (install via `pip install boto3`).  
- **psycopg2** or **PyMySQL**: Libraries for connecting to PostgreSQL or MySQL databases from Python (install via `pip install psycopg2-binary` or `pip install pymysql`).  
- **Pandas**: A powerful data manipulation and analysis library that supports time series analysis (install via `pip install pandas`).  
- **Matplotlib**: A plotting library for creating static, interactive, and animated visualizations in Python (install via `pip install matplotlib`).

### **Amazon Redshift**

**Title**: Real-Time Bitcoin Analysis using Amazon Redshift

**Difficulty**: Easy

**Description**  
Amazon Redshift is a fully managed, petabyte-scale data warehouse service offered by Amazon Web Services (AWS). It makes it simple and cost-effective to analyze large volumes of data using SQL and other familiar data processing tools. This project focuses on using Redshift's capabilities to handle real-time data ingestion and processing, specifically geared toward ingesting and analyzing Bitcoin price data. You'll learn the basics of Redshift, its integration within the AWS ecosystem, and how it can be combined with Python for data processing tasks.

**Describe technology**

- **Amazon Redshift**: A cloud-based data warehousing service optimized for online analytical processing (OLAP), allowing for high-performance analytical queries.  
- **Key Features**: Columnar storage, parallel query execution, and automatic backups make Redshift well-suited for querying and analyzing large data sets.  
- **Integration with AWS**: Seamless integration with other AWS services, including S3 for data storage and AWS Lambda for real-time data processing.

**Describe the project**

- **Objective**: Ingest real-time Bitcoin pricing data from a public API like CoinGecko and store it in Amazon Redshift for further analysis.  
- **Steps**:  
  1. **Data Ingestion**: Set up an AWS Lambda function to fetch live Bitcoin pricing data at regular intervals from the API and stream it into Redshift using the COPY command.  
  2. **Data Management**: Use Python scripts to create tables in Redshift for storing the incoming data and applying a suitable schema.  
  3. **Time Series Analysis**: Implement a Python-based time series analysis script to query the Bitcoin price data from Redshift and perform basic analyses such as moving averages or detecting price anomalies.  
  4. **Visualization**: Optionally, extract analyzed results and visualize them using basic Python libraries like matplotlib or seaborn to gain insights.

**Useful resources**

- [Amazon Redshift Documentation](https://docs.aws.amazon.com/redshift/index.html)  
- [AWS Lambda Documentation](https://docs.aws.amazon.com/lambda/)  
- [CoinGecko API Documentation](https://www.coingecko.com/en/api)  
- Tutorials on using boto3 with AWS services for programmatic interactions.

**Is it free?**  
Amazon Redshift offers a free trial that allows new users to explore its capabilities without any cost. However, beyond the free tier, normal usage fees will apply based on the storage and computing resources used.

**Python libraries / bindings**

- **boto3**: The AWS SDK for Python, essential for interacting with AWS services such as Redshift and Lambda (install using `pip install boto3`).  
- **psycopg2**: PostgreSQL adapter for Python, used to connect to Amazon Redshift (install using `pip install psycopg2-binary`).  
- **Requests**: A simple HTTP library for Python, useful to fetch data from APIs (install using `pip install requests`).  
- **Pandas**: For data manipulation and analysis (install using `pip install pandas`).

### **Amazon S3**

**Title**: Analyzing Real-Time Bitcoin Prices with Amazon S3

**Difficulty**: 1 (easy)

**Description**  
Amazon S3 (Simple Storage Service) is a scalable, high-speed web-based cloud storage service designed for online data and backup archiving. It is part of Amazon Web Services (AWS) and provides object storage through a web service interface. The fundamental purpose of Amazon S3 is to provide storage infrastructure on a pay-as-you-go basis, offering reliability, fast data access, and secure solutions for a wide variety of applications, including big data analytics. This project introduces students to the basic functionalities of Amazon S3, focusing on data storage, retrieval, and management through Python, with an emphasis on real-time Bitcoin data ingestion and analysis.

**Describe Technology**

- **Amazon S3 Features**:  
    
  - Secure, durable, and scalable storage solution allowing organizations to store and retrieve any volume of data anytime from anywhere.  
  - Uses REST API for easy data access and integration with other AWS services.  
  - Capable of handling large-scale data analytics tasks by integrating seamlessly with AWS services like AWS Lambda, AWS Glue, or Amazon EMR.  
  - Supports various data import methods including direct data transfers, AWS DataSync, and managed file transfers for large datasets.


- **Example Use Cases**:  
    
  - Data backup and archival  
  - Data lakes for big data analytics  
  - Static website hosting  
  - Media storage and distribution

**Describe the Project**

In this project, students will utilize Amazon S3 to implement a simple data pipeline for ingesting and analyzing real-time Bitcoin prices:

- **Objective**: Ingest real-time Bitcoin price data from a public API and store the data efficiently in Amazon S3 for future analysis.  
- **Steps**:  
  1. **Set Up Amazon S3**: Create an Amazon S3 bucket to store Bitcoin price data.  
  2. **Ingest Data**: Write a Python script to call a Bitcoin API (e.g., CoinGecko) and capture real-time price data periodically using a simple scheduling approach like time.sleep().  
  3. **Store Data**: Save the ingested data into a CSV file format and upload it to the S3 bucket. Students will use the Boto3 library to interact with S3 for uploading and accessing files.  
  4. **Analyze Data**: Implement a basic time series analysis in Python using libraries such as pandas to process the data retrieved from S3, focusing on trends or moving averages of the Bitcoin prices over time.  
  5. **Visualization**: Create basic plots to visualize the price changes using Python’s matplotlib or seaborn libraries.

**Useful Resources**

- [AWS S3 Documentation](https://docs.aws.amazon.com/s3/index.html)  
- [Boto3 Documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)

**Is it free?**

To use Amazon S3, you need to create an AWS account. Amazon S3 offers a free tier, which includes 5 GB of standard storage space, but it's limited to the first year of service and certain data transfer limits.

**Python Libraries / Bindings**

- **Boto3**: The AWS SDK for Python. It allows you to interact with Amazon S3 programmatically. Install it via `pip install boto3`.  
- **Pandas**: A powerful Python library for data manipulation, especially useful for time series data analysis. Install it via `pip install pandas`.  
- **Matplotlib/Seaborn**: Libraries for data visualization in Python. Matplotlib can be installed via `pip install matplotlib` and Seaborn via `pip install seaborn`.

### **Amazon Sagemaker**

**Title**: Real-Time Bitcoin Price Analysis with Amazon SageMaker

**Difficulty**: 3 (difficult)

**Description**

In this project, students will leverage Amazon SageMaker to ingest and process real-time Bitcoin price data for time series analysis. Amazon SageMaker is a fully managed service that provides every developer and data scientist with the ability to build, train, and deploy machine learning (ML) models quickly. It covers the entire ML workflow, including preparing data, building models, training and tuning them, deploying them to production, and scaling them as needed. This project involves gathering real-time Bitcoin price data from an external API, performing time series analysis, and deploying a predictive model using SageMaker.

**Describe technology**

- **Amazon SageMaker**: A comprehensive managed service that enables users to quickly create and manage machine learning models. It provides tools for data preparation, feature engineering, model training, tuning, and deployment.  
- **Notebooks**: SageMaker offers Jupyter Notebooks for data exploration and model building, integrating with data sources within AWS.  
- **Built-in Algorithms**: SageMaker includes a range of built-in algorithms optimized for speed, scale, and accuracy.  
- **Real-Time Endpoints**: Deploy models as scalable API endpoints.

**Describe the project**

- **Data Ingestion**: Utilize Python's `requests` library to fetch real-time Bitcoin price data from an API, such as CoinGecko. Use Amazon Kinesis Data Streams to manage the data ingestion pipeline for streaming data into Amazon S3.  
- **Data Storage**: Store ingested data in Amazon S3 for further processing and analysis.  
- **Data Processing**: Use SageMaker Processing Jobs with pre-loaded Jupyter Notebooks containing Pandas and NumPy for dataset cleaning and pre-processing.  
- **Time Series Analysis**: Implement a time series forecasting model using advanced built-in algorithms like DeepAR in SageMaker. Complement it with statistical methods in Python for validation.  
- **Model Training and Evaluation**: Train the model, tune hyperparameters, and evaluate accuracy using SageMaker’s built-in capabilities.  
- **Model Deployment**: Deploy the trained model as a SageMaker endpoint to predict future Bitcoin prices, demonstrating how time series forecasting can be applied to financial datasets.  
- **Visualization**: Leverage Matplotlib and Seaborn for generating visual insights into the Bitcoin price trend, including forecasting vs actual comparisons.

**Useful resources**

- [Amazon SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/index.html)  
- [DeepAR Forecasting Algorithm](https://docs.aws.amazon.com/sagemaker/latest/dg/deepar.html)  
- [Amazon Kinesis Data Streams](https://docs.aws.amazon.com/kinesis/index.html)

**Is it free?**  
You need an AWS account to use Amazon SageMaker. The AWS Free Tier provides limited free monthly usage of SageMaker and Kinesis, which may be sufficient for small-scale experiments. Exceeding these limits will incur charges.

**Python libraries / bindings**

To effectively implement this project, the following Python libraries will be utilized:

- `boto3`: AWS SDK for Python, enables interaction with Amazon SageMaker, S3, and Kinesis.  
- `Pandas` and `NumPy`: For data manipulation and numerical operations.  
- `Matplotlib` and `Seaborn`: For data visualization.  
- `requests`: To fetch Bitcoin data from a public API.  
- SageMaker SDK: Provides an interface for managing SageMaker resources and kdeploying models.

### **Amazon Snowflake**

**Title**: Real-time Bitcoin Data Processing with Amazon Snowflake

**Difficulty**: 2 (Medium)

**Description**  
This project focuses on leveraging Snowflake, a cloud-based data warehousing solution, for real-time ingestion and processing of Bitcoin price data. By using Python to connect to Snowflake's powerful data platform, students will gain hands-on experience with integrating cloud data warehouses with real-time data streams. The project will cover configuring data ingestion pipelines, processing live data, and performing time series analysis to derive meaningful insights from Bitcoin's price fluctuations.

**Describe Technology**  
Snowflake is a cloud-native, fully managed data warehouse that allows organizations to store and analyze large datasets without managing hardware or software. Key features include:

- **Elastic Scalability**: Snowflake automatically scales to handle increased workloads, allowing for efficient resource usage.  
- **Secure Data Sharing**: It enables seamless data sharing across different accounts while maintaining security and privacy.  
- **Data Lake Integration**: Snowflake can easily integrate with data lakes, making it versatile for various data sources.  
- **Built-in SQL Support**: It supports SQL queries, facilitating an easy transition for users familiar with the language.

**Describe the Project**

- **Data Ingestion**: Use Python to connect to a public Bitcoin price API such as CoinGecko, and fetch real-time data. Load this data into Snowflake using its native connectors.  
- **Data Transformation**: Once data is ingested, use Python's SQL capabilities to create and execute queries that clean and prepare the data for analysis. This includes handling missing values or anomalous entries.  
- **Time Series Analysis**: Implement a basic time series analysis script in Python to explore Bitcoin's price trends. This includes computing moving averages, volatility, or even applying simple forecasting techniques for price prediction.  
- **Reporting**: Use tools like Snowflake's native dashboard, or integrate with third-party visualization libraries in Python, such as Matplotlib or Plotly, to visualize findings.

**Useful Resources**

- [Snowflake Documentation](https://docs.snowflake.com/)  
- [CoinGecko API Documentation](https://www.coingecko.com/en/api)  
- [Python Snowflake Connector Documentation](https://docs.snowflake.com/en/user-guide/python-connector.html)

**Is it free?**  
Snowflake offers a usage-based pricing model, which allows you to pay for what you use. New customers can register for a free trial, which includes credits to explore Snowflake features.

**Python Libraries / Bindings**

- **snowflake-connector-python**: A library to connect and run queries on Snowflake from Python scripts. Install via `pip install snowflake-connector-python`.  
- **requests**: Use this library to access the CoinGecko API for real-time Bitcoin data. Install via `pip install requests`.  
- **pandas**: For data manipulation and analysis, especially for handling tabular data in the project. Install via `pip install pandas`.  
- **matplotlib/plotly**: For creating visualizations of your time series analysis findings. Install via `pip install matplotlib plotly`.

### 

### **Amazon Timestream**

**Title**: Real-time Bitcoin Price Analysis with Amazon Timestream

**Difficulty**: 1 (easy)

**Description**  
Amazon Timestream is a fast, scalable, serverless time series database service by Amazon Web Services (AWS), specifically designed to efficiently store and process time series data. With its ability to scale based on the volume of incoming data and its serverless nature, users can perform real-time analytics without the need for managing infrastructure. This project involves ingesting real-time Bitcoin price data from a public API, storing it in Amazon Timestream, and performing basic time series analysis using Python.

**Describe technology**

- **Amazon Timestream** is purpose-built for time series data, which is data that arrives incrementally and can be timestamped. It is designed to handle trillions of events per day, with the ability to query and analyze data quickly.  
- **Serverless architecture**: No need to provision or manage servers, enabling focus on application logic rather than infrastructure.  
- **Automatic scaling**: Handles varying workloads by automatically adjusting capacity.  
- **Built-in time series functions**: Supports time series-specific queries, like windowed aggregations, to ease data analysis.

**Describe the project**

- **Objective**: Use Amazon Timestream to ingest and query Bitcoin price data.  
- **Step 1**: Set up a data pipeline to fetch Bitcoin price data from a public API, such as CoinGecko, using Python.  
- **Step 2**: Integrate with Amazon Timestream to store the fetched Bitcoin prices. This will involve creating a database and table in Timestream using Python's AWS SDK (boto3).  
- **Step 3**: Use Amazon Timestream's querying capabilities to perform simple time series analyses, such as calculating average price over specific intervals or detecting trends.  
- **Step 4**: Visualize the time series data using basic plotting libraries in Python (like matplotlib) to demonstrate price trends or fluctuations.

**Useful resources**

- [Amazon Timestream Developer Guide](https://docs.aws.amazon.com/timestream/latest/developerguide/what-is-timestream.html)  
- [AWS Boto3 Documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)  
- [Tutorial on using Timestream with Python](https://aws.amazon.com/getting-started/hands-on/analyze-time-series-data-amazon-timestream/)

**Is it free?**  
Amazon Timestream offers a free tier that covers the basic usage levels suitable for this project. Ensure you check the latest AWS pricing to stay within any limits.

**Python libraries / bindings**

- **boto3**: Essential for creating and managing Amazon Timestream resources. Install with `pip install boto3`.  
- **requests**: Use this to fetch real-time data from APIs. Install with `pip install requests`.  
- **matplotlib**: For visualizing time series data in Python. Install with `pip install matplotlib`.

### **Amazon X-Ray**

**Title**: Predictive Bottleneck Detection in Real-Time Data Pipelines Using AWS X-Ray  
**Difficulty**: 3 (Hard)  
**Description**  
This project combines **data engineering** and **data science** to build a real-time data pipeline instrumented with AWS X-Ray, analyze trace data for performance bottlenecks, and predict future failures using machine learning. Students will design a fault-tolerant pipeline, trace its execution, and build a predictive model to anticipate system degradation.

**Describe Technology**

- **AWS X-Ray**:  
  - Traces requests across distributed systems (APIs, microservices, serverless functions).  
  - Captures metadata like latency, errors, and dependencies.  
  - **Relevance**: Critical for debugging data pipelines (e.g., ETL workflows, streaming systems).  
- **AWS Services Integration**:  
  - Use X-Ray with AWS Lambda (serverless), Kinesis (streaming), and S3 (storage) to mirror real-world data engineering workflows.


**Describe the Project**

1. **Build a Real-Time Data Pipeline** (Data Engineering Focus):  
-   
  - Create a Python-based pipeline that:  
    - Ingests streaming data (e.g., IoT sensor data, stock prices) via Kinesis Data Streams.  
    - Processes data with AWS Lambda (e.g., filtering, aggregation).  
    - Stores results in S3 and a DynamoDB table.  
  - Deploy using AWS CloudFormation or CDK for infrastructure-as-code.


2. **Instrument with AWS X-Ray** (Observability Engineering):  
   - Use the `aws-xray-sdk` to trace:  
     - End-to-end latency of data from Kinesis → Lambda → S3.  
     - Errors in Lambda functions (e.g., failed transformations).  
   - Annotate traces with custom metadata (e.g., data volume, processing time).

3. **Analyze Pipeline Performance** (Data Science Focus):  
   - Use `boto3` to query X-Ray trace data and export it to a Pandas DataFrame.  
   - Engineer time-series features:  
     - Rolling average latency.  
     - Error rates per Lambda function.  
     - Data throughput per shard (Kinesis).  
   - Train a model (e.g., LSTM or Prophet) to predict future bottlenecks (e.g., latency spikes, Lambda timeouts).

   

4. **Automated Alerting & Visualization**:  
   - Trigger AWS SNS alerts when predicted latency exceeds thresholds.  
   - Build a dashboard with `Plotly Dash` showing real-time traces vs. predictions.

**Useful Resources**

- [AWS X-Ray Developer Guide](https://docs.aws.amazon.com/xray/)  
- [AWS X-Ray Python SDK Documentation](https://docs.aws.amazon.com/xray-sdk-for-python/latest/reference/)  
- [Flask X-Ray Integration Tutorial](https://aws.amazon.com/blogs/devops/instrumenting-flask-applications-with-aws-x-ray/)

**Is it free?**  
AWS X-Ray offers a free tier (100,000 traces/month), but costs apply for large-scale usage.

**Python Libraries / Bindings**

- `aws-xray-sdk` for tracing Python applications.  
- `boto3` to fetch and analyze X-Ray trace data.  
- `flask` or `fastapi` for building microservices.  
- `pandas`/`numpy` for time series aggregation.  
- `matplotlib`/`seaborn` for visualization.

### **Amundsen**

**Title**: Implementing Bitcoin Price Alert System using Amundsen

**Difficulty**: 1 (easy)

**Description**  
Amundsen is an open-source data discovery and metadata engine introduced by Lyft. It helps data scientists, analysts, and engineers increase productivity by making the data they need easy to find, understand, and trust. This project focuses on leveraging Amundsen to catalog and discover Bitcoin price data and using basic Python packages to explore real-time data analysis tasks.

**Describe technology**

- Amundsen is built on top of various components like Apache Atlas for metadata storage, Elasticsearch for search, and Neo4j for graph data stored.  
- It provides an intuitive user interface for data users to search for datasets, explore associated metadata, and understand lineage and usage.  
- Key features include data discovery via search, lineage view to understand data dependencies, and a profiler for table quality and freshness.

**Describe the project**

- This project aims to establish a Bitcoin Price Alert System using Amundsen and basic Python packages.  
- First, students will ingest Bitcoin price data from a public API like CoinGecko and store it in a database such as SQLite.  
- They will then configure Amundsen to catalog the ingested data, making it easily searchable and discoverable.  
- Students will create a simple Python program to monitor real-time price changes using the ingested data and trigger alerts for significant price movements, using the Amundsen catalog to track data quality and freshness.  
- Finally, students will visualize time series data and explore historical price trends to understand long-term patterns.

**Useful resources**

- [Amundsen GitHub Repository](https://github.com/amundsen-io/amundsen)

**Is it free?**   
Yes, Amundsen is open-source and free to use.

**Python libraries / bindings**

- Amundsen Libraries: Client libraries to interact with Amundsen components for metadata and data cataloging tasks.  
- Requests: For pulling real-time data from APIs like CoinGecko.  
- SQLite3: For storing and managing ingested data.  
- Pandas: For data manipulation and analysis.  
- Matplotlib or Plotly: For visualizing time series data and patterns.

### **Ansible**

Title: Real-Time Bitcoin Price Analysis with Ansible

**Difficulty**: 1 (easy)

**Description**

Ansible is an open-source automation tool used for configuration management, application deployment, and task automation. It enables DevOps professionals and data scientists to automate complex processes in an efficient and repeatable way. This project involves using Ansible to automate the ingestion and processing of real-time Bitcoin price data and perform basic time series analysis using Python.

**Describe Technology**

- Ansible employs a simple, human-readable YAML syntax for its playbooks, which define automation jobs.  
- It is agentless, meaning no additional software needs to be installed on target machines.  
- Ansible leverages SSH for secure and efficient communication, making it ideal for managing multiple servers and automation tasks.  
- The tool supports complex workflows via tasks and roles, allowing for flexibility and reusability.

**Describe the Project**

- **Objective**: Automate the process of ingesting Bitcoin price data from a public API, using Ansible to deploy the required infrastructure, and analyze the data to identify trends over time.  
- **Steps**:  
  1. **Setup Environment**: Use Ansible to automate the installation of necessary Python libraries and dependencies required for data ingestion and analysis on a remote server (or local environment).  
  2. **Data Ingestion**: Implement a Python script that regularly fetches Bitcoin price data from an API like CoinGecko or CoinMarketCap. Use a cron job or similar scheduling tool to automate the data fetch process.  
  3. **Data Processing**: Use Python to clean and preprocess the data. Basic operations could include converting time fields, handling missing data, and normalizing price data.  
  4. **Time Series Analysis**: Conduct a simple moving average analysis or other basic time series analysis on the Bitcoin price data to identify trends.  
  5. **Automation with Ansible**: Write Ansible playbooks to automate the deployment and scheduling of the Python scripts, ensuring the data ingestion and processing tasks run periodically without manual intervention.

**Useful Resources**

- [Ansible Documentation](https://docs.ansible.com/)  
- [CoinGecko API Documentation](https://www.coingecko.com/en/api)  
- Python basics for time series analysis

**Is it free?**

Yes, Ansible is an open-source tool and free to use. However, deploying on remote servers might incur costs depending on your hosting provider.

**Python Libraries / Bindings**

- `requests`: For interacting with APIs and fetching JSON data. Install via `pip install requests`.  
- `pandas`: For data manipulation and analysis, especially useful for handling time series data. Install via `pip install pandas`.  
- `matplotlib` or `seaborn`: For data visualization to plot price trends over time. Install via `pip install matplotlib seaborn`.  
- `ansible`: To write and execute Ansible playbooks. Install via `pip install ansible`.

### 

### **Anthropic**

**Title**: Real-Time Bitcoin Transaction Anomaly Detection with Anthropic Claude  
**Difficulty**: 3 (difficult)

**Description**  
Build an explainable AI system to detect anomalous Bitcoin transactions using Anthropic's constitutional AI models. The system will process blockchain data in real-time, provide human-readable explanations for flagged transactions, and integrate with alerting systems while maintaining auditability.

**Describe Technology**

- **Anthropic Claude**: AI system focused on:  
  - Interpretable anomaly detection through constitutional AI principles  
  - Natural language explanations for model decisions  
  - Real-time processing capabilities via API  
  - Built-in safety constraints for financial applications  
  - Support for structured data parsing through system prompts

**Describe the Project**

1. **Data Pipeline**:  
-   
  - Stream raw blockchain data from Blockchair API (1000+ tps)  
  - Preprocess transactions with PySpark:  
    - Extract features (value, fee, input/output ratios, temporal patterns)  
    - Windowed aggregation (1min/5min buckets)  
  - Handle imbalanced data (99% normal transactions) through synthetic minority oversampling  
-   
2. **Anomaly Detection**:  
-   
  - Implement two-stage detection:  
    1. Rule-based filtering (known attack patterns)  
    2. Anthropic model analysis with chain-of-thought prompting for example,:

```py
prompt = f"""Analyze this Bitcoin transaction {tx_data}. 
Consider: - Deviation from account history patterns
          - Network-wide statistical baselines
          - Common attack signatures"""
```

  - Generate natural language reports for flagged transactions

2. **Temporal Analysis**:  
   - Track anomaly rates with exponential weighted moving averages  
   - Detect coordinated attacks through cross-account pattern matching  
   - Predict future anomaly clusters using Prophet time series forecasting

3. **Operational System**:  
   - Build alert workflows with priority scoring (Slack/email/PagerDuty)  
   - Create Dash dashboard showing:  
     - Real-time transaction map  
     - Model confidence distributions  
     - Explanation audit trails  
   - Deploy on AWS EMR with auto-scaling for spike handling

**Useful Resources**

- [Anthropic Constitutional AI Paper](https://www.anthropic.com/constitutional-ai)  
- [Blockchain Analytics Toolkit](https://blockchain.com/explorer/api)  
- [Financial Anomaly Detection Patterns](https://arxiv.org/abs/2207.10418)

**Is it free?**

- Anthropic: Free trial credits available, production usage requires API payment  
- Blockchair: Free tier (1000 requests/day)  
- Apache Spark/Dash: Open-source


**Python Libraries / Bindings**

- `anthropic`: Official SDK for Claude models \- install with `pip install anthropic`  
- `pyspark`: Distributed data processing \- `pip install pyspark`  
- `dash`: Interactive dashboard \- `pip install dash`  
- `prophet`: Time series forecasting \- `pip install prophet`  
- `slack-sdk`: Alert integration \- `pip install slack-sdk`  
- `imbalanced-learn`: Handle data skew \- `pip install imbalanced-learn`


### 

### **Anthropic MCP**

Title: Real-Time Bitcoin Price Monitoring with Anthropic MCP

Difficulty: 2 (Medium)

**Description**

- The Anthropic MCP (Machine Learning and Conversational Platform) provides an environment where students can explore the processing and analysis of complex data systems, focusing on real-time ingestion and processing tasks. In this project, you will have the opportunity to combine Python programming with data analysis skills to work on a practical example involving the real-time monitoring of Bitcoin prices.  
- The goal of the project is to build a system that can ingest, process, and analyze real-time Bitcoin prices, leveraging time series analysis to extract insights and generate alerts for significant price changes, trends, or patterns.

**Describe technology**

- Anthropic MCP is designed for automation and enhancing data-driven decision-making using machine learning and natural language processing techniques.  
- It provides tools for monitoring and analyzing complex data streams, which can be used to handle real-time data ingestion tasks.  
- The platform supports integration with external data sources and APIs, allowing for easy connectivity and real-time data processing.  
- Anthropic MCP enables customized model creation and deployment for automated decision-making based on real-time data inputs.

**Describe the project**

- The project involves setting up a real-time data pipeline using Python and the Anthropic MCP to fetch Bitcoin price data from a robust API like CoinGecko or Bitcoin.org.  
- Students will create a Python script using basic libraries like requests or AIOHTTP to manually pull in real-time price data every few minutes.  
- The script will interface with the Anthropic MCP platform to ingest data, enabling automated alerts and analytics if significant price changes surpass a pre-defined threshold.  
- Students will implement time series analysis methods to detect price trends, using libraries like Pandas or Statsmodels.  
- Visualization of the data can be achieved through Matplotlib or Plotly, providing graphical insights into price fluctuations over a specified time frame.  
- The project will conclude with a report detailing the analysis process, findings, and potential real-world applications of the system.

**Useful resources**

- Official API documentation for CoinGecko or other cryptocurrency data providers for accessing real-time Bitcoin price data.  
- Anthropic MCP documentation for detailed guidance on setting up analyses and automated systems.  
- Python data science libraries documentation like Pandas and Statsmodels for time series analysis methods.  
- Articles or papers on real-time data processing techniques for further insights into best practices and strategies.

**Is it free?**

- Access to Anthropic MCP may require a registration or licensing fee depending on its usage model.  
- Many of the Python libraries used in this project, like requests, Pandas, Statsmodels, Matplotlib, and Plotly, are open-source and available for free.

**Python libraries / bindings**

- requests or AIOHTTP: Used for fetching real-time Bitcoin price data from APIs.  
- Pandas: Essential for data manipulation and time-series analysis.  
- Statsmodels: Provides advanced statistical tools for time series analysis.  
- Matplotlib or Plotly: For creating visualizations of Bitcoin price trends and patterns.  
- Anthropic API bindings (if available): To interface with the Anthropic MCP for advanced analytics and automation features.

### **Apache Beam**

**Title**: Real-Time Bitcoin Price Analysis Using Apache Beam  
**Difficulty**: 1 (Easy)

**Description**  
This project involves utilizing Apache Beam for ingesting and processing real-time Bitcoin price data. Apache Beam is a unified model for defining both batch and streaming data-parallel processing pipelines. As part of this beginner project, students will gather real-time Bitcoin data from a public API and perform basic time series analysis using Python.

**Describe technology**

- Apache Beam: An open-source unified programming model to define and execute data processing pipelines, including ETL and batch/streaming.  
- It abstracts away the details of the execution engines, enabling decoupling of logic and execution.  
- Focus is on simplicity and ease of use, making it suitable for introducing new users to big data systems.

**Describe the project**

- Set up an environment to run Apache Beam locally using the Python SDK.  
- Use Python scripts to fetch real-time Bitcoin price data from an open-source API like CoinGecko.  
- Construct an Apache Beam pipeline to stream and process the Bitcoin data. This includes steps like:  
  - Fetching data every minute.  
  - Extracting relevant fields such as timestamp and price.  
  - Performing basic calculations to analyze price trends over a specific time window (e.g., compute average price over 10-minute intervals).  
- Output the processed data to a local file system, Google Cloud Storage, or any other supported storage backend.  
- Visualize the trend using a simple plot in Python.

**Useful resources**

- [Apache Beam Documentation](https://beam.apache.org/documentation/)  
- [Apache Beam Python SDK Quickstart](https://beam.apache.org/get-started/quickstart-py/)  
- [CoinGecko API Documentation](https://www.coingecko.com/en/api)

**Is it free?**  
Yes, Apache Beam is open-source and free to use. However, operational costs might incur if you choose to deploy the pipeline on a cloud data processing service such as Google Cloud Dataflow.

**Python libraries / bindings**

- `apache-beam`: Install via `pip install apache-beam`. It provides the necessary tools to define and execute Beam data processing pipelines in Python.  
- `requests`: Utility library for making HTTP requests to fetch Bitcoin price data from the API (install via `pip install requests`).  
- `matplotlib` or `seaborn`: For creating simple plots to visualize the processed time series data (install via `pip install matplotlib` or `seaborn`).

### **Apache Flink**

**Title**: Ingest and Analyze Bitcoin Prices Using Apache Flink  
**Difficulty**: Medium (2)

**Description**  
Apache Flink is an open-source stream processing framework for distributed, high-performing, always-available, and accurate data streaming applications. It is designed to process unbounded data streams with low latency and guarantees exactly-once state consistency. Flink's powerful and flexible windowing mechanism allows real-time data processing, making it a strong fit for financial market applications like analyzing cryptocurrency price movements.

**Describe technology**

- Apache Flink is a stream processing framework that supports real-time processing of data streams.  
- It offers features such as event time processing, batch processing, and stateful computations.  
- Built-in support for complex event processing (CEP), allowing users to detect patterns in data streams.  
- Integrates with various data sources and sinks like Apache Kafka, Kinesis, Elasticsearch, and more.  
- Flink's high throughput and low latency make it suitable for complex analytical tasks.

**Describe the project**  
The project focuses on creating a Flink job to ingest and process real-time Bitcoin price data from a public API such as CoinGecko or Binance. Students will:

- Set up an Apache Flink environment locally or via a cloud service supporting Flink.  
- Develop a Flink streaming job to consume bitcoin price data from an API.  
- Implement time windowing to perform real-time time series analysis on Bitcoin price data, such as calculating moving averages or identifying price trends.  
- Use Python with Apache Flink's PyFlink API for crafting the streaming job logic.  
- Explore Flink's state management abilities to track and update the state of Bitcoin prices over time.  
- Optionally, integrate the project with a visualization tool like Grafana for real-time data visualization and analysis results.

**Useful resources**

- [Official Apache Flink Documentation](https://flink.apache.org/)  
- [PyFlink API Documentation](https://nightlies.apache.org/flink/flink-docs-stable/docs/dev/python/getting-started/)  
- [Real-time stream processing with PyFlink](https://flink.apache.org/news/2020/05/21/news-flink-python.html)

**Is it free?**  
Yes, Apache Flink is open-source software available under the Apache License 2.0, which is free to use and modify. However, associated costs might arise if using a cloud service to run Flink jobs.

**Python libraries / bindings**

- **PyFlink**: The Python API for Apache Flink. It provides bindings for developing Flink jobs using Python and is essential for this project. You can install PyFlink via `pip install apache-flink`.  
- **Requests**: Helps in making HTTP requests to access the Bitcoin price API. Install it using `pip install requests`.  
- **Matplotlib/Plotly** (optional): For data visualization, you might want to use libraries like Matplotlib or Plotly. Install via `pip install matplotlib` or `pip install plotly`.

### **Apache Hadoop**

**Title**: Real-time Bitcoin Data Processing with Apache Hadoop

**Difficulty**: 1 (easy)

**Description**  
Apache Hadoop is an open-source software framework used for distributed storage and processing of large datasets across clusters of computers using simple programming models. It is designed to scale up from a single server to thousands of machines, with a high degree of fault tolerance.

In this project, students will explore the fundamental concepts of Apache Hadoop, including its core components such as Hadoop Distributed File System (HDFS) and MapReduce. The aim is to use Hadoop to ingest real-time Bitcoin price data and perform basic time series analysis using Python.

**Describe technology**

- **Hadoop Distributed File System (HDFS)**: A distributed file system that provides high-throughput access to application data.  
- **MapReduce**: A programming model for large-scale data processing, consisting of Map (transform data) and Reduce (aggregate data) tasks.  
- **YARN**: Yet Another Resource Negotiator, which handles resource management and job scheduling in Hadoop clusters.  
- **Hadoop Ecosystem**: Complements include tools like Hive (data warehousing) and Pig (scripting for data transformation).

**Describe the project**

- **Objective**: Ingest and process Bitcoin price data using Apache Hadoop, and conduct a basic time series analysis.  
- **Step 1**: Set up a small Hadoop cluster using Apache Hadoop on local machines or in a cloud-based environment.  
- **Step 2**: Use a public API, such as CoinGecko, to periodically fetch real-time Bitcoin price data and store it in HDFS.  
- **Step 3**: Implement a MapReduce job using Python to process the stored Bitcoin data. The job should perform a simple analysis, such as calculating the moving average of Bitcoin prices over specific time intervals.  
- **Step 4**: Present findings and visualizations of the time series analysis using Python plotting libraries like matplotlib or seaborn.

**Useful resources**

- [Apache Hadoop Official Documentation](https://hadoop.apache.org/docs/)  
- [Apache Hadoop Setup Guide](https://hadoop.apache.org/docs/stable/hadoop-project-dist/hadoop-common/ClusterSetup.html)  
- [Intro to MapReduce on Hadoop](https://hadoop.apache.org/docs/stable/hadoop-mapreduce-client/hadoop-mapreduce-client-core/MapReduceTutorial.html)  
- [CoinGecko API Documentation](https://coingecko.com/en/api)

**Is it free?**  
Yes, Apache Hadoop is an open-source framework and free to use. Note that using cloud services may incur costs.

**Python libraries / bindings**

- **Hadoop Streaming**: A utility to create and run Map/Reduce jobs with any executable or script as the mapper and/or reducer. Use this for writing Hadoop jobs in Python.  
- **matplotlib**: A plotting library for the Python programming language to create static, interactive, and animated visualizations.  
- **requests**: A simple HTTP library for Python for making API requests to fetch Bitcoin data.

This project gives students hands-on experience with Hadoop's capabilities and the basics of working with real-time data ingestion for time series analysis.

### **Apache Kafka**

**Title**: Real-Time Bitcoin Price Analysis using Apache Kafka  
**Difficulty**: 1 (easy)

**Description**  
Apache Kafka is an open-source distributed event streaming platform designed to handle real-time data feeds with high throughput and low latency. It is commonly used for building real-time streaming data pipelines and applications that require data integration across different sources.

In this easy project, students will use Apache Kafka to ingest real-time Bitcoin price data from a public API and process it for a simple time series analysis. This project will be completed over one week and offers practical experience with real-time data ingestion and processing.

**Describe technology**

- **Apache Kafka**: A distributed publishing and subscribing messaging system designed to scale to handle multiple producers and consumers simultaneously.  
  - **Core components**: Topics (a stream of records), Producers (writing data to topics), Consumers (reading data from topics), Brokers (Kafka server instances), and Zookeepers (coordinate brokers).  
  - **Key features**:  
    - Reliability and fault-tolerance through distributed architecture.  
    - Scalability allowing it to manage extensive data feeds.  
    - Durability by persisting records on disk.  
    - High throughput, suitable for real-time processing.

**Describe the project**

- **Objective**: Ingest real-time Bitcoin price data and conduct a basic time series analysis using Apache Kafka and Python.  
- **Steps**:  
  1. **Setup Apache Kafka**:  
     - Install Apache Kafka on your local machine or use a managed Kafka service.  
     - Create a Kafka topic  ("bitcoin\_prices") for ingesting real-time Bitcoin price data.  
  2. **Data Ingestion**:  
     - Use a simple Python script to act as a producer, fetching real-time price data from a public Bitcoin API (such as CoinGecko) and sending it to the Kafka topic.  
  3. **Data Processing**:  
     - Create a Python consumer script to consume messages from the Kafka topic.  
     - Perform basic transformations, such as aggregating prices over fixed intervals (e.g., average price per minute).  
  4. **Time Series Analysis**:  
     - Store the aggregated data in a local file or database.  
     - Use basic Python statistical packages like Pandas to analyze the time series data, calculate metrics (e.g., moving averages), and visualize trends.

**Useful resources**

- [Apache Kafka Quickstart](https://kafka.apache.org/quickstart)  
- [CoinGecko API Documentation](https://coingecko.com/en/api)  
- [Pandas: Python Data Analysis Library](https://pandas.pydata.org/docs/)

**Is it free?**  
Yes, Apache Kafka is open-source and free to use. However, hosting Kafka might incur costs depending on the infrastructure used (e.g., cloud services).

**Python libraries / bindings**

- **Kafka-Python**: A Python client for the Apache Kafka platform `pip install kafka-python`  
- **Requests**: To fetch real-time Bitcoin data from APIs  `pip install requests`  
- **Pandas**: For data analysis and manipulation `pip install pandas`

### **Apache Spark**

Title: Real-time Bitcoin Price Analysis with Apache Spark

**Difficulty**: 1 (easy)

**Description**  
Apache Spark is an open-source unified analytics engine for large-scale data processing. It provides high-level APIs in Java, Scala, Python, and R, as well as a rich set of libraries, including MLlib for machine learning, GraphX for graph processing, and Structured Streaming for real-time data processing. Spark is designed for fast computation and can handle batch and real-time data with ease.

In this project, students will learn how to use Apache Spark to ingest and process real-time Bitcoin pricing data using basic Python packages. The focus will be on performing time series analysis to understand price movements and trends over time.

**Describe Technology**

- **Core Concepts**: Understand the key components of Apache Spark, including RDDs (Resilient Distributed Datasets), transformations, actions, and the Spark SQL module for structured data processing.  
    
- **Structured Streaming**: Leverage Spark's Structured Streaming to process real-time Bitcoin data.  
    
- **Integration with Python**: Use PySpark, the Python API for Spark, to write Spark applications.  
    
- **Ease of Use**: Benefit from Spark’s user-friendly syntax and scalability, making it suitable for processing large datasets with minimal code.

**Describe the Project**

- **Objective**: Ingest real-time Bitcoin prices from a public API and conduct time series analysis using Apache Spark.  
    
- **Steps**:  
    
  1. **Data Ingestion**: Use an API like CoinGecko to fetch real-time Bitcoin prices. Store the data in a temporary storage before processing.  
  2. **Real-time Processing**: Set up a Spark Structured Streaming job to continuously ingest and process data as it arrives.  
  3. **Time Series Analysis**: Implement basic transformations on the data, such as computing moving averages, highlighting peaks, and plotting price trends over time.  
  4. **Visualization**: Use basic Python plotting libraries like Matplotlib to visualize the trends and analyses derived from the data.


- **Outcome**: Gain hands-on experience in setting up and running a real-time data processing pipeline using Spark, with an introduction to time series analysis techniques.

**Useful Resources**

- [Official Apache Spark Documentation](https://spark.apache.org/docs/latest/)  
- [Structured Streaming Programming Guide](https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html)  
- [PySpark API Documentation](https://spark.apache.org/docs/latest/api/python/)

**Is it free?**  
Yes, Apache Spark is open-source and free to use. However, if dealing with larger datasets or collaborative projects, consider a hosted service like Databricks, which may have associated costs.

**Python Libraries / Bindings**

- **PySpark**: The Python API for Apache Spark. Install via `pip install pyspark`.  
- **Requests**: For making API calls to fetch Bitcoin prices. Install using `pip install requests`.  
- **Pandas**: For handling intermediate data frames during data transformations. Install with `pip install pandas`.  
- **Matplotlib**: For visualizing the results of time series analysis. Install using `pip install matplotlib`.

### **Argo Workflows**

Title: Real-Time Bitcoin Data Processing with Argo Workflows

**Difficulty**: 2 (medium)

**Description**

Argo Workflows is an open-source container-native workflow engine for orchestrating jobs on Kubernetes. It manages the execution of complex workflows, particularly in scenarios involving big data processing and machine learning pipelines. In this project, students will leverage Argo Workflows to handle real-time bitcoin price data. The focus will be on utilizing the workflow engine to automate data ingestion and processing tasks, while basic Python packages will support data manipulation and analysis, specifically targeting time series analysis.

**Describe technology**

- **Argo Workflows**:  
  - Designed for Kubernetes, Argo Workflows allows users to define, execute, and monitor multi-step workflows using Kubernetes Custom Resource Definitions (CRDs).  
  - Supports directed acyclic graphs (DAGs) of tasks, which can be executed in parallel or sequentially.  
  - Highly scalable and suitable for automating complex data processing tasks without needing an external scheduler.  
  - Integrates well with other Kubernetes tools and services, capitalizing on Kubernetes’ scalability and reliability.

**Describe the project**

- **Objective**: Implement a real-time data processing pipeline using Argo Workflows to ingest and process bitcoin prices from a live feed.  
    
- **Project Steps**:  
    
  1. **Setup Environment**: Configure a Kubernetes cluster where Argo Workflows will be installed and operated.  
  2. **Define Workflows**:  
     - Create a workflow to continuously ingest bitcoin prices from an API like Coinbase or CoinGecko.  
     - Configure parallel tasks in the workflow: one for ingesting new data and another for cleaning and preparing it for analysis.  
  3. **Processing and Storing Data**:  
     - Use Python scripts to process and transform the price data into a structured format.  
     - Analyze the data to infer trends and patterns over time, focusing on metrics like volatility and moving averages.  
     - Store processed data in a database or file system for further analysis or machine learning models.  
  4. **Real-Time Analysis**:  
     - Implement simple Python functions for time series analysis focusing on trend detection and forecasting.


- **Outcome**: Gain practical experience in orchestrating workflows on Kubernetes while handling real-time market data, which provides valuable insights into building scalable data processing pipelines.

**Useful resources**

- [Argo Workflows Documentation](https://argoproj.github.io/argo-workflows/)  
- [Kubernetes Documentation](https://kubernetes.io/docs/home/)  
- [Coinbase API Documentation](https://developers.coinbase.com/)  
- [CoinGecko API Documentation](https://www.coingecko.com/en/api_documentation)

**Is it free?**

- Argo Workflows itself is open-source and free to use. However, running Kubernetes clusters may incur costs depending on the platform (such as using AWS EKS, GKE, etc.). Many platforms offer free tiers for small-scale, experimental clusters.

**Python libraries / bindings**

- **Requests**: For making HTTP requests to fetch bitcoin data from APIs.  
- **Pandas**: Essential for organizing and analyzing time series data.  
- **NumPy**: Useful for numerical computing needed in data calculations.  
- **Matplotlib/Seaborn**: For plotting and visualizing processed time series data.  
- **Kubernetes Python Client**: For interactions with the Kubernetes cluster, if needed.

### **Avro**

**Title**: Real-Time Bitcoin Price Processing using Apache Avro

**Difficulty**: 2 (medium)

**Description**  
Apache Avro is a data serialization framework developed as part of the Apache Hadoop project. It uses JSON for defining data types and protocols and serializes data in a compact binary format, making it suitable for both batch and streaming data processing. Avro is a versatile data exchange framework, offering a rich data structure and a compact, fast serialization format.

**Describe technology**

- **Compact Serialization**: Avro serializes data in a compact binary format, minimizing storage space and making it efficient for data transmission.  
- **Schema Evolution**: Avro supports schema evolution, allowing data formats to change without requiring applications to be recompiled.  
- **Interoperability**: By using JSON to represent schemas, Avro allows data interchange between programs written in different languages.  
- **Integration**: Avro integrates with big data tools, such as Apache Kafka, Spark, and Hadoop, to process and store large datasets.

**Describe the project**

- **Objective**: To build a system using Apache Avro for ingesting and processing real-time Bitcoin price data to analyze trends using time series analysis.  
- **Steps**:  
  1. **Data Ingestion**: Retrieve Bitcoin price data from a public API (e.g., CoinGecko) every minute.  
  2. **Schema Definition**: Define an Avro schema that represents the Bitcoin price data structure, including fields such as `timestamp`, `price`, `currency`, and `volume`.  
  3. **Serialization**: Serialize incoming data using the defined Avro schema for efficient storage and transport.  
  4. **Integration with Kafka**: Set up an Apache Kafka topic to manage streaming data, using Kafka's Avro support for message serialization.  
  5. **Processing**: Write a Python script using Apache Spark to consume messages from Kafka, deserialize the data using Avro, and perform time series analysis to identify Bitcoin trends.  
  6. **Storing Results**: Store the results of the analysis in a structured format (e.g., CSV or Parquet) for further use.

**Useful resources**

- [Avro Specification](https://avro.apache.org/docs/current/spec.html)  
- [Apache Avro’s GitHub Repository](https://github.com/apache/avro)  
- [Kafka with Avro Integration](https://docs.confluent.io/platform/current/schema-registry/avro.html)  
- [Avro with Spark](https://spark.apache.org/docs/latest/sql-data-sources-avro.html)

**Is it free?**  
Yes, Apache Avro is open-source and free to use.

**Python libraries / bindings**

- **`avro-python3`**: The official Avro Python library for schema creation, serialization, and deserialization. Install via `pip install avro-python3`.  
- **`confluent-kafka-python`**: Provides a Kafka client for Python, supporting integration with Confluent's Schema Registry for Avro serialization. Install via `pip install confluent-kafka`.  
- **`pyspark`**: Used for processing the serialized data in Spark. Install via `pip install pyspark`.

### **Azure SDK for Python**

**Title**: Analyze Real-Time Bitcoin Data with Azure SDK for Python

**Difficulty**: 3 (difficult)

**Description**

The Azure SDK for Python provides developers with a comprehensive suite of client libraries for accessing and utilizing Microsoft Azure cloud resources and services. It enables seamless integration with Azure’s extensive array of cloud services, including data storage, computation, machine learning, and more, through a Pythonic interface. The SDK abstracts the complexity of directly working with cloud APIs and facilitates easy development of cloud-based applications, making it adept for handling scalable real-time data processing tasks.

**Describe technology**

- **Azure SDK for Python**: Offers a collection of libraries that provide access to various Azure services, such as Azure Blob Storage, Azure Synapse Analytics, and Azure Event Hubs. These libraries are designed to be consistent, idiomatic to Python, and user-friendly, enabling developers to interact with Azure resources with less friction.  
- **Core Components**: Explore Azure Event Hubs for ingesting high-throughput real-time data streams, such as live Bitcoin transaction data, and Azure Synapse Analytics for processing and analyzing large volumes of data efficiently.  
- **Azure Authentication**: Use Azure credentials to authenticate and connect to Azure services securely. Understand how to manage your service principal with proper access rights to ensure secure data operations.

**Describe the project**

- **Objective**: Implement a real-time data ingestion and processing system for Bitcoin prices using the Azure SDK for Python.  
- **Ingestion**: Use Azure Event Hubs to establish a connection with a public API (e.g., CoinGecko) for continuously ingesting real-time Bitcoin pricing data.  
- **Storage**: Store the ingested data to Azure Blob Storage for persistent storage and later use, ensuring minimal latency and high availability.  
- **Processing**: Utilize Azure Synapse Analytics to perform a time series analysis on the stored data. This involves transforming raw price data into a structured dataset, performing aggregations, calculating moving averages, and detecting anomalies in Bitcoin price fluctuations over time.  
- **Visualization**: Optionally, leverage Azure Power BI or use Python libraries like matplotlib to visualize the time series analysis results.

**Useful resources**

- [Azure for Python Developers](https://docs.microsoft.com/en-us/azure/developer/python/)  
- [Azure SDK for Python Documentation](https://docs.microsoft.com/en-us/azure/python/)  
- [Quickstart: Create an Event Hub using Python](https://docs.microsoft.com/en-us/azure/event-hubs/event-hubs-python-get-started-send)

**Is it free?**

While Azure offers a free tier with limited usage of certain services, extensive use of Azure Event Hubs, Blob Storage, and Synapse Analytics may incur costs. Students can explore Azure’s free tier or use Azure for Students offers for educational purposes.

**Python libraries / bindings**

- `azure-eventhub`: Use this library to send and receive events from Azure Event Hubs. Install it using `pip install azure-eventhub`.  
- `azure-storage-blob`: To handle interaction with Azure Blob Storage for storing ingested Bitcoin data. Install this library using `pip install azure-storage-blob`.  
- `azure-synapse`: This library helps in conducting analytics and data processing tasks within Azure Synapse. Install using `pip install azure-synapse`.  
- `msrestazure`: For handling Azure authentication workflows. Install using `pip install msrestazure`.

### **BigDL**

**Title**: Real-Time Bitcoin Data Processing with BigDL

**Difficulty**: 2 (medium difficulty)

**Description**  
BigDL is a distributed deep learning library for Apache Spark, enabling scalable data analytics and machine learning tasks directly within a Spark environment. It is particularly beneficial for processing and analyzing large datasets, enabling users to leverage Spark's parallel processing capabilities to train deep learning models efficiently.

This project involves using BigDL to ingest real-time Bitcoin price data, perform time series analysis, and generate predictive insights. Implementing this project will help students gain hands-on experience with BigDL's functionalities and explore how it integrates with Apache Spark for big data processing.

**Describe technology**

- **BigDL Overview**:  
    
  - A powerful library that extends Apache Spark with deep learning capabilities.  
  - Enables distributed training and inference on large-scale datasets by leveraging Spark’s cluster computing infrastructure.  
  - Supports a variety of deep learning applications, offering prebuilt models, feature engineering tools, and a seamless integration with Spark MLlib.


- **Core Functionalities of BigDL**:  
    
  - Data pre-processing and augmentation tools to handle raw input data.  
  - Model creation and management for various types of neural networks.  
  - Advanced model training capabilities, including parallel processing and optimization.  
  - Support for loading pre-trained models and transfer learning.

**Describe the project**

- **Objective**: Implement a system to ingest real-time Bitcoin prices, process the data using BigDL, and perform time series analysis to predict future price trends.  
    
- **Steps**:  
    
  1. **Data Ingestion**: Utilize a public API, such as CoinGecko, to fetch real-time Bitcoin price data.  
  2. **Data Processing**: Leverage Spark DataFrames to perform initial data cleansing and preparation.  
  3. **Model Selection and Training**:  
     - Set up a simple recurrent neural network (RNN) using BigDL for time series prediction.  
     - Train the RNN model with historical Bitcoin price data to learn patterns and trends.  
  4. **Prediction and Analysis**:  
     - Use the trained model to make predictions on future Bitcoin prices.  
     - Visualize the results using basic Python libraries such as Matplotlib for trend analysis.


- **Outcome**: Employ BigDL and Apache Spark to develop a basic predictive analytics model capable of processing and analyzing real-time Bitcoin data.

**Useful resources**

- [BigDL Documentation](https://bigdl.readthedocs.io/)  
- [Apache Spark Documentation](https://spark.apache.org/docs/latest/)  
- [CoinGecko API Documentation](https://www.coingecko.com/en/api)

**Is it free?**  
Yes, BigDL is an open-source library. However, running BigDL applications may require access to Spark clusters, which might incur costs if using cloud services.

**Python libraries / bindings**

- **BigDL**: Install BigDL with `pip install bigdl-spark` to access its deep learning features within a Spark environment.  
- **PySpark**: Utilize PySpark (`pip install pyspark`) for handling large-scale data processing tasks.  
- **Requests**: Use `requests` (`pip install requests`) to interact with external APIs for data ingestion.  
- **Matplotlib**: Visualize the time series predictions using Matplotlib (`pip install matplotlib`).

### **Bokeh**

**Title**: Visualizing Real-Time Bitcoin Prices Using Bokeh

**Difficulty**: 1 (Easy)

**Description**

Bokeh is an interactive visualization library for Python that enables you to create elegant and informative graphics for both the web and output files. It is designed to provide high-performance visual presentation of large datasets in a concise format.

In this project, students will use Bokeh to visualize real-time Bitcoin price data. The goal is to create an interactive dashboard that displays live Bitcoin prices using time series plotting. Students will gain experience in using Bokeh for data visualization and learn the basics of ingesting and processing real-time data from a public API.

**Describe technology**

- Bokeh allows you to build interactive plots, dashboards, and data applications.  
- It supports a variety of charts, including scatter, line, bar, and area plots, with interactive widgets.  
- Bokeh can integrate with other Python libraries such as Pandas for data manipulation, allowing seamless operations on dataframes.  
- It provides real-time streaming capabilities via its server, ideal for live data visualization tasks.  
- Output options include static HTML files or integration with Flask/Django for web applications.

**Describe the project**

1. **Data Ingestion**: Use Python to connect to a public API such as CoinDesk's Bitcoin Price Index API to fetch real-time Bitcoin prices.  
     
2. **Data Processing**: Process the incoming JSON data using basic Python libraries to convert it into a Pandas DataFrame for easier manipulation and analysis.  
     
3. **Visualization Using Bokeh**:  
     
   - Setup a Bokeh server application to handle the live data stream.  
   - Create a simple time series line plot of Bitcoin prices that updates in real-time.  
   - Add interactive features such as date range filtering, hover tooltips to display exact price points, and zoom/pan capabilities.  
   - Customize the plot with themes, labels, and legends to enhance readability and aesthetics.

   

4. **Deployment**: Deploy the Bokeh server locally or, optionally, integrate it into a web application using Flask for a more complete dashboard experience.

**Useful resources**

- [Bokeh Official Documentation](https://docs.bokeh.org/en/latest/)  
- [Pandas Documentation](https://pandas.pydata.org/docs/)  
- [CoinDesk API Documentation](https://www.coindesk.com/coindesk-api)

**Is it free?**

Yes, Bokeh is an open-source library and is free to use. Accessing public APIs like CoinDesk may also be free, but be sure to check any usage limitations or subscription models.

**Python libraries / bindings**

- `bokeh`: For creating the visualization and setting up the server for real-time updates.  
- `pandas`: For data manipulation and preparation.  
- `requests`: For making HTTP requests to retrieve data from the Bitcoin API.

Installation can be done using `pip install bokeh pandas requests`.

### [**Bonobo**](?tab=t.0#bookmark=id.lwsyvf35wbjj)

**Title**: Real-time Bitcoin Data Processing Using Bonobo

**Difficulty**: 2 (Medium)

**Description**:  
Bonobo is a lightweight, easy-to-use ETL (Extract, Transform, Load) framework for Python, enabling you to create simple but effective data pipelines. Designed with simplicity and extensibility in mind, Bonobo offers a versatile toolkit for anyone looking to process and analyze data. In this project, students will gain hands-on experience by implementing a real-time data processing system to ingest Bitcoin price data from a live API and perform time series analysis on this data.

**Describe Technology**:

- Bonobo is a data ETL framework in Python that emphasizes ease of use and flexibility.  
- It allows you to define your ETL processes using simple, reusable Python functions.  
- Bonobo introduces the concept of graphs to configure and execute data workflows.  
- The framework supports a variety of data sources, including APIs, databases, and files.  
- Bonobo pipelines are executed in parallel, allowing for efficient real-time data processing.

**Describe the Project**:

- **Goal**: Ingest Bitcoin price data from a public API (such as CoinGecko) and carry out basic time series analysis using Bonobo.  
    
- **Steps**:  
    
  1. **Ingest Data**: Set up Bonobo to regularly fetch Bitcoin price data from a live API.  
  2. **Transform Data**: Parse and clean the JSON response to extract necessary fields such as timestamp, price, and volume.  
  3. **Load Data**: Store the cleaned data into a suitable file format (e.g., CSV) or a lightweight database (e.g., SQLite) for further analysis.  
  4. **Analyze Data**: Implement basic time series analysis methods such as moving averages, trend detection, and volatility analysis using standard Python libraries.  
  5. **Visualize Results**: Use a Python plotting library (e.g., Matplotlib or Seaborn) to visualize the Bitcoin price trends and analysis results.


- **Outcome**: Students will understand the capabilities of Bonobo and how to handle real-time data pipelines efficiently. They will also gain experience in time series analysis, crucial for financial data applications.

**Useful Resources**:

- [Bonobo Documentation](https://bonobo.readthedocs.io/en/latest/)  
- [CoinGecko API Reference](https://coingecko.com/en/api)  
- [Basic Time Series Analysis in Python](https://towardsdatascience.com/basic-time-series-manipulation-with-pandas-4432afee64ea)

**Is it free?**:

- Yes, Bonobo is an open-source framework and free to use. Most public Bitcoin data APIs, such as CoinGecko, offer free access with limitations based on usage.

**Python Libraries / Bindings**:

- **Bonobo**: Install using `pip install bonobo` to create ETL pipelines quickly.  
- **Requests**: For interacting with the Bitcoin price API, install with `pip install requests`.  
- **Pandas**: Utilize for data manipulation and simple time series operations, install with `pip install pandas`.  
- **Matplotlib/Seaborn**: Use these libraries for visualizing data, install with `pip install matplotlib seaborn`.

By completing this project, students will acquire essential skills in using Bonobo for ETL processes and performing basic time series analysis, which are critical for handling big data systems in real-world scenarios.

### **Boto3**

**Title**: Ingesting Bitcoin Prices Using Boto3

**Difficulty**: 1 (easy)

**Description**  
This project focuses on ingesting and processing real-time Bitcoin price data using Boto3, the AWS SDK for Python. Students will learn how to leverage AWS services to fetch, store, and process data using Boto3's functionalities. By the end of this project, students will have a foundational understanding of Boto3's capabilities and hands-on experience with building a simple data ingestion and processing pipeline.

**Describe technology**

- **Boto3**: Boto3 is the official AWS SDK for Python, enabling developers to interact programmatically with AWS services. It facilitates operations like creating and managing AWS resources, uploading and downloading data to/from S3, and tapping into AWS's computing power offered by services such as Lambda and EC2.  
- Boto3 provides a simple, user-friendly interface for interacting with AWS and is suitable for small tasks and automation scripts within Python.

**Describe the project**

- This project involves using Boto3 to:  
    
  - Fetch real-time Bitcoin price data from a public API, such as CoinGecko.  
  - Store the fetched data in an AWS S3 bucket for durability and easy access.  
  - Use basic Python packages, such as Pandas, to perform simple time series analysis on Bitcoin prices. Students will analyze price trends over given time intervals.


- **Steps**:  
    
  1. Set up your AWS account and configure your Python environment to use Boto3.  
  2. Write a Python script using Boto3 to periodically fetch Bitcoin price data and upload it to an S3 bucket.  
  3. Implement a script to fetch the stored data from S3 and use Pandas to perform basic time series analysis, such as calculating moving averages or identifying price spikes.  
  4. Write a report summarizing your findings and discuss any patterns or insights derived from the data.

**Useful resources**

- Boto3 Documentation: [https://boto3.amazonaws.com/v1/documentation/api/latest/index.html](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)  
- AWS S3 Documentation: [https://docs.aws.amazon.com/s3/index.html](https://docs.aws.amazon.com/s3/index.html)  
- CoinGecko API Documentation: [https://www.coingecko.com/en/api](https://www.coingecko.com/en/api)  
- Pandas Documentation: [https://pandas.pydata.org/pandas-docs/stable/](https://pandas.pydata.org/pandas-docs/stable/)

**Is it free?**

- To complete this project, you will need an AWS account. AWS offers a free tier for S3, but usage beyond the free limits may incur costs.

**Python libraries / bindings**

- **Boto3**: For interacting with AWS S3, available via `pip install boto3`.  
- **Pandas**: For data manipulation and time series analysis, available via `pip install pandas`.

### **Causal-learn**

**Title**: Bitcoin Price Causality Analysis with Causal-Learn  
**Difficulty**: 2 (Medium)

**Description**  
A project to analyze causal relationships in Bitcoin price movements and related market variables (e.g., trading volume, social sentiment) using the `causal-learn` package for time series causal inference.

**Describe technology**

- **`causal-learn`**: A Python library for causal discovery, offering algorithms like Granger causality and constraint-based methods to infer causal links in time series data.  
- Designed to work with observational/time series datasets (e.g., financial data).  
- Integrates with Python’s data stack (`pandas`, `numpy`) for preprocessing and analysis.

**Describe the project**

1. **Data Collection**:  
   - Collect Bitcoin historical data (price, trading volume) and external variables (e.g., Google Trends for "Bitcoin," S\&P 500 index) using APIs like `yfinance` or `Cryptocompare`.  
   - Example: Download hourly/daily Bitcoin price and volume data for the past 2 years.  
2. **Preprocessing**:  
   - Handle missing values, normalize data, and engineer lagged features (e.g., lagged price changes).  
3. **Causal Discovery**:  
   - Use `causal-learn`’s time-series methods to identify causal drivers of Bitcoin price changes.  
   - Example code snippet for Granger causality:  
4. **Interpretation**:  
   - Analyze which variables (e.g., trading volume, S\&P 500\) Granger-cause Bitcoin price changes.  
5. **Validation**:  
   - Validate results against known economic hypotheses (e.g., "trading volume precedes price changes").  
6. **Visualization**:  
   - Plot causal graphs and time series interactions using `matplotlib` or `seaborn`.  
7. **Optional Extension**:  
   - Incorporate sentiment analysis from social media (e.g., Reddit/Twitter) using `textblob` and test its causal impact.

**Useful resources**

- [Causal-Learn Documentation](https://causal-learn.readthedocs.io)  
- [Yahoo Finance API (`yfinance`) Tutorial](https://pypi.org/project/yfinance/)  
- [Paper: "Causal Relationships in Cryptocurrency Markets" (arXiv)](https://arxiv.org/abs/2203.12114)


**Is it free?**  
Yes. `causal-learn`, `yfinance`, and other suggested tools are open-source.

**Python libraries / bindings**

- Core: `causal-learn`  
- Data: `pandas`, `yfinance`, `numpy`  
- Visualization: `matplotlib`, `seaborn`  
- Optional: `textblob` (for sentiment analysis)  
- 

### **CausalImpact**

Title: Analyze Bitcoin Price Impact with CausalImpact

Difficulty: 1 (easy)

Description: CausalImpact is an open-source Python library that allows users to perform causal inference on time series data. It was initially developed by Google and provides a straightforward way to evaluate the effect of an intervention on a time series. This project will guide students through applying CausalImpact to analyze the impact of a major event (e.g., a government ban or a significant legal announcement) on Bitcoin prices. Over the course of a week, students will learn the basics of causal inference, how to frame a hypothesis, and apply CausalImpact to test their hypothesis using real-time Bitcoin price data.

**Describe technology:**

- CausalImpact: This library helps estimate the causal effect of a predictive event on time series data using a Bayesian structural time-series model.  
- It is particularly useful for cases where controlled experiments or randomized trials are not feasible.  
- Students will learn to set up and interpret the model which will ultimately enable them to investigate whether a suspected event has had a significant impact on Bitcoin prices.

**Describe the project:**

- The project will involve using an API like CoinGecko or CryptoCompare to collect real-time Bitcoin price data.  
- Identify a significant event that could plausibly impact Bitcoin's market behavior (e.g., a major regulatory announcement).  
- Use Python basic libraries to ingest and process this data for a specific time window before and after the event.  
- Apply the CausalImpact library to the prepared dataset to assess the impact of the chosen event.  
- Students will interpret the results, visualize the causal impact using graphical tools (like matplotlib), and prepare a brief report of their findings.

**Useful resources:**

- [CausalImpact Documentation](https://pypi.org/project/causalimpact/)  
- [CoinGecko API Documentation](https://www.coingecko.com/en/api/documentation)  
- [CryptoCompare API](https://min-api.cryptocompare.com/documentation)  
- [Time Series Analysis Books and Tutorials](https://www.analyticsvidhya.com/blog/2018/02/time-series-forecasting-methods/)

**Is it free?**

- Yes, CausalImpact is an open-source library and free to use.  
- Collecting Bitcoin price data is free with limitations regarding the number of API calls, depending on the service provider.

**Python libraries / bindings:**

- `CausalImpact`: For causal inference on time series data (can be installed via pip: `pip install causalimpact`).  
- `pandas`: For data ingestion and preliminary data manipulation (install via pip: `pip install pandas`).  
- `requests`: For using APIs to fetch real-time Bitcoin data (install via pip: `pip install requests`).  
- `matplotlib`: For creating visual plots of data and results (install via pip: `pip install matplotlib`).

### **Celery**

**Title**: Real-Time Bitcoin Data Processing with Celery

**Difficulty**: Medium

**Description**  
Celery is an open-source distributed task queue library for Python. It is designed to handle asynchronous tasks and process them in the background. Celery is particularly useful for scheduling and executing tasks in a distributed manner, thus facilitating real-time data processing in Python applications. This project employs Celery to ingest and process real-time Bitcoin price data to perform time series analysis.

**Describe technology**

- Celery allows developers to define tasks that can be executed asynchronously. These tasks can be processed in parallel across multiple worker servers, making it highly scalable.  
- A key feature of Celery is task scheduling, which allows recurring tasks to be automatically executed at specified intervals.  
- Celery requires a message broker (such as RabbitMQ or Redis) to send and receive task messages, which facilitates communication between producers (task issuers) and consumers (workers executing tasks).

**Describe the project**

- The goal of this project is to develop a pipeline that periodically fetches Bitcoin price data from a public API (e.g., CoinDesk or CoinGecko) using Celery.  
- Set up a Celery task to retrieve Bitcoin price data every minute and schedule it to run continuously.  
- Implement a task pipeline where the fetched data is pre-processed, such as cleaning or converting to a different format, and stored in a time-series database (like InfluxDB).  
- Analyze the processed time series data to identify trends, calculate moving averages, and detect anomalies in Bitcoin prices over time.  
- The project will be implemented using basic Python packages for tasks that extend beyond Celery, such as APIs requests or data analysis.

**Useful resources**

- [Celery Documentation](https://docs.celeryproject.org/en/stable/)  
- [CoinDesk Bitcoin Price Index API](https://www.coindesk.com/coindesk-api)  
- [CoinGecko API Documentation](https://coingecko.com/en/api)  
- [InfluxDB Documentation](https://docs.influxdata.com/influxdb/v2.0/)

**Is it free?**  
Yes, Celery is an open-source project released under the BSD License. Using public APIs from data providers like CoinDesk or CoinGecko is typically free but may have usage limitations or require email registration for an API key.

**Python libraries / bindings**

- **Celery**: Install with `pip install celery` to set up the asynchronous task queue.  
- **requests**: For making HTTP requests to fetch Bitcoin prices (install via `pip install requests`).  
- **pandas**: For time series data manipulation and analysis (install via `pip install pandas`).  
- **InfluxDB Client**: Python client to interact with an InfluxDB time-series database (install via `pip install influxdb`).  
- **Message Broker**: Choose between Redis (install via `pip install redis`) or RabbitMQ (requires separate installation) as a message broker to facilitate task management.

### **Click**

**Title**: Bitcoin Time Series Analysis CLI Tool with Click  
**Difficulty**: 1 (easy)  
**Description**  
**Describe technology**  
Click is a Python package for creating composable and user-friendly command-line interfaces (CLIs). It simplifies parsing command-line arguments, options, and subcommands, enabling developers to build robust CLI tools with minimal code. Key features include:

- Decorator-based syntax for defining commands and options.  
- Automatic help page generation.  
- Support for nested commands and input validation.  
  Example: A CLI tool that fetches Bitcoin prices with `@click.command()` and processes user inputs like `--start-date` or `--interval`.

**Describe the project**  
Create a CLI tool using Click to ingest real-time Bitcoin price data (via a free API like CoinGecko or Coinbase) and perform basic time series analysis. The project steps:

1. **CLI Setup**: Use Click to define commands like `fetch-data` (to retrieve real-time prices) and `analyze` (to compute metrics).  
2. **Data Ingestion**: Fetch Bitcoin prices every N minutes (configurable via CLI options) and save to a CSV file.  
3. **Time Series Analysis**: Add commands to calculate:  
   - Rolling averages (e.g., 10-minute window).  
   - Price volatility (standard deviation).  
   - Detect sudden price spikes/drops.  
4. **Output**: Generate visualizations (e.g., matplotlib plots) or export results to a formatted report.

**Useful resources**

- **Click documentation**: [https://click.palletsprojects.com/](https://click.palletsprojects.com/)  
- **CoinGecko API guide**: [https://www.coingecko.com/en/api](https://www.coingecko.com/en/api)  
- **Pandas time series basics**: [https://pandas.pydata.org/docs/user\_guide/timeseries.html](https://pandas.pydata.org/docs/user_guide/timeseries.html)

**Is it free?**   
Yes. Click, CoinGecko API (free tier), and all suggested libraries are open-source and free to use.

**Python libraries / bindings**

- click: Core library for building the CLI.  
- requests: Fetch data from Bitcoin API.  
- pandas: Handle time series data (timestamps, rolling calculations).  
- matplotlib: Generate basic visualizations (optional). Example installation:

### **Clickhouse**

**Title**: Real-time Bitcoin Data Analysis using ClickHouse

**Difficulty**: 3 (Difficult)

**Description**  
ClickHouse is an open-source column-oriented database management system designed for online analytical processing (OLAP) of queries. With its ability to quickly ingest and process large volumes of data, ClickHouse is ideal for real-time analytics tasks involving high ingestion rates and low-latency query responses. This project will provide hands-on experience in setting up a ClickHouse-based system in Python to analyze real-time Bitcoin price data and perform time series analysis.

**Describe technology**

- ClickHouse is highly performant due to its capability of parallel processing, data compression, and vectorized query execution.  
- It's designed to optimize read-heavy workloads typical in analytical use cases, supporting complex queries on large datasets.  
- Key features include materialized views, aggregation functions, and support for real-time data ingestion.  
- Native support for time series analysis through various functions tailored for handling date and time data.  
- ClickHouse interacts with clients using HTTP and native protocols, with support for various query languages, including SQL.

**Describe the project**

- Students will set up a local development environment with ClickHouse using Docker, configuring it for optimal performance to handle real-time data streams.  
- Utilize Python to ingest Bitcoin price data from a public API, such as CoinGecko, in real-time and store it directly in a ClickHouse database.  
- Implement data ingestion pipelines leveraging ClickHouse's HTTP interface for seamless integration with APACHE Kafka or other real-time data sources.  
- Create and manage ClickHouse tables optimized for time series data, employing features like TTLs (Time-To-Live) for automatic data expiration.  
- Develop a time series analysis module utilizing ClickHouse SQL queries; this includes computing moving averages, detecting anomalies, and generating alerts based on predefined thresholds.  
- Conclude the project with a data visualization component using Python libraries like Matplotlib or Plotly to display insights gleaned from the ClickHouse database.

**Useful resources**

- Official ClickHouse Documentation: [https://clickhouse.com/docs/en/](https://clickhouse.com/docs/en/)  
- ClickHouse SQL Reference: [https://clickhouse.com/docs/en/sql-reference/](https://clickhouse.com/docs/en/sql-reference/)  
- Docker Setup for ClickHouse: [https://clickhouse.com/docs/en/development/tools/docker/](https://clickhouse.com/docs/en/development/tools/docker/)

**Is it free?**  
Yes, ClickHouse is open-source and free to use. It can be deployed on your infrastructure without licensing fees.

**Python libraries / bindings**

- **requests**: for HTTP interactions with data APIs.

```
pip install requests
```

- **clickhouse-driver**: A native Python client for ClickHouse to execute SQL queries and manage databases.

```
pip install clickhouse-driver
```

- **pandas**: To manipulate and prepare the data before ingestion & for preliminary analysis.

```
pip install pandas
```

- **plotly or matplotlib**: For data visualization.

```
pip install plotly
pip install matplotlib
```

This project involves working with ClickHouse to develop a robust real-time data analytics system for Bitcoin, providing students with practical skills in data ingestion, storage, and analysis using a high-performance database.

### 

### **Cline**

**Title**: Autonomous Bitcoin Analytics System Development with Cline  
**Difficulty**: 3 (difficult)

**Description**  
Develop an end-to-end Bitcoin market analysis system using Cline's AI agentic capabilities. The project involves creating a real-time dashboard that ingests cryptocurrency data, performs time series forecasting, detects anomalies, and auto-deploys a web visualization \- with Cline handling everything from API integration to error correction through VSCode integration.

**Describe Technology**

- **Cline**: AI assistant using Claude 3.7 Sonnet that:  
  - Creates/edits files while monitoring linter/compiler errors  
  - Executes terminal commands with human approval  
  - Performs browser-based testing & debugging  
  - Extends capabilities via Model Context Protocol (MCP)  
  - Manages context through AST analysis and regex searches

**Describe the Project**

1. **System Architecture Design**:  
     
   - Use Cline to scaffold project structure:

```
cline "Create Python project with modules for data ingestion, analysis, visualization, and tests"
```

   - Implement real-time data pipeline:  
     - CoinGecko/Binance API integration (WebSocket & REST)  
     - Redis caching for rate limiting  
     - Batch processing with PySpark

   

2. **AI-Driven Development**:  
     
   - Have Cline:  
     - Write data ingestion script with retry logic (`@problems` context for error fixing)  
     - Implement ARIMA forecasting using `statsmodels`  
     - Create anomaly detection with Isolation Forest (`scikit-learn`)  
     - Build React dashboard (via `npm run dev` browser testing)

   

3. **Auto-Remediation System**:  
     
   - Create MCP tools for:  
     - Automated AWS EC2 scaling based on price volatility  
     - Slack alerts for threshold breaches  
     - CI/CD pipeline for dashboard updates

   

4. **Context-Aware Maintenance**:  
     
   - Use Cline's snapshot system to:  
     - Roll back failed deployments  
     - A/B test different ML models  
     - Compare performance across git branches

**Useful Resources**

- [Cline Documentation](https://github.com/cline/cline)  
- [CoinGecko Streaming API](https://www.coingecko.com/en/api/docs/v3)  
- [Time Series Forecasting Guide](https://otexts.com/fpp3/)

**Is it free?**  
Cline extension is free, but requires API credits for AI models (OpenRouter/Anthropic/etc). CoinGecko API has free tier limits.

**Python Libraries / Bindings**

- `websockets`/`aiohttp`: Real-time data ingestion  
- `pandas`/`numpy`: Data transformation  
- `plotly`/`dash`: Visualization dashboard  
- `scikit-learn`: ML models  
- `pytest`: AI-generated test cases

### **Cloudflare Workers AI**

**Title**: Real-time Bitcoin Data Analysis Using Cloudflare Workers AI

**Difficulty**: Medium (2)

**Description** Cloudflare Workers AI is a cutting-edge platform for deploying serverless functions utilizing AI-powered capabilities at the edge of the network. It allows for real-time data processing and analysis with minimal latency. The main components include Workers for running code, and various SDKs and APIs for integrating AI functionalities and handling data processing tasks.

**Describe technology**

- **Cloudflare Workers**: Lightweight and fast, this service allows users to run JavaScript or WebAssembly code on Cloudflare's edge servers, reducing latency by processing requests closer to end-users.  
- **AI Integration**: Workers AI provides built-in support for handling AI tasks such as machine learning inference, natural language processing, and image recognition.  
- **Edge Computing**: By leveraging the edge network, Cloudflare Workers AI can handle data processing tasks, ensuring real-time and efficient management of applications like live data analysis.

**Describe the project** The project focuses on implementing a real-time system using Cloudflare Workers AI to ingest and process live Bitcoin price data:

- **Data Ingestion**: Use Cloudflare Workers to fetch live Bitcoin price data from a public API such as CoinGecko. The data will be handled at the edge to minimize latency.  
- **Data Storage and Transformation**: Write a Cloudflare Worker function to preprocess and organize incoming data. Apply AI techniques for basic time series analysis, such as predicting short-term trends and calculating moving averages.  
- **Real-time Analysis**: Utilize Workers AI capabilities to run machine learning models on the processed data to identify potential market opportunities.  
- **Visualization**: Develop a simple client-side application to query processed data from Workers and visualize potential trends or alerts in real time.

**Useful resources**

- [Cloudflare Workers Documentation](https://developers.cloudflare.com/workers/)  
- [Cloudflare AI Platform Overview](https://www.cloudflare.com/products/workers-ai/)  
- [CoinGecko API Documentation](https://www.coingecko.com/en/api/documentation)  
- [Python for Time Series Data Analysis](https://www.analyticsvidhya.com/blog/2020/07/a-quick-guide-to-time-series-forecasting-python/)

**Is it free?** Cloudflare Workers offers a free tier with limitations on compute and requests. Additional resources and features require a paid plan.

**Python libraries / bindings**

- **Cloudflare Python API**: While the main Workers tasks will be in JavaScript or WebAssembly, you can use the Cloudflare Python SDK for setting up Cloudflare accounts and managing workers programmatically.  
- **Requests**: Useful for testing API interactions to fetch Bitcoin data before implementing the edge functions.

This project helps students learn the fundamentals of edge computing and how to implement real-time data processing with AI capabilities, combining both serverless technology and Python data analysis.

### **Coach (by Intel)**

**Title**: Real-Time Bitcoin Price Analysis using Coach (by Intel)

**Difficulty**: 2 (medium)

**Description**:  
Coach (by Intel) is an advanced deep reinforcement learning framework designed to enable rapid design, training, and evaluation of reinforcement learning agents. It's built to support various reinforcement learning algorithms and environments, making it a versatile tool to apply in real-time data analyses, like predicting Bitcoin price fluctuations over time. This project will guide students through understanding the key aspects of Coach, focusing on how to set up reinforcement learning environments, implement several basic algorithms, and tune them for the best performance on streaming Bitcoin price data. The project has a specific emphasis on time series analysis, facilitating understanding of market trends and dynamic pricing strategies.

**Describe technology**:

- **Coach (by Intel)**: A framework that supports multiple reinforcement learning algorithms, from simple to sophisticated, allowing rapid experimentation and development.  
- Focuses on modularity and flexibility, enabling integration with various environments and data sources.  
- Implements advanced techniques like policy gradients, value iterations, and actor-critic methods.

**Describe the project**:

- **Objective**: Develop a reinforcement learning (RL) agent to predict and respond to real-time Bitcoin price data.  
- **Steps**:  
  1. **Data Ingestion**: Use a public API to stream Bitcoin price data into your Python environment.  
  2. **Environment Setup**: Configure the RL environment using Coach to process the streamed data.  
  3. **Algorithm Implementation**: Implement basic reinforcement learning algorithms like Q-learning or SARSA within Coach.  
  4. **Training and Tuning**: Train your RL agents on historical Bitcoin price data to find the optimal strategy.  
  5. **Real-Time Testing**: Deploy the trained RL agent on live data to predict price movements and suggest trading actions.  
  6. **Analysis**: Analyze the agent's predictions and actions to assess performance, refine models by hyper-parameter tuning and model adjustments based on results.

**Useful resources**:

- [Coach GitHub Repository](https://github.com/NervanaSystems/coach) \- Includes codebase and documentation.  
- [Intel Developer Zone: Intel AI and Analytics Toolkit](https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/ai-analytics-toolkit.html) \- Provides resources and tools for implementing AI projects.  
- [Bitcoin APIs for real-time data](https://www.coingecko.com/en/api) \- Access current and historical market data.

**Is it free?**  
Yes, Coach (by Intel) is open-source and free to use. You might incur costs from cloud services or data providers if opting for such resources beyond the free tier.

**Python libraries / bindings**:

- **Coach**: Python framework for implementing RL algorithms.  
- **Numpy**: Basic numerical computing.  
- **Pandas**: For data handling and processing values.  
- **Matplotlib** / **Seaborn**: Visualization libraries to plot data trends and results.  
- **Requests**: For API calls to ingest real-time Bitcoin price data.

### **CouchDB**

**Title**: Real-time Bitcoin Data Ingestion and Analysis using CouchDB

**Difficulty**: Easy

**Description**  
Apache CouchDB is an open-source NoSQL database that uses a document-oriented data model. CouchDB is known for its ease of use, real-time data updating capabilities, and its efficient replication protocol designed for offline-first applications. The aim of this project is to teach students how to use CouchDB for storing and processing real-time data, such as Bitcoin prices, using basic Python packages for additional processing.

**Describe technology**

- CouchDB stores data in a JSON-based, document-based model allowing complex data structures to be easily handled.  
- It uses MapReduce views and indexing, which provide powerful querying capabilities.  
- CouchDB supports multi-master setups, meaning that you can have multiple copies of your database actively synchronizing with each other.  
- It is highly scalable and has built-in support for fault-tolerance and database sync across distributed systems.

**Describe the project**

1. **Ingest Real-time Bitcoin Data**: Use the public CoinGecko API to continuously fetch Bitcoin prices.  
2. **Store Data in CouchDB**: Set up a CouchDB instance and create a database to store the data fetched from the API. Each data point (price information) will be stored as a JSON document.  
3. **Data Processing**: Use Python to access the data stored in CouchDB and perform basic time-series analysis, such as calculating moving averages or visualizing price trends over time.  
4. **Query and View Creation**: Create MapReduce views in CouchDB to filter and sort data according to specific criteria, for example, finding price trends over specific intervals.  
5. **Presentation**: Implement a simple command-line interface using Python to query the database and present the analyzed data.

**Useful resources**

- [CouchDB Official Documentation](https://docs.couchdb.org/en/stable/)  
- [CoinGecko API Documentation](https://www.coingecko.com/en/api)  
- [Python CouchDB Library](https://pypi.org/project/CouchDB/)

**Is it free?**  
Yes, CouchDB is open-source and free to use. CoinGecko also provides free access to its API with certain rate limitations.

**Python libraries / bindings**

- CouchDB: To interact with the CouchDB instance and manage databases, you can use the CouchDB Python client: `pip install CouchDB`.  
- Requests: To fetch data from the CoinGecko API, you might want to use the Requests library to manage HTTP requests: `pip install requests`.  
- Pandas: For data processing and analysis of time-series data: `pip install pandas`.  
- Matplotlib: For visualizing data and creating graphs: `pip install matplotlib`.

### **cryptography**

**Title**: Secure Bitcoin Price Ingestion using Cryptography

**Difficulty**: 3 (difficult)

**Description**  
Cryptography is a critical technology in securing data transmission and storage, especially in applications dealing with sensitive information like financial data. This project involves the use of cryptographic techniques to ingest and process real-time Bitcoin price data securely. Students will explore the fundamental aspects of cryptography, including encryption, decryption, and digital signatures, and learn how these can be applied in a Python-based big data system.

**Describe Technology**  
Cryptography involves techniques for secure communication in the presence of third parties. The key functionalities include:

- **Encryption and Decryption**: Transforming readable data into unreadable format (encryption) and vice versa (decryption), using keys.  
- **Digital Signatures**: Ensuring data integrity and authenticity by allowing the receiver to verify that the data was not altered.  
- **Hashing**: Converting data into a fixed-size string of characters, which acts as a "fingerprint" of the data.

Examples of cryptographic algorithms include AES for symmetric encryption, RSA for asymmetric encryption, and SHA-256 for hashing.

**Describe the Project**  
The project's objective is to design a system that securely ingests real-time Bitcoin price data from a public API (e.g., CoinGecko) while ensuring data integrity and confidentiality. The project involves the following tasks:

1. **Data Ingestion**: Set up a Python script to fetch Bitcoin price data from the API at regular intervals.  
2. **Data Encryption**: Encrypt the incoming data using a symmetric key algorithm like AES to ensure privacy during transmission.  
3. **Data Storage**: Store the encrypted data in a secure format.  
4. **Data Processing**: Decrypt the data for analysis, verifying its integrity using digital signatures or hashes.  
5. **Time Series Analysis**: Implement a basic time series analysis on the decrypted data, such as moving averages or trend analysis.  
6. **Security Reporting**: Document the cryptographic methods used and their effectiveness in securing the data pipeline.

**Useful Resources**

- "Cryptography and Network Security" by William Stallings  
- PyCryptodome Documentation ([https://pycryptodome.readthedocs.io](https://pycryptodome.readthedocs.io))  
- Python's hashlib Documentation ([https://docs.python.org/3/library/hashlib.html](https://docs.python.org/3/library/hashlib.html))

**Is it free?**  
Yes, cryptographic libraries in Python, such as PyCryptodome and hashlib, are open-source and free to use.

**Python Libraries / Bindings**

- **PyCryptodome**: A self-contained Python package of low-level cryptographic primitives. You can install it with `pip install pycryptodome`.  
- **hashlib**: A built-in Python module for various secure hash and message digest algorithms.  
- **requests**: For HTTP requests to fetch data from the Bitcoin price API (`pip install requests`).

This project will provide students with hands-on experience in cryptography, data security, and time series analysis, preparing them for real-world challenges in data science and cybersecurity.

### **cuDF**

**Title**: Real-Time Bitcoin Data Processing using cuDF  
**Difficulty**: 3 (difficult)

**Description**:  
In this project, students will delve into the world of high-performance data processing using `cuDF`, a Python GPU DataFrame library. Developed by Rapids AI, `cuDF` enables fast computation on dataframes by leveraging NVIDIA GPUs. This project will involve ingesting real-time Bitcoin price data, processing it to perform time series analysis, and ultimately examining trends and patterns. By implementing this project, students will gain a deeper understanding of GPU acceleration in data analysis and the application of dataframes in handling complex, large-scale datasets.

**Describe technology**:

- **Overview**: cuDF is a GPU-accelerated library for manipulating data frames, akin to pandas but designed for high performance using NVIDIA GPUs. It is part of the RAPIDS AI suite and provides a familiar DataFrame API that mimics pandas to offer seamless transition for data scientists familiar with pandas operations.  
    
- **Key Features**:  
    
  - GPU acceleration to speed up data processing tasks significantly.  
  - Familiar pandas-like API for DataFrame operations.  
  - Integration with other RAPID AI libraries for machine learning and graph analytics.  
  - Efficient handling of large datasets and support for operations including filtering, aggregation, and joins.

**Describe the project**:

- **Objective**: Implement a real-time data processing pipeline using `cuDF` to analyze Bitcoin price trends over time by drawing on streaming data from a public API like CoinGecko or CoinAPI.  
    
- **Step 1 \- Data Ingestion**: Use basic Python libraries such as `requests` or `websockets` to stream Bitcoin prices data in real-time.  
    
- **Step 2 \- Data Processing**:  
    
  - Convert the streaming data into cuDF DataFrames.  
  - Perform time series analysis to compute key measures such as moving averages, volatility, and rate of change.  
  - Carry out slice or window operations to understand trends within specific timeframes.


- **Step 3 \- Visualization**: Use libraries like Matplotlib or Plotly with `cuDF` to visualize the processing results, showcasing price trends and potential anomalies in the market data.  
    
- **Step 4 \- Optimization**: Focus on optimizing the pipeline for scalability; explore the computational improvements using different-sized data batches and GPU contexts.  
    
- **Outcome**: Students will demonstrate how to ingest, process, and analyze large volumes of streaming financial data efficiently with GPU power, providing insights into the undercurrents driving Bitcoin price movements.

**Useful resources**:

- [cuDF Documentation](https://docs.rapids.ai/api/cudf/stable/)  
- [RAPIDS AI Getting Started Guide](https://rapids.ai/start.html)  
- [Bitcoin API example \- CoinGecko](https://www.coingecko.com/en/api)

**Is it free?**  
Yes, RAPIDS AI, including cuDF, is open-source and freely available. However, access to a CUDA-capable GPU is recommended to maximize performance benefits. Cloud options like Google Colab may offer free GPU usage for limited computation.

**Python libraries / bindings**:

- **cuDF**: Leverage the `cudf` package for fast DataFrame operations. Install via `conda install -c rapidsai -c nvidia -c conda-forge cudf=21.08 python=3.8 cudatoolkit=11.2`.  
- **requests** or **websockets**: For data ingestion from web APIs.  
- **Matplotlib/Plotly**: Visualization libraries for plotting and visual analysis of results.  
- **NumPy**: For mathematical operations as needed within processing logic.

Students completing this project will not only refine their Python programming and data science skills but also gain hands-on experience in GPU-accelerated data processing with cuDF, enabling them to tackle future big data challenges efficiently.

### **Customer.io**

**Title**: Customer Engagement Time Series Analysis Using Customer.io Event Data  
**Difficulty**: 2 (medium)  
**Description**  
Implement a system to analyze time-based customer interaction data from Customer.io, focusing on event patterns, trend forecasting, and anomaly detection.

**Describe technology**

- **Customer.io**: A customer engagement platform that tracks user interactions (e.g., email opens, app events) and stores them as timestamped events.  
- Basic functionalities:  
  1. Retrieve event data via API (e.g., `GET /api/v1/customers/{id}/events`).  
  2. Track user behaviors over time (e.g., login frequency, campaign responses).


**Describe the project**

1. **Data Ingestion**: Use Customer.io’s Python client to fetch timestamped event data (e.g., email opens, clicks) for a 6-month period.  
2. **Time Series Processing**:  
   - Aggregate events into daily/weekly counts (e.g., "number of logins per day").  
   - Identify trends (e.g., spikes after marketing campaigns).  
3. **Forecasting**: Use a simple ARIMA model (via `statsmodels`) to predict future engagement.  
4. **Anomaly Detection**: Flag unusual activity (e.g., sudden drops in email opens) using threshold-based rules.  
5. **Visualization**: Plot trends and forecasts with `matplotlib`.  
   

**Useful resources**

- [Customer.io API Documentation](https://customer.io/docs/api/)  
- [Pandas Time Series Guide](https://pandas.pydata.org/docs/user_guide/timeseries.html)  
- [ARIMA Modeling Tutorial](https://www.machinelearningplus.com/time-series/arima-model-time-series-forecasting-python/)


**Is it free?**  
Yes (with a free-tier Customer.io trial account).

**Python libraries / bindings**

- `customerio` (official client)  
- `pandas` (data wrangling)  
- `statsmodels` (ARIMA forecasting)  
- `matplotlib` (visualization)  
- 

### 

### **Dagster**

**Title**: Real-time Bitcoin Data Ingestion and Analysis with Dagster   
**Difficulty**: Medium 

Description: Dagster is an open-source data orchestration platform designed to develop, run, monitor, and maintain data processing pipelines. It is particularly useful for managing complex workflows that require several stages of transformation and different data sources. Dagster provides robust error handling, logging, and monitoring capabilities, making it a powerful tool for data scientists and engineers. Its unique approach to defining data pipelines through solid compositions allows users to modularly build and reuse components across projects.

In this project, students will set up a real-time data ingestion and processing system for Bitcoin prices using Dagster. The aim is to fetch real-time Bitcoin price data from a public API (e.g., CoinGecko or Binance), store it for historical analysis, and perform preliminary time series analysis. Students will focus on building a Dagster pipeline that continuously collects Bitcoin data, performs some initial transformations, and stores the results in a database like SQLite or a local CSV file.

- **Describe technology:**  
    
  - Understand Dagster's core elements: solids, pipelines, and resources.  
  - Learn to define and execute pipelines using Dagster's domain-specific language and Python.  
  - Explore Dagster's capabilities for handling failures, retries, and logging.  
  - Utilize Dagster's UI for monitoring and visualizing pipeline execution and debugging processes.


- **Describe the project:**  
    
  - Set up a Dagster environment and create a new repository for the project.  
  - Develop a solid for making API requests to fetch Bitcoin price data.  
  - Create a solid to process and transform the fetched data, extracting relevant features like date-time and price.  
  - Implement a storage solution using SQLite or CSV to save the processed data for historical tracking and analysis.  
  - Design a pipeline to automate the ingestion and storage process.  
  - Perform basic time series analysis on the stored Bitcoin data, such as calculating moving averages or detecting trends.  
  - Learn to schedule the pipeline to run at defined intervals to ensure continuous data ingestion.


- **Useful resources:**  
    
  - [Dagster Documentation](https://docs.dagster.io/)  
  - [Getting Started with Dagster](https://docs.dagster.io/getting-started)  
  - [CoinGecko API Documentation](https://www.coingecko.com/en/api)


- **Is it free?**  
    
  - Yes, Dagster is open-source and free to use. You might incur costs depending on how you choose to store data (e.g., cloud storage services).


- **Python libraries / bindings:**  
    
  - `Dagster`: Core libraries to set up and run data orchestrations. (Install with `pip install dagster`)  
  - `Requests`: A simple HTTP library for making API requests. (Install with `pip install requests`)  
  - `Pandas`: A powerful data manipulation library for processing data. (Install with `pip install pandas`)  
  - `SQLite`: A lightweight database accessible through Python's built-in `sqlite3` module.  
  - `Matplotlib` or `Plotly`: For visualizing historical Bitcoin trends as part of the time series analysis.

### **Databricks CLI**

**Title**: Real-Time Bitcoin Price Analysis with Databricks CLI

**Difficulty**: 3 (difficult)

**Description**:  
The Databricks CLI (Command Line Interface) is a tool designed to make interacting with the Azure and AWS Databricks workspaces easier. By providing a platform for developing, managing, and deploying large-scale data processing tasks, the Databricks CLI can streamline many of the tasks that data scientists and engineers typically perform within the Databricks environment. This project focuses on utilizing Databricks CLI to build a real-time Bitcoin price analysis system.

**Describe technology**:

- Databricks CLI is an interface to automate and programmatically interact with workspaces.  
- It allows users to perform a variety of tasks such as creating clusters, submitting jobs, or managing file systems without needing to rely on the Databricks web app.  
- Integration with Unix-based systems allows for easy and efficient script-based control of Databricks resources.

**Describe the project**:

- **Objective**: Implement a system using Databricks CLI to ingest and process real-time Bitcoin pricing data and perform a time series analysis to forecast future price trends.  
- **Steps**:  
  - Set up a Databricks workspace and install Databricks CLI.  
  - Configure authentication using a personal access token to securely interact with your workspace.  
  - Utilize Databricks CLI to create and configure a cluster in Databricks for processing data.  
  - Write a Python script that fetches real-time Bitcoin price data from a public API (e.g., CoinGecko).  
  - Schedule jobs using Databricks CLI to run the data ingestion script at regular intervals.  
  - Store the ingested data into a distributed file system, such as DBFS, for further analysis.  
  - Implement a time series analysis model using Python libraries such as `pandas` and `statsmodels` to predict future prices.  
  - Use Databricks notebooks to visualize the results and forecast plots.  
  - Automate the entire workflow using shell scripts to ensure seamless data processing and analysis.

**Useful resources**:

- [Databricks CLI Documentation](https://docs.databricks.com/dev-tools/cli/index.html)  
- [CoinGecko API Documentation](https://www.coingecko.com/en/api/documentation)  
- [Time Series Analysis with Python: from Basics to Advanced](https://www.analyticsvidhya.com/blog/2021/07/time-series-forecasting-using-python-a-comprehensive-guide/)

**Is it free?**

- You need to create an account on Databricks. Free trials are often available, but long-term usage might require subscription plans depending on resource usage.

**Python libraries / bindings**:

- `Databricks CLI`: To manage clusters, jobs, and files. [Installation Guide](https://docs.databricks.com/dev-tools/cli/index.html)  
- `pandas`: For data manipulation and analysis.  
- `statsmodels`: For statistical modeling, including time series forecasting.  
- `requests`: To fetch data from the CoinGecko API.

### **Django ORM**

**Title**: Real-Time Bitcoin Analysis using Django ORM

**Difficulty**: 2 

**Description**: Django ORM (Object-Relational Mapping) is a core component of the Django web framework that allows you to interact with your database, like SQL statements. It offers an intuitive and efficient way to manage database queries through Python classes and methods rather than writing raw SQL. This project involves using the Django ORM to ingest real-time Bitcoin price data from an API, store it in a database, and perform time series analysis to gain insights into price trends and volatility.

**Describe technology**:

- *Django ORM* is an integral part of Django that simplifies the communication between relational databases and Python applications through a high-level Python API.  
- It supports various database backends (e.g., PostgreSQL, MySQL, SQLite).  
- Django models represent database tables, leveraging Python classes and attributes, making it easy and intuitive to manipulate complex queries.  
- With migrations, Django ORM automatically adapts the database schema as models evolve, which ensures a smooth evolution of the database structure.  
- It offers filtering and querying capabilities through chaining methods and Pythonic syntax, providing powerful and flexible data retrieval options.

**Describe the project**:

- **Objective**: Use Django ORM to ingest, store, and analyze live Bitcoin pricing data.  
- **Data Ingestion**: Fetch real-time Bitcoin prices from a public API, such as CoinGecko, and use Django models to store the fetched data in a SQLite database.  
- **Data Processing**: Develop Django model methods or separate functions to perform basic data processing such as calculating average prices, detecting peaks, or measuring volatility over a specific period.  
- **Time Series Analysis**: Implement a simple time series analysis function or feature within the project using standard Python libraries to visualize trends, cyclic patterns, or anomalies in bitcoin prices.  
- **Deployment**: Set up a Django web application to provide users with the ability to visualize real-time Bitcoin price data and analysis directly in a web interface.

**Useful resources**:

- [Django Official Documentation](https://docs.djangoproject.com/en/stable/)  
- [Django ORM Documentation](https://docs.djangoproject.com/en/stable/topics/db/queries/)  
- [SQLite Database](https://www.sqlite.org/index.html)  
- [CoinGecko API Documentation](https://www.coingecko.com/en/api)

**Is it free?**: Yes, Django is an open-source framework, and SQLite is a free database engine. All tools mentioned in the project can be used at no charge.

**Python libraries / bindings**:

- **Django**: For using Django ORM and building the overall project structure. Install it via `pip install django`.  
- **Requests**: To fetch real-time Bitcoin data from APIs. Install it using `pip install requests`.  
- **Matplotlib or Plotly**: For creating visualizations of the time series analysis, install using `pip install matplotlib` or `pip install plotly`.  
- **Pandas**: Optional but useful for handling complex data manipulation and analysis tasks. Install it with `pip install pandas`.

### **Datadog**

**Title**: Analyze Bitcoin Time Series Data with Datadog

**Difficulty**: Medium 

**Description**  
This project aims to leverage Datadog as a monitoring and analytics platform to ingest and process real-time bitcoin price data. Datadog provides observability services that integrate seamlessly with various data sources. Through this project, students will get hands-on experience with Datadog, learning how to set up data pipelines, monitor performance, and conduct real-time analytics. This will involve setting up data ingestion from a public API, visualizing time-series data, and performing basic trend analysis using Python.

**Describe Technology**  
Datadog is a monitoring and analytics platform for developers, IT operation teams, and business users in the cloud age. It offers a comprehensive view of real-time data from various sources, allowing users to monitor servers, databases, tools, and services through a unified platform. Key features include:

- Customizable dashboards for visualization.  
- Built-in support for monitoring time-series data.  
- Alerts based on real-time data insights.  
- Seamless integration with various data sources and services.

**Describe the Project**  
The focus of this project is to employ Datadog for real-time monitoring and analysis of Bitcoin price data. The steps include:

1. **Data Ingestion**: Use an API like CoinGecko to fetch real-time Bitcoin price data and send it to Datadog for monitoring.  
2. **Datadog Setup**: Create a new Datadog account if not already existing. Configure a new custom event in Datadog to receive price data in real-time.  
3. **Dashboards and Visualization**:  
   - Create a real-time dashboard to visualize Bitcoin price fluctuations.  
   - Configure alerts to notify when the price crosses certain thresholds.  
4. **Time Series Analysis**: Utilize Datadog’s built-in tools to perform basic time series analysis.  
   - Implement moving average calculations to smooth out price data.  
   - Analyze trends and volatility over a set period.  
5. **Python Integration**: Write Python scripts to automate data fetching and transmission to Datadog.  
6. **Reporting and Optimization**: Present findings through automatically generated reports and optimize data ingestion for performance.

**Useful Resources**

- [Datadog Documentation](https://docs.datadoghq.com/)  
- [CoinGecko API Documentation](https://coingecko.com/en/api)  
- [Python Requests Library](https://requests.readthedocs.io/)

**Is it Free?**  
Datadog offers a 14-day free trial for new users. However, a paid subscription is required for continued use beyond the trial period with full features.

**Python Libraries / Bindings**

- **Requests**: A comprehensive HTTP library for Python that allows you to send HTTP requests for fetching API data. Install it using `pip install requests`.  
- **Datadog API Client**: Datadog provides a Python client to interact with their API for sending data and managing resources. Install with `pip install datadog`.  
- **Pandas (optional)**: For additional data manipulation and analysis in Python, install using `pip install pandas`.

### 

### **DataHub**

**Title**: Real-time Bitcoin Data Processing with DataHub

**Difficulty**: Easy

**Description**  
DataHub is an open-source platform that helps users to discover, understand, and collaborate on datasets. It's built to handle metadata management, making it easier to manage large data ecosystems by facilitating dataset lineages, versioning, and collaborative data efforts. Through this project, students will learn the basic concepts of DataHub, including how to set up and manage a simple data lineage and metadata management process.

**Describe technology**

- DataHub offers features like easy dataset discovery, data lineage tracking, and metadata storage, which make it ideal for handling datasets in real-time environments.  
- Students will get familiar with DataHub's basic concepts, including ingesting metadata and managing data flows through a centralized platform.  
- Understand how DataHub supports integration with third-party data platforms and tools using its Python-friendly API.

**Describe the project**

- **Objective**: Implement a real-time data ingestion and processing system for Bitcoin price data utilizing DataHub for metadata management.  
- Students will initially set up a basic DataHub environment, enabling them to create a metadata catalog for their datasets and interactions.  
- Use a public API like CoinGecko to fetch real-time Bitcoin prices.  
- Implement a script in Python to continuously ingest these prices and update the DataHub metadata catalog accordingly.  
- Carry out time series analysis on the ingested Bitcoin data to identify trends, visualize price changes, and store these analyses within DataHub to facilitate easy access and collaboration.  
- The project reinforces key concepts of data ingestion, real-time data processing, and metadata management.

**Useful resources**

- [DataHub Official Documentation](https://datahubproject.io/docs/)  
- [CoinGecko API Documentation](https://www.coingecko.com/en/api)  
- [Introduction to Time Series Analysis with Python](https://towardsdatascience.com/introduction-to-time-series-analysis-using-python-3eca768058b4)

**Is it free?**  
Yes, DataHub is open-source and free to use. However, if you require cloud infrastructure to deploy it, there might be associated costs.

**Python libraries / bindings**

- DataHub’s Python Client: Use for interfacing with DataHub to catalog your data.  
- `requests` library: For handling API requests to fetch real-time Bitcoin data.  
- `pandas`: For easy manipulation and analysis of the time series data.  
- `matplotlib` or `seaborn`: For data visualization to facilitate time series trend analysis.

### 

### **Dataprep**

**Title**: Real-time Bitcoin Data Processing using Dataprep

**Difficulty**: Difficult

**Description**:  
This project involves implementing a system for ingesting and processing real-time Bitcoin data using Dataprep. Dataprep is a Python library designed to simplify data collection, exploration, cleaning, and visualization. It streamlines data preparation tasks, traditionally elaborate and error-prone, into straightforward processes. The main focus of this project is to demonstrate Dataprep's capabilities in handling time series analysis of Bitcoin prices in a big data context.

**Describe technology**:

- **Dataprep**: A Python library that offers modules to easily collect, explore, clean, and visualize data. Dataprep simplifies common data preparation tasks through:  
    
  - `Dataprep.eda`: A module for exploratory data analysis (EDA), enabling the quick generation of statistics, visualizations, and reports.  
  - `Dataprep.connector`: Facilitates connection to a wide variety of data sources, supporting seamless data ingestion.  
  - `Dataprep.clean`: Provides tools for cleaning and preparing data with flexible and intuitive methods.  
  - `Dataprep.feature`: (optional, based on needs) Assists in feature engineering tasks.


  Dataprep is designed to be intuitive and to integrate smoothly with commonly used data manipulation libraries like pandas.

**Describe the project**:

- **Objective**: Build a system that ingests real-time Bitcoin price data from a public API and performs time series analysis using Dataprep.  
- **Steps involved**:  
  1. **Data Ingestion**: Use the `Dataprep.connector` module to fetch real-time Bitcoin price data from a chosen public API, such as CoinGecko.  
  2. **Data Cleaning**: With the `Dataprep.clean` module, process the ingested data to handle missing values, normalize data fields, and remove inconsistencies.  
  3. **Data Exploration**: Utilize `Dataprep.eda` for exploratory data analysis to understand trends and patterns in Bitcoin price movements.  
  4. **Time Series Analysis**: Implement a time series analysis method (e.g., ARIMA) to forecast Bitcoin prices. This step involves preparing the datasets for analysis, selecting a model, and evaluating its performance.  
  5. **Visualization**: Generate insightful visualizations of historical Bitcoin prices, forecast results, and potential future trends using the visualization capabilities of `Dataprep.eda`.

**Useful resources**:

- [Dataprep Official Documentation](https://docs.dataprep.ai/index.html)  
- [Dataprep GitHub Repository](https://github.com/sfu-db/dataprep)  
- [Time Series Forecasting Methods](https://otexts.com/fpp3/)  
- [CoinGecko API Documentation](https://www.coingecko.com/en/api)

**Is it free?**:  
Yes, Dataprep is an open-source library that is free to use. However, using certain data source APIs, such as CoinGecko, may have rate limits or usage tier plans if heavy data usage is required.

**Python libraries / bindings**:

- `dataprep`: The primary library for data preparation tasks. Installable using `pip install -U dataprep`.  
- `pandas`: For general data manipulation and cleaning tasks. Installable with `pip install pandas`.  
- `requests`: To facilitate API calls for fetching Bitcoin data. Installable using `pip install requests`.  
- `statsmodels`: A library for performing time series analysis, such as ARIMA. Install with `pip install statsmodels`.  
- `matplotlib`/`seaborn`: For plotting and visualization of results. Install using `pip install matplotlib seaborn`.

### **Docker SDK for Python**

**Title**: Real-Time Bitcoin Price Analysis using Docker SDK for Python

**Difficulty**: 3 (difficult)

**Description**  
This project aims to develop a real-time data ingestion and analysis system for Bitcoin price data utilizing the Docker SDK for Python. Docker SDK for Python is a powerful tool that allows for the management and orchestration of Docker containers, providing an abstraction layer for container interactions through Python scripts. This project will involve using Docker containers to set up an isolated environment for ingesting real-time Bitcoin price data, storing it, and then performing time series analysis.

**Describe technology**  
Docker SDK for Python provides a programmatic way to control and manage Docker containers from within Python programs. It abstracts many complex tasks related to container management, such as starting, stopping, and linking containers, managing volumes, and networks. This SDK enables users to easily create portable and consistent environments, which is particularly beneficial for deploying big data systems that require scalable and isolated data processing pipelines.

**Describe the project**

- **Ingest and Store Real-Time Data**: Use Python with requests library to fetch Bitcoin price data from a public API like CoinGecko. Store the data in a time-series database like InfluxDB running within a Docker container. Docker SDK will be used to manage and automate the setup and configuration of the InfluxDB container.  
- **Set Up Processing Pipelines using Docker Containers**: Design a series of Docker containers for specific tasks (e.g., fetching data, performing calculations, visualization). The SDK will help automate the deployment of these containers and manage their interactions.  
- **Time Series Analysis**: Utilize Python packages like pandas and statsmodels within a containerized environment to perform time series analysis on the collected data. Implement analysis algorithms such as moving averages, ARIMA models, or others to extract trends and make forecasts on Bitcoin prices.  
- **Visualization**: Deploy a Grafana container using Docker SDK for real-time visualization of the Bitcoin price trends. Connect Grafana to the InfluxDB instance to illustrate the analytics in a user-friendly dashboard.

**Useful resources**

- [Docker SDK for Python Documentation](https://docker-py.readthedocs.io/en/stable/)  
- [InfluxDB Documentation](https://docs.influxdata.com/influxdb/)  
- [Grafana Documentation](https://grafana.com/docs/grafana/latest/)  
- [CoinGecko API](https://www.coingecko.com/en/api)

**Is it free?**  
Yes, using Docker SDK for Python, InfluxDB, and Grafana is free. For accessing these technologies, students need to install Docker Desktop, which is also available for free, though some limitations might apply in community editions.

**Python libraries / bindings**

- **docker**: The Python library for Docker SDK, installable via `pip install docker`. It will be used to interact programmatically with Docker services.  
- **requests**: For making HTTP requests to fetch real-time data from APIs, installable via `pip install requests`.  
- **pandas**: To manage and manipulate data, installable via `pip install pandas`.  
- **statsmodels**: For performing advanced statistical time series analysis, installable via `pip install statsmodels`.

### **DocsGPT**

**Title**: Real-Time Bitcoin Data Q\&A Bot with DocsGPT  
**Difficulty**: 1 (medium)

**Describe technology**  
DocsGPT is an open-source AI tool designed to generate or retrieve documentation answers using natural language. It leverages language models (like GPT) to understand user queries and provide context-aware responses from documentation sources. Key features:

- Integration with custom datasets (e.g., CSV, text files).  
- Natural language processing for querying structured data.  
- Simple API or local deployment for small-scale projects.

**Describe the project**  
Build a CLI tool that ingests real-time Bitcoin price data (from CoinGecko API) and uses DocsGPT to answer time series-related questions. The project steps:

-   
1. **Data Ingestion**: Fetch Bitcoin prices every 5 minutes and store them in a CSV file with timestamps.  
2. **Time Series Processing**: Calculate basic metrics (e.g., hourly average, daily volatility) using pandas.  
3. **DocsGPT Setup**: Train DocsGPT on the Bitcoin dataset to understand fields like `timestamp`, `price`, and `volatility`.  
4. **Q\&A Interface**: Create a CLI where users ask questions like:  
   - "What was the highest price in the last 6 hours?"  
   - "When did the price drop by more than 2% today?"  
     DocsGPT processes the query, retrieves data, and 

**Useful resources**

- DocsGPT GitHub: [https://github.com/arc53/DocsGPT](https://github.com/arc53/DocsGPT)  
- CoinGecko API: [https://www.coingecko.com/en/api](https://www.coingecko.com/en/api)  
- Pandas time series guide: [https://pandas.pydata.org/docs/user\_guide/timeseries.html](https://pandas.pydata.org/docs/user_guide/timeseries.html)

**Is it free?**  
Yes. DocsGPT is open-source, and CoinGecko’s free tier supports up to 50 calls/minute.

**Python libraries / bindings**

- requests: Fetch Bitcoin data from CoinGecko API.  
  - pandas: Process time series data and calculate metrics.  
  - langchain (optional): Simplify DocsGPT integration for local LLM workflows.  
  - python-dotenv: Manage API keys (if using cloud-based LLMs).  
  - returns a plain-English answer.

### **Dolibarr**

**Title**: Real-Time Bitcoin Analysis with Dolibarr  
**Difficulty**: 2 (Medium)

**Description**  
Dolibarr is an open-source ERP and CRM software that integrates seamlessly with various data sources and can be customized to fit a range of business needs. It is well known for its modular design, allowing users to add specific functionalities required for their operations. Though typically used for business management, Dolibarr can be adapted for big data applications, including ingesting and processing real-time data streams, such as Bitcoin price data. For this project, students will explore Dolibarr's capabilities by implementing a system that captures, processes, and analyzes real-time Bitcoin prices.

**Describe technology**

- **Core Features**: Dolibarr offers a user-friendly interface for managing business operations, with modules available for accounting, sales, CRM, inventory, and more.  
- **Modular Design**: The platform is built to be highly modular, allowing users to add only the functionalities they need. Modules can be easily developed or customized to extend the system's capabilities, making it versatile for tasks beyond traditional ERP functionalities.  
- **Customization**: Dolibarr can be extended with custom modules to process and analyze financial data, which makes it an intriguing choice for time series analysis projects.  
- **Real-time Data Handling**: Though not originally designed for big data handling, Dolibarr can be integrated with external scripts and APIs to accommodate real-time data processing.

**Describe the project**

- **Objective**: Develop a module within Dolibarr to handle the ingestion and processing of Bitcoin price data in real-time. Extend its CRM module features to store Bitcoin transaction data.  
- **Data Source**: Utilize a public API, such as CoinGecko, to fetch real-time Bitcoin data.  
- **Data Storage**: Use Dolibarr’s extensible database structure to store Bitcoin prices, handling updates or additions for each price change.  
- **Time Series Analysis**: Implement time series analysis features within Dolibarr by creating a new module or integrating external Python scripts. Analyze trends, compute moving averages, and visualize price changes over time.  
- **User Interface**: Create a dashboard within Dolibarr to display real-time data and analysis results, making use of the platform's existing UI capabilities.

**Useful resources**

- [Dolibarr Official Website](https://www.dolibarr.org/)  
- [Dolibarr GitHub Repository](https://github.com/Dolibarr/dolibarr)  
- [Dolibarr Module Development Documentation](https://wiki.dolibarr.org/index.php/Module_development)

**Is it free?**  
Yes, Dolibarr is open-source and free to use. It is distributed under the GNU General Public License (GPL), which allows for free use, modification, and distribution of the software.

**Python libraries / bindings**

- **Requests**: To handle API calls for real-time Bitcoin price data.  
- **Pandas**: Useful for handling data analysis and manipulation tasks.  
- **Matplotlib/Plotly**: For visualization of time series data within Dolibarr’s dashboard.  
- **SQLite3/MySQL Connector**: Depending on the database setup, these libraries can be used to directly interact with Dolibarr’s database for storing and retrieving Bitcoin data.

### **Domo**

**Title**: Real-time Bitcoin Data Analysis using Apache Kafka

**Difficulty**: 3 (difficult)

**Description**:  
Apache Kafka is a distributed event streaming platform capable of handling trillions of events a day. Initially developed by LinkedIn, Apache Kafka is open-source software that provides a unified, high-throughput, low-latency platform for handling real-time data feeds. It can be used to build real-time streaming data pipelines that reliably get data between systems or applications. This project introduces Apache Kafka's core concepts, including producers, consumers, topics, and brokers.

**Describe technology**:

- **Producers**: Applications or systems that publish (write) messages or "events" to Kafka topics.  
- **Consumers**: Applications or systems that subscribe to (read) messages from Kafka topics.  
- **Topics**: Kafka stores streams of records (events) in categories called topics.  
- **Brokers**: Kafka runs as a cluster on one or more servers, and each server is called a broker. Each broker holds a part of the data within the Kafka cluster.  
- **Use case in this project**: In this project, Kafka will be used to ingest real-time data from a chosen Bitcoin data API. This data will be used to perform time series analysis for price prediction and trend monitoring.

**Describe the project**:

- **Setup and Configuration**:  
  - Install Kafka locally or set up a managed Kafka service.  
  - Configure Kafka topics for ingesting Bitcoin price data.  
  - Set up producers to fetch Bitcoin prices from a real-time API like CoinGecko.  
- **Data Ingestion and Storage**:  
  - Develop Python scripts to act as Kafka producers to periodically pull data from the API and publish it to Kafka topics.  
  - Create a Kafka consumer using Python that subscribes to these topics to fetch and store data into a suitable database for further analysis (e.g., PostgreSQL).  
- **Time Series Analysis**:  
  - Implement data preprocessing steps to clean and prepare the ingested data.  
  - Apply time series analysis techniques, such as ARIMA or Prophetic modeling, using libraries like statsmodels or fbprophet, to conduct price trend analysis and forecasting.  
- **Visualization**:  
  - Utilize tools such as Matplotlib or Plotly to graphically represent the forecasted data against real-time updates, showcasing predicted trends and cycles in Bitcoin pricing.

**Useful resources**:

- Apache Kafka Documentation: [Kafka Documentation](https://kafka.apache.org/documentation/)  
- Kafka Python Library Documentation: [confluent-kafka-python Docs](https://docs.confluent.io/platform/current/clients/confluent-kafka-python/index.html)  
- CoinGecko API Documentation: [CoinGecko API](https://www.coingecko.com/en/api)  
- Time Series Analysis with Python libraries: [statsmodels](https://www.statsmodels.org/), [fbprophet (or now called Prophet)](https://facebook.github.io/prophet/)

**Is it free?**:  
Apache Kafka is open-source and free to use. However, there might be costs associated with hosting solutions or managed services like Confluent Kafka if cloud services are preferred instead of a local setup.

**Python libraries / bindings**:

- **confluent-kafka**: This Python library is used to produce and consume messages from Kafka. Install it via `pip install confluent-kafka`.  
- **requests**: Used for making HTTP requests to the Bitcoin API. Install it via `pip install requests`.  
- **pandas**: Used for data manipulation and analysis. Install it via `pip install pandas`.  
- **statsmodels or fbprophet (Prophet)**: For time series forecasting modeling. Install via `pip install statsmodels` or `pip install prophet`.  
- **matplotlib / Plotly**: For data visualization. Install via `pip install matplotlib` or `pip install plotly`.

### **DoWhy**

**Title**: Real-Time Bitcoin Causal Analysis with DoWhy  
**Difficulty**: 2 (medium)

**Description**:

- Build a Python-based big data system that continuously ingests real-time Bitcoin price data and applies causal inference techniques using DoWhy.  
- The project aims to evaluate the impact of market events on Bitcoin prices by defining causal relationships and testing their robustness.  
- Designed to be completed in around 10 days, it challenges students to integrate streaming data ingestion, time series processing, and causal inference.

**Describe technology**:

- **DoWhy Library**:  
  - A Python package for causal inference that combines graphical models with the potential outcomes framework to estimate treatment effects.  
  - Allows users to define causal models using directed acyclic graphs (DAGs) and perform counterfactual analysis.**DoWhy**  
  -   
  - Supports various refutation tests to assess the reliability of causal conclusions.  
- **Example Usage**:  
  - Use DoWhy to model the effect of a regulatory announcement or major market event on Bitcoin prices, comparing pre-event and post-event periods while controlling for confounding variables.

**Describe the project**:

- **Data Ingestion**:  
  - Create a Python module to fetch real-time Bitcoin price data from a public API (e.g., CoinGecko) using libraries like `requests` or `websockets`.  
- **Data Processing**:  
  - Utilize pandas to clean and format the streaming data into a structured time series format suitable for analysis.  
- **Causal Model Setup**:  
  - Define the causal model by identifying the treatment (e.g., a market event), the outcome (Bitcoin price changes), and potential confounders (such as trading volume or market sentiment).  
- **Application of DoWhy**:  
  - Estimate the causal effect of the identified event on Bitcoin prices using DoWhy’s causal inference methods.  
  - Run refutation tests to validate the causal assumptions and robustness of the estimated effects.  
- **Visualization and Reporting**:  
  - Generate visualizations of the time series data and the inferred causal relationships using matplotlib or seaborn.  
  - Compile a report summarizing the causal analysis, including the defined causal graph, treatment effect estimates, and validation results.

**Useful resources**:

- [DoWhy Documentation](https://github.com/py-why/dowhy)  
- [pandas Documentation](https://pandas.pydata.org/docs/)  
- [matplotlib Documentation](https://matplotlib.org/stable/contents.html)  
- [CoinGecko API Documentation](https://www.coingecko.com/en/api)

**Is it free?**:

- Yes, the project uses open-source Python libraries and free-access APIs. DoWhy is available under an open-source license.

**Python libraries / bindings**:

- **DoWhy**: For causal inference and modeling.  
- **pandas**: For data ingestion, cleaning, and time series processing.  
- **matplotlib / seaborn**: For data visualization.  
- **requests**: For fetching real-time Bitcoin data.  
- **websockets** (optional): For handling streaming data if needed.

### **DVC**

**Title**: Real-Time Bitcoin Data Processing with DVC

**Difficulty: 2 (medium difficulty)**

**Description**  
The Data Version Control (DVC) tool is an open-source version control system for machine learning projects. It is designed to handle large datasets and manage machine learning models and experiments in a reproducible environment using version control techniques. In this project, you will gain practical experience in using DVC to manage and version your machine learning experiments and data related to real-time Bitcoin price analysis.

**Describe Technology**

- **DVC**: A version control system designed specifically for machine learning data and experiments. Key features include:  
  - Data Management: Track large datasets and ML models with lightweight metafiles without storing the actual data in Git.  
  - Reproducibility: Ensure experiments and results are reproducible and the pipeline stages are clearly defined and organized.  
  - Data Pipelines: Create and manage complex pipelines using a combination of stages and commands for processing data.

**Describe the Project**

- **Objective**: Implement a DVC-based system to manage real-time Bitcoin price data collection and processing, with an emphasis on versioning and reproducibility.  
- **Data Ingestion**:  
  - Use a public API such as CoinGecko to fetch real-time Bitcoin prices.  
  - Utilize basic Python libraries like `requests` to collect data at regular intervals.  
- **Data Processing**:  
  - Implement a time series analysis to track Bitcoin price changes over specific intervals.  
  - Use libraries like `pandas` for data manipulation and `matplotlib` for visualization of Bitcoin pricing trends.  
- **Pipeline Setup**:  
  - Define and set up a DVC pipeline to automate data ingestion, processing, and analysis.  
  - Ensure all stages of the pipeline are versioned for easy reproducibility.  
- **Experiment Tracking**:  
  - Utilize DVC to track multiple experiments by changing time intervals or analysis methods and compare their results.

**Useful Resources**

- **DVC Official Documentation**: [DVC Documentation](https://dvc.org/doc)  
- **CoinGecko API**: [CoinGecko API Documentation](https://www.coingecko.com/en/api)  
- **Python requests Library**: [Requests Documentation](https://docs.python-requests.org/)  
- **pandas Library**: [Pandas Documentation](https://pandas.pydata.org/docs/)  
- **matplotlib Library**: [Matplotlib Documentation](https://matplotlib.org/3.1.1/contents.html)

**Is it Free?**  
Yes, DVC is open-source software and free to use. Most of the additional Python libraries used in this project are also open-source and free.

**Python Libraries / Bindings**

- **DVC**: Install via `pip install dvc`  
- **requests**: For making API calls to fetch data, install via `pip install requests`  
- **pandas**: For data manipulation and analysis, install via `pip install pandas`  
- **matplotlib**: For data visualization, install via `pip install matplotlib`

### 

### **EconML**

**Title**: Bitcoin Time Series Analysis with EconML

**Difficulty**: 2 (medium difficulty)

**Description**  
In this project, students will explore the application of EconML, a Python library developed by Microsoft Research, for understanding and analyzing causal inference in economics. EconML is designed to interpret machine learning models to estimate causal effects in observational data. Students will leverage this library to perform a time series analysis on real-time Bitcoin price data. By the end of this project, students will gain hands-on experience with causal inference techniques, time series data manipulation, and the analysis of economic phenomena using Python.

**Describe technology**

- EconML: An open-source library for estimating heterogeneous treatment effects in Python based on methods from the field of causal inference.  
- Key Components:  
  - DML (Double Machine Learning) and DR (Doubly Robust) methods for causal effect estimation.  
  - Support for integrating with common machine learning tools such as scikit-learn.  
  - Ability to estimate and interpret causal effects in a variety of setups, particularly instrumental variable and panel data settings.

**Describe the project**

- **Objective**: Analyze real-time Bitcoin price data to identify causal relationships using EconML. The focus will be on understanding how various factors, potentially including market indicators or external economic data, impact Bitcoin prices.  
    
- **Steps**:  
    
  1. **Ingest Bitcoin Data**: Use a Python package like `requests` to fetch real-time Bitcoin price data from a public API (e.g., CoinGecko).  
  2. **Pre-process and Explore Data**: Utilize Pandas to clean and prepare the data, check time stamps, and remove any outliers or missing values.  
  3. **Apply EconML Models**: Implement EconML’s DML or DR learners to estimate the causal impact of one or more independent variables on Bitcoin prices over time.  
  4. **Time Series Analysis**: Leverage time-based features and analyze how causal effects vary over time. This may involve separating data into training and testing sets based on time, for example, pre- and post-event analysis.  
  5. **Results Interpretation**: Interpret the output of the model, examining causal effects and discussing potential real-world economic implications.

**Useful resources**

- EconML Documentation:   
- [https://econml.azurewebsites.net/](https://econml.azurewebsites.net/)  
- Tutorials and Examples: [https://github.com/py-why/EconML](https://github.com/py-why/EconML)  
- CoinGecko API Documentation: [https://www.coingecko.com/en/api](https://www.coingecko.com/en/api)

**Is it free?**  
Yes, EconML is an open-source package and freely available to use. However, any API queries may be subject to terms set by the data providers.

**Python libraries / bindings**

- `econml`: Install via pip (`pip install econml`) to access the library functionalities.  
- `pandas`: For data manipulation and preprocessing, install via pip (`pip install pandas`).  
- `requests`: To interact and fetch data from Bitcoin APIs.  
- `scikit-learn`: For integrating machine learning models with EconML. Install via pip (`pip install scikit-learn`).

### **ERPNext**

**Title**: Sales Trend Analysis and Reporting with ERPNext and Python  
**Difficulty**: 1 (Easy)

**Description**  
This project involves extracting sales data from ERPNext, analyzing trends using Python, and generating visual reports. Students will learn to interact with ERPNext's database/API and apply basic data analysis techniques.

**Describe Technology**

- **ERPNext**: Open-source ERP platform for managing business operations (sales, inventory, accounting).  
  1. Built on **Frappe Framework** (Python \+ MariaDB).  
  2. Use cases: Sales order tracking, inventory management, financial reporting.  
  3. Relevance: ERP systems are foundational for business data engineering.  
     

**Describe the Project**

1. **Set Up ERPNext (Local/Cloud)**:  
   - Install ERPNext locally or use a free cloud trial.  
   - Populate dummy sales data (e.g., 100+ sales orders with dates, items, prices).  
2. **Extract Data**:  
   - Option 1: Use ERPNext’s REST API (`requests` library) to fetch sales orders.  
   - Option 2: Directly query the MariaDB database with `pymysql` to extract sales data.  
3. **Analyze Sales Trends**:  
   - Clean data with `pandas` (e.g., handle missing values, format dates).  
   - Calculate metrics:  
     - Monthly revenue trends.  
     - Top-selling items.  
     - Customer purchase frequency.  
4. **Visualization**:  
   - Create bar charts (top-selling items) and line graphs (revenue trends) with `matplotlib`.  
   - Export results to a PDF report using `reportlab` or a Jupyter Notebook.

**Useful Resources**

- [ERPNext Documentation](https://docs.erpnext.com/)  
- [ERPNext REST API Guide](https://frappeframework.com/docs/user/en/api/rest)  
- [PyMySQL Tutorial](https://pynative.com/python-mysql-database-connection/)

**Is it free?**  
ERPNext is open-source (free for local use). Cloud trials are free for 14 days.

**Python Libraries / Bindings**

- `pandas` (data manipulation), `matplotlib` (visualization).  
- `requests` (API calls), `pymysql` (database connection).  
- `jupyter` (optional for interactive analysis).

- 

### 

### **Facebook prophet**

**Title**: Bitcoin Price Forecasting with Facebook Prophet

**Difficulty**: 3 (Difficult)

**Description**  
This project involves implementing a real-time Bitcoin price forecasting system using Facebook Prophet, a robust time-series forecasting library developed by Facebook's Core Data Science team. The aim is for students to ingest live Bitcoin price data from a public API, process it, and produce forecasts that can anticipate future price movements using machine learning models. This project requires knowledge in data science, Python programming, and time series analysis, providing students with hands-on experience in using sophisticated analytics tools to derive insights from financial data.

**Describe technology**

- **Facebook Prophet** is an open-source tool designed for time series forecasting. It is especially useful for data with daily observations that display patterns on different time scales. Prophet is known for its versatility and ease of use, making it possible for both experts and non-experts to work with time series data efficiently.  
- Core features include:  
  - **Automatic seasonality detection**: Handles yearly, weekly, and daily seasonality, including holiday effects.  
  - **Robust to missing data and shifts in the trend**: Useful for real-world data with irregularities.  
  - **Human-friendly**: Allows inclusion of holidays, custom seasonality terms, and offers extensive control over other model aspects.

**Describe the project**

- **Data Ingestion**:  
    
  - Use a public API, such as CoinDesk or CoinGecko, to continuously fetch real-time Bitcoin price data.  
  - Implement a data buffer to store the incoming data within a local or cloud-based database.


- **Data Processing**:  
    
  - Pre-process the ingested data to fill missing values, remove outliers, and format the data correctly for forecasting.  
  - Use basic Python libraries like pandas for data manipulation and cleaning.


- **Modeling**:  
    
  - Integrate Facebook Prophet to model the time-series data.  
  - Train the model using historical Bitcoin prices and include key features such as trend lines, daily/weekly/yearly seasonality.


- **Forecasting**:  
    
  - Generate future price forecasts and visualize these predictions alongside historical data using charting libraries like matplotlib or plotly.  
  - Implement a reporting module to create alerts or notifications based on significant forecast changes.


- **Evaluation**:  
    
  - Evaluate the performance of your forecasts by comparing them to actual price movements using metrics such as RMSE or MAE.

**Useful resources**

- [Facebook Prophet Documentation](https://facebook.github.io/prophet/)  
- [Kaggle Datasets for Historical Bitcoin Prices](https://www.kaggle.com/datasets)  
- [CoinDesk Developer API](https://www.coindesk.com/coindesk-api)  
- [Plotly Website](https://plotly.com/python/)

**Is it free?**  
Yes, Facebook Prophet is open source and can be used for free. The public APIs like CoinDesk or CoinGecko typically have a free tier for basic usage but may require registration to obtain an API key.

**Python libraries / bindings**

- **prophet**: Installable via `pip install prophet`, requires additional dependencies like pystan for fitting Bayesian models.  
- **pandas**: For data manipulation and cleaning tasks.  
- **matplotlib and plotly**: For data visualization and plotting results/forecasts.  
- **requests**: To handle API requests for real-time data ingestion.

### **Falcon**

**Title**: Scalable Real-Time Bitcoin Analytics with Falcon  
**Difficulty**: 3 (hard)

**Description**  
**Describe technology**  
Falcon is a high-performance, minimalist Python web framework designed for building ultra-fast APIs and microservices. It is optimized for low latency and high throughput, making it ideal for real-time data systems. Key features:

- Native support for ASGI/WSGI standards.  
  - Middleware for authentication, rate limiting, and logging.  
  - Async capabilities for non-blocking I/O operations.  
    Example: Building a REST API endpoint to ingest Bitcoin transaction data at scale.

**Describe the project**  
Develop a distributed real-time Bitcoin analytics platform using Falcon to handle high-frequency data ingestion, processing, and predictive modeling. The project involves:

1. **Real-Time Data Pipeline**:  
   - Integrate with WebSocket APIs (e.g., Coinbase Pro, Binance) to stream Bitcoin price/trade data.  
   - Use Falcon to create a high-throughput API endpoint (`/ingest`) to receive and validate data.  
2. **Distributed Processing**:  
   - Implement async workers (via Celery or Redis Queue) to parallelize tasks:  
     - Anomaly detection (e.g., sudden price deviations).  
     - Sentiment analysis integration (scrape Twitter data in parallel).  
3. **Predictive API**:  
   - Train a time-series forecasting model (e.g., Facebook Prophet, LSTM) on historical data.  
   - Expose a Falcon endpoint (`/predict`) to return predictions for the next 24 hours.  
4. **Scalability Challenges**:  
   - Containerize the API and workers with Docker.  
   - Stress-test the system using Locust to simulate 1,000+ concurrent requests.  
   - Implement rate limiting and caching (e.g., Redis) to optimize performance.

**Useful resources**

- Falcon Documentation: [https://falcon.readthedocs.io](https://falcon.readthedocs.io)  
- Celery Distributed Task Queue: [https://docs.celeryq.dev](https://docs.celeryq.dev)  
- Facebook Prophet Guide: [https://facebook.github.io/prophet/docs/quick\_start.html](https://facebook.github.io/prophet/docs/quick_start.html)

**Is it free?**   
Yes. Falcon, Celery, and Prophet are open-source. Docker and Redis have free tiers.

**Python libraries / bindings**

- falcon: Core API framework.  
- websockets: Real-time data ingestion.  
- celery: Distributed task processing.  
- prophet/keras: Time-series forecasting.  
- docker: Containerization.  
- prometheus-client: Monitoring.

### **FastAPI**

**Title**: Real-Time Bitcoin Price Analysis with FastAPI

**Difficulty**: 2 (medium difficulty)

**Description**  
FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.6+ based on standard Python-type hints. It allows developers to create efficient backend services quickly and intuitively, with an emphasis on performance and an easy-to-use syntax. This project focuses on utilizing FastAPI to ingest and process real-time Bitcoin price data, perform time series analysis, and provide an API for querying and visualizing these analyses.

**Describe technology**

- **Key Features**: FastAPI enhances productivity by enabling automatic generation of interactive API documentation (Swagger and ReDocs) and validations based on Python-type hints.  
- **Asynchronous Support**: FastAPI supports asynchronous programming features in Python, enhancing the I/O performance crucial for real-time data processing.  
- **Performance**: Built on Starlette for the web parts and Pydantic for the data parts, FastAPI is one of the fastest Python web frameworks.

**Describe the project**

- **Objective**: Implement a system using FastAPI to fetch real-time Bitcoin price data from a public API (e.g., CoinGecko or Binance), perform basic time series analyses (such as moving averages), and expose this data through a RESTful API.  
- **Steps**:  
  1. **Setting up FastAPI**: Create a FastAPI application that has endpoints for fetching, storing, and serving processed Bitcoin price data.  
  2. **Data Ingestion**: Use FastAPI to continuously fetch Bitcoin price data at regular intervals and store it in memory or a lightweight database like SQLite.  
  3. **Processing**: Implement basic time series analysis techniques to evaluate trends in the data, such as calculating moving averages or identifying patterns.  
  4. **Exposing Data**: Use FastAPI's capabilities to build RESTful endpoints, allowing users to query processed data and view the analyses in a structured format.  
  5. **Visualization**: Optionally, incorporate simple data visualization (e.g., using Plotly or Matplotlib for graphs) to enhance the API output.

**Useful resources**

- [FastAPI Official Documentation](https://fastapi.tiangolo.com/)  
- [CoinGecko API Documentation](https://www.coingecko.com/en/api)  
- [Asyncio for Python (Async programming guide)](https://docs.python.org/3/library/asyncio.html)  
- [Pydantic (Data validation and settings management)](https://pydantic-docs.helpmanual.io/)

**Is it free?**  
Yes, FastAPI is an open-source framework and is free to use. CoinGecko API also provides free access with rate limits for public data.

**Python libraries / bindings**

- **FastAPI**: Used to create the API service. Install using `pip install fastapi`.  
- **Uvicorn**: An ASGI server needed to run FastAPI applications. Install using `pip install uvicorn`.  
- **HTTPX or AIOHTTP**: For making asynchronous HTTP requests to fetch Bitcoin prices. Install using `pip install httpx` or `pip install aiohttp`.  
- **SQLite (built-in Python library)**: For lightweight storage of time-series data.  
- **Pandas**: Utilized for data manipulation and time series analysis. Install using `pip install pandas`.  
- **Plotly or Matplotlib**: For optional data visualization. Install using `pip install plotly` or `pip install matplotlib`.

### **FastText**

Title: Real-Time Bitcoin Analysis with FastText

**Difficulty**: 2 (medium)

**Description**  
FastText is an open-source library released by Facebook’s AI Research (FAIR) lab designed for efficient learning of word representations and text classification. It provides both unsupervised and supervised learning algorithms for creating word vectors that are highly performant in capturing syntactic and semantic relationships. FastText models both word-level and character-level details, making it adept at processing textual data with morphological differences, such as variations in word spellings.

**Describe technology**

- FastText allows creating word embeddings from textual data quickly and efficiently, even for large datasets.  
- It supports text classification which can be used for further analysis like sentiment analysis of texts.  
- FastText uses subword information to create word embeddings, which makes it robust against out-of-vocabulary issues and capable of handling synthetic tokens like cryptographic denominations.  
- It provides pre-trained language models in various languages, ready for deployment in diverse data processing tasks.

**Describe the project**  
For this project, you will use FastText to analyze real-time bitcoin-related textual data, such as tweets or news headlines, to cluster them based on sentiment (e.g., positive, negative, and neutral sentiment towards Bitcoin).

- First, use a public API, such as the Twitter API, to ingest real-time data streams related to Bitcoin.  
- Implement FastText's models to preprocess and vectorize the ingested text data, leveraging subword information for more precise representations.  
- Perform a supervised learning task for sentiment analysis using FastText’s text classification capabilities.  
- Visualize the time-based sentiment fluctuations, correlating them with Bitcoin price changes over the same periods.  
- Optionally, use the outcomes of the sentiment analysis to predict potential market movements, contributing to investment insights or strategies.

**Useful resources**

- FastText official documentation: [https://fasttext.cc/docs/en/support.html](https://fasttext.cc/docs/en/support.html)  
- Twitter API documentation: [https://developer.twitter.com/en/docs](https://developer.twitter.com/en/docs)  
- CoinGecko API documentation for market data: [https://coingecko.com/en/api](https://coingecko.com/en/api)

**Is it free?**  
Yes, FastText is completely free and open-source under the MIT license. However, access to real-time data through APIs like Twitter might require an account with possible billing depending on usage.

**Python libraries / bindings**

- **fasttext**: The Python package for FastText to efficiently train and use FastText models. You can install it using `pip install fasttext`. It's the primary library you'll use for text processing and classification within the project.  
- **tweepy**: A Python wrapper for the Twitter API, useful for fetching real-time tweets related to Bitcoin. Install it via `pip install tweepy`.  
- **pandas**: For data handling and manipulation; install with `pip install pandas`.  
- **matplotlib** and **seaborn**: For visualizations of time series data and sentiment analysis results. Install them using `pip install matplotlib seaborn`.  
- Any additional libraries, such as `requests` for HTTP requests to fetch data from web APIs.

### **Faust**

**Title**: Real-Time Bitcoin Analysis using Faust

**Difficulty**: 3 (difficult)

**Description**  
Faust is a Python stream processing library, inspired by Kafka Streams, that leverages the power and simplicity of Python to provide real-time data processing with minimal setup and robust results. It allows users to easily implement stream processing applications, conduct transformations, and perform complex operations on continuously arriving data streams.

This project involves utilizing Faust to build a real-time Bitcoin price analysis system. The focus will be on ingesting real-time Bitcoin price data, processing and analyzing it, and then performing time series analysis to visualize trends and generate insights. The challenge is to efficiently manage data streams and deal with high throughput while maintaining low latency in processing.

**Describe technology**

**Faust**: A stream processing library for Python, built on top of Kafka.  
**Core functionalities**:  
**Agents**: Create tasks that process streams of data.  
**Tables**: Persistent data structures for stateful stream processing.  
**Streams**: Ingest and emit data using Kafka topics.  
**Rebalancing**: Automatic load redistribution among nodes in a cluster.

**Describe the project**

**Objective**: Implement a system using Faust to ingest real-time Bitcoin price data from a public API (e.g., CoinGecko or Binance) and perform time series analysis.  
**Components**:  
**Data Ingestion**: Use Faust to create a streaming pipeline that continually fetches Bitcoin price data from the API.  
**Real-time Processing**: Develop an agent to transform the data, filter significant price changes, and compute rolling averages.

- **Time Series Analysis**: Enhance the system by implementing ARIMA models to predict price trends and visualize the results.  
- **Outputs**: The processed and analyzed data should be used to monitor price movements and detect anomalies or patterns, displayed via dynamic plots or dashboards.

**Useful resources**

- [Faust Documentation](https://faust.readthedocs.io/)  
- [Getting Started with Faust](https://faust.readthedocs.io/en/latest/playbooks/quickstart.html)  
- [Kafka Python Documentation](https://kafka-python.readthedocs.io/en/master/)  
- [ARIMA Model Details](https://otexts.com/fpp2/arima.html)

**Is it free?** Yes, Faust is open-source and free to use. However, setting up Kafka for production environments might involve costs, depending on the chosen solution (self-hosted vs. managed services).

**Python libraries / bindings**

- **Faust**: Install it using `pip install faust-streaming`. Utilize it for defining agents, tables, and streams.  
- **Kafka-Python**: Needed for integration with Apache Kafka, install via `pip install kafka-python`.  
- **Statsmodels**: For implementing ARIMA models, install using `pip install statsmodels`.  
- **Matplotlib/Seaborn**: For visualizations, install via `pip install matplotlib seaborn`.

By the end of this project, students will have gained practical insights into streaming data, performing complex analytics, and managing real-time data flow efficiently.

### **Fernet**

**Title**: Real-Time Bitcoin Data Processing with Fernet Encryption

**Difficulty**: 1 (easy)

**Description**  
Fernet is a part of the cryptography package in Python, which provides a way to securely encrypt and decrypt data. It ensures that the message encrypted cannot be read or altered without the encryption key. In this project, we will use Fernet encryption to securely process real-time Bitcoin price data.

**Describe technology**  
Fernet is a symmetric encryption method provided by the `cryptography` package in Python. It generates a unique key for encrypting and decrypting data, using advanced encryption mechanisms, including AES (Advanced Encryption Standard). The main functionalities of Fernet include:

- **Key Generation**: Creates a secure encryption key.  
- **Encryption**: Encrypts plaintext information, ensuring data security.  
- **Decryption**: Decrypts encrypted information back to plaintext.  
- **Token Management**: Handles secure tokens to ensure data is neither reused nor tampered with.

Example Usage:

```py
from cryptography.fernet import Fernet

# Generate a key for encryption
key = Fernet.generate_key()

# Initialize Fernet with the generated key
cipher = Fernet(key)

# Encrypt some data (e.g., a string representation of Bitcoin data)
encrypted_data = cipher.encrypt(b"Bitcoin price: $50000")

# Decrypt the data
decrypted_data = cipher.decrypt(encrypted_data)

print(decrypted_data.decode())  # Output: Bitcoin price: $50000
```

**Describe the project**  
The project focuses on ingesting real-time Bitcoin price data from a public API (e.g., CoinGecko or another open API) and securely processing this data using Fernet encryption. The steps for the project are:

1. **Data Ingestion**: Fetch real-time Bitcoin price data at regular intervals using basic HTTP requests.  
2. **Data Encryption**: Use Fernet to encrypt the fetched data immediately after retrieval.  
3. **Data Storage**: Store the encrypted data locally or in a simple database.  
4. **Data Decryption and Analysis**: Decrypt the data for basic time series analysis, such as calculating moving averages over a specified period.  
5. **Output**: Display time series analysis results on a simple console dashboard or save to a CSV file.

**Useful resources**

- [cryptography documentation](https://cryptography.io/en/latest/)  
- [Cryptography \- Fernet Documentation](https://cryptography.io/en/latest/fernet/)  
- [CoinGecko API](https://www.coingecko.com/en/api)

**Is it free?**  
Yes, Fernet is part of the open-source `cryptography` package for Python.

**Python libraries / bindings**

- **cryptography**: To install, use `pip install cryptography`. It's required to use Fernet for encrypting and decrypting data.  
- **requests**: To fetch real-time Bitcoin prices via API calls. Install using `pip install requests`.

This project will provide students a practical introduction to data security in Python, emphasizing encryption's importance when handling sensitive and real-time data.

### **FireDucks**

**Title**: High-Speed Bitcoin Time Series Analysis with FireDucks  
**Difficulty**: 1 (easy)  
**Description**  
**Describe technology**  
**FireDucks** is a compiler-accelerated Python dataframe library optimized for speed and pandas compatibility. Key features:

- **Pandas-like syntax**: Seamlessly replace `pandas` with minimal code changes.  
- **Query optimization**: Automatic parallelization and efficient memory management.  
- **TPU/CPU acceleration**: Leverage multi-core systems (e.g., Google Colab TPUs) for large datasets.  
  Example: Process 1M+ Bitcoin price records 5-10x faster than vanilla pandas.


**Describe the project**  
Build a real-time Bitcoin price analysis pipeline using FireDucks to demonstrate its performance advantages over pandas. Steps:

1. **Setup**:  
   - Use Google Colab with a **v2-8 TPU runtime** (high CPU cores/memory).  
   - Install FireDucks (`pip install fireducks`) and enable its import hook.  
2. **Data Ingestion**:  
   - Fetch hourly Bitcoin price data (Jan 2023–present) from CoinGecko API.  
   - Load into FireDucks DataFrame:

   

```py
%load_ext fireducks.pandas  # Magic command for Jupyter  
import fireducks.pandas as pd  
btc_df = pd.read_csv("https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=365")  
```

   

3. **Time Series Analysis**:  
   - Resample daily prices and compute:  
     - 30-day rolling volatility.  
     - Weekly average closing price.  
   - Compare execution time vs pandas (using `%%timeit` in Jupyter).  
4. **Visualization**:  
   - Plot price trends and volatility with `matplotlib`.  
   - Bonus: Process 10x larger synthetic dataset to stress-test FireDucks.

   

**Useful resources**

- [FireDucks GitHub](https://github.com/fireducks/fireducks) (TPU demo notebooks included)  
- [CoinGecko API Docs](https://www.coingecko.com/en/api)  
- [FireDucks vs Pandas vs Polars Demo](https://github.com/fireducks/fireducks/blob/main/examples/FireDucks_vs_Pandas_vs_Polars.ipynb)

**Is it free?**  
Yes. FireDucks is BSD-licensed, and CoinGecko’s API has a free tier. Google Colab TPUs are free for basic usage.

**Python libraries / bindings**

- `fireducks`: Core library (replace `pandas`).  
- `requests`: Fetch CoinGecko data.  
- `matplotlib`: Visualization.  
- 

### **Flask**

**Title**: Real-Time Bitcoin Monitoring with Flask

**Difficulty**: 2 (medium difficulty)

**Description**  
This project entails creating a web application using Flask, a lightweight web framework for Python, to ingest and process real-time Bitcoin data. Flask is known for its simplicity and flexibility, making it an ideal choice for creating web applications and APIs. The project focuses on using Flask to develop a real-time data processing solution where students fetch, display, and analyze Bitcoin price data. Through this project, students will gain hands-on experience in building a web application and performing basic time series analysis using Python libraries.

**Describe technology**

- **Flask**: Flask is a micro web framework written in Python. It is designed to be straightforward and easy to use, allowing developers to quickly create web applications or APIs. Flask provides essential functionalities such as routing, request handling, and templating but leaves out complexities of larger frameworks, promoting modular and adaptable development.  
  - Example functionalities:  
    - **Routing**: Define URL routes to handle client requests.  
    - **Templating**: Use Jinja2 to render dynamic HTML pages.  
    - **Request Handling**: Manage HTTP methods like GET and POST.

**Describe the project**

- The goal is to develop a Flask-based web application that:  
  1. **Ingests Real-Time Bitcoin Data**: Set up a routine using Flask's task scheduling (e.g., with APScheduler) to fetch live Bitcoin price data from a public API like CoinGecko or CryptoCompare at regular intervals.  
  2. **Data Storage**: Store the fetched data in a local database (such as SQLite) to manage and query historical prices.  
  3. **Data Display**: Create a simple web interface to display real-time Bitcoin price charts and allow users to query past data.  
  4. **Time Series Analysis**: Implement basic time series analysis functionalities within Flask, such as computing moving averages or identifying trends in Bitcoin prices over time.  
  5. **API Integration**: Optionally, expose an API endpoint using Flask that returns processed data (e.g., average price over a specified period) to external clients.

**Useful resources**

- [Flask Documentation](https://flask.palletsprojects.com/en/latest/)  
- [Real Python Flask Tutorial](https://realpython.com/tutorials/flask/)  
- [CoinGecko API Documentation](https://www.coingecko.com/en/api)  
- [CryptoCompare API Documentation](https://min-api.cryptocompare.com/documentation)

**Is it free?**  
Yes, Flask is an open-source framework and free to use. Public APIs like CoinGecko offer free tiers, although they may have rate limits.

**Python libraries / bindings**

- **Flask**: Main framework for building the web application.  
- **Requests**: To handle HTTP requests to fetch Bitcoin data. Install via `pip install requests`.  
- **SQLite3**: Built-in Python library for database operations.  
- **Matplotlib/Plotly**: For plotting Bitcoin price data. Install via `pip install matplotlib` or `pip install plotly`.  
- **pandas**: For data manipulation and basic time series analysis. Install via `pip install pandas`.  
- **APScheduler**: For scheduling periodic tasks within the Flask app. Install via `pip install apscheduler`.

### **Gensim**

**Title**: Real-Time Bitcoin Data Processing with Gensim

**Difficulty**: 3 (difficult)

**Description**  
The project focuses on using Gensim, a powerful Python library to analyze real-time Bitcoin data. Gensim is popular for natural language processing (NLP) tasks, specifically for topic modeling, document similarity, and word embedding. This project will explore Gensim's capabilities in processing time-series data, and students will apply its functionalities to perform complex analyses on Bitcoin price trends. The objective is to ingest real-time Bitcoin data using standard Python packages, transform it using Gensim to draw insights, and implement time-series analysis.

**Describe Technology**

- **Gensim**: Gensim is a robust library designed primarily for topic modeling and document similarity analysis using NLP. Key features include training word2vec, doc2vec models, and creating topic models using Latent Dirichlet Allocation (LDA). Although Gensim isn't commonly associated with time-series data, its vector space modeling can be ingeniously adapted for this purpose.  
- **Key Functionalities**:  
  - Topic modeling using LDA and LSI  
  - Creating document vectors using Doc2Vec  
  - Word Embedding using Word2Vec and FastText  
  - Efficient Similarity Queries  
- **Use in this Project**:  
  - Transform time-series data (Bitcoin prices) into vector space representation  
  - Identify trends or emerging patterns as "topics" in dataset  
  - Use cosine similarity to compare different time periods

**Describe the Project**

- **Data Ingestion**:  
  - Use Python’s `requests` or `websockets` to fetch real-time Bitcoin data from APIs such as CoinGecko or Binance.  
- **Data Transformation**:  
  - Pre-process the time-series data to convert Bitcoin price changes into a suitable format for analysis  
  - Segment data into suitable time intervals (e.g., 5-minute windows)  
- **Vectorization**:  
  - Use Gensim to transform each data segment into a vector  
  - Analyze these vectors to infer trends, highlighting price volatilities or significant market shifts  
- **Analysis**:  
  - Model the "topics" that are indicative of price dynamics  
  - Use similarity measures to find analogous price movement periods  
- **Outcome**:  
  - Provide comprehensive insights into Bitcoin pricing trends over time

**Useful Resources**

- [Gensim Documentation](https://radimrehurek.com/gensim/)  
- [Gensim Tutorials and Examples](https://radimrehurek.com/gensim/auto_examples/index.html)  
- [CoinGecko API Documentation](https://www.coingecko.com/en/api)  
- [Binance API Documentation](https://binance-docs.github.io/apidocs/spot/en/)

**Is it Free?**  
Yes, Gensim is an open-source library that is completely free to use. Data collection through APIs like CoinGecko or Binance is generally free, though they might have rate limits or require registration for API keys.

**Python Libraries / Bindings**

- **Gensim**: Primary library for modeling and analysis. Installable via `pip install gensim`.  
- **NumPy** and **SciPy**: For numerical computing and any mathematical operations required. Installable with `pip install numpy scipy`.  
- **Pandas**: Used for data manipulation and transforming API response into an analyzable format. Install with `pip install pandas`.  
- **Requests** or **Websockets**: Used for making HTTP requests to API endpoints or establishing WebSocket connections for real-time data. Install using `pip install requests` or `pip install websockets`.

### **Geopandas**

**Title**: Visualizing Bitcoin Price Trends Using Geopandas

**Difficulty**: 1 (easy)

**Description**  
This project involves creating a simple system for ingesting real-time Bitcoin price data and visualizing it using Geopandas, which is a Python library specifically designed for handling geospatial data. Geopandas extends the data types used by pandas to allow spatial operations on geometric types. This task will guide students through the basics of Geopandas and how it can be used to visualize data in a geographic context.

**Describe technology**

- **Geopandas** is a Python library that makes working with geospatial data in Python easier. It extends the capabilities of pandas to support spatial data operations.  
- Geopandas allows you to read common geospatial file formats including ESRI shapefile, GeoJSON, TopoJSON, and others using the `geopandas.read_file()` function.  
- Once data is imported, Geopandas provides powerful operations for data analysis such as spatial joins, geometric manipulations, and data visualization.

**Describe the project**

- **Objective**: To design a system that ingests real-time Bitcoin price data and visualizes daily trends on a map using Geopandas.  
- **Data Ingestion**  
  - Use an API like CoinGecko to fetch the real-time Bitcoin price data. You will use Python requests library to handle the API requests.  
- **Data Processing**  
  - Store the extracted data in a convenient format using pandas DataFrames.  
  - Perform simple data manipulation to format timestamps and calculate required metrics like moving averages.  
- **Data Visualization**  
  - Utilize Geopandas to plot a geographic visualization of Bitcoin price changes.  
  - For simplicity, you will visualize the data as a simple time series overlay on a geographic plot (though not typically geospatial, this is an exercise in using Geopandas and plots).  
  - Enhance the plot with matplotlib to show time series trends of Bitcoin prices over a specific period.

**Useful resources**

- [Geopandas Documentation](https://geopandas.org/)  
- [CoinGecko API Documentation](https://www.coingecko.com/en/api)  
- [Python Requests Documentation](https://docs.python-requests.org/en/master/)  
- [Pandas Documentation](https://pandas.pydata.org/pandas-docs/stable/)

**Is it free?**  
Yes, Geopandas is an open-source library. CoinGecko's API also provides a free tier suitable for educational projects.

**Python libraries / bindings**

- **Geopandas** for geospatial data manipulation and visualization. Install using `pip install geopandas`.  
- **Pandas** for general data manipulation and analysis. Install using `pip install pandas`.  
- **Matplotlib** for creating static, interactive, and animated plots. Install using `pip install matplotlib`.  
- **Requests** for making HTTP requests to call the CoinGecko API. Install using `pip install requests`.

### **Google**

**Title**: Real-time Bitcoin Price Analysis using Google Cloud Functions  
**Difficulty**: 1 (easy)

**Description**:  
Google Cloud Functions is a serverless execution environment for building and connecting cloud services. With Cloud Functions, you can write simple, single-purpose functions that are attached to events emitted from your cloud infrastructure and services. By using Google Cloud Functions, you can efficiently process real-time Bitcoin price data, responding quickly to updates, without the need to manage server infrastructure.

**Describe technology**:

- **Google Cloud Functions**: Allows you to run your code in response to events without provisioning servers. Functions can be triggered by HTTP requests, Cloud Pub/Sub messages, and other Google Cloud services.  
- **Core Concepts**:  
  - **Event-driven**: Functions execute in response to triggers from supported Google Cloud services.  
  - **Auto-scaling**: Automatically scales based on the load.  
  - **Pay-per-use**: Charges based on actual usage – only pay for the time your code runs.

**Describe the project**:  
This project involves creating a simple real-time data ingestion and processing system using Google Cloud Functions to analyze Bitcoin price data.

1. **Objective**: Implement a serverless solution to ingest real-time Bitcoin prices from a public API, such as CoinAPI or CoinGecko.  
2. **Steps**:  
   - **Set Up Cloud Function**: Create a Google Cloud Function that is triggered by an HTTP request to fetch Bitcoin prices continuously.  
   - **Data Processing**: Process the incoming data to extract necessary information, such as current price, timestamp, and compare it to previous data points to find trends.  
   - **Storage**: Use Cloud Firestore or Cloud Storage to store the ingested and processed data for future analysis.  
   - **Time Series Analysis**: Perform basic time series analysis, such as calculating moving averages or identifying price spikes.  
3. **Outcome**: Gain hands-on experience in setting up real-time data processing systems using serverless architecture, understand the basics of time series analysis, and familiarize with Google Cloud Platform.

**Useful resources**:

- [Google Cloud Functions Documentation](https://cloud.google.com/functions/docs)  
- [CoinGecko API Documentation](https://www.coingecko.com/en/api)  
- [Understanding Time Series Analysis](https://towardsdatascience.com/an-introduction-to-time-series-analysis-1197a97a4f85)

**Is it free?**:  
Google Cloud Functions offers a free tier, which should be sufficient for small-scale applications like this project. However, some usage may incur charges if the free-tier limits are exceeded.

**Python libraries / bindings**:

- **Requests**: To send HTTP requests and fetch data from the Bitcoin price API. Install it using `pip install requests`.  
- **google-cloud-functions**: While not a separate package, your function will be deployed and managed via the Google Cloud console or SDK.  
- **pandas**: For processing and analyzing time series data. Install using `pip install pandas`.  
- **Firestore or Google Cloud Storage client library**: For storing the data. Install using `pip install google-cloud-firestore` or `pip install google-cloud-storage` depending on the chosen storage solution.


### **Google Cloud Platform**

**Title**: Scalable Bitcoin Data Pipeline on Google Cloud Platform

**Difficulty**

- **Level:** 3 

**Description**

- **Technology Overview:**  
  Google Cloud Platform (GCP) offers cloud services for storage, compute, machine learning, and data analytics, including Google Pub/Sub, BigQuery, AI Platform, and Data Studio. This project leverages GCP for a scalable Bitcoin data pipeline.  
    
- **Project Details:**  
  This project involves:  
    
  - Using Google Pub/Sub to ingest real-time Bitcoin data (prices, transactions, social media sentiment).  
  - Storing and processing data in Google BigQuery for historical analysis and real-time queries.  
  - Implementing time series analysis (e.g., price forecasting) using Google AI Platform or BigQuery ML.  
  - Detecting anomalies in transaction data with machine learning models.  
  - Visualizing results in Google Data Studio with real-time dashboards.  
  - Ensuring scalability and cost-effectiveness using GCP’s managed services.


  The complexity lies in integrating multiple GCP services, handling large-scale data, and optimizing for performance and cost.

**Useful Resources**

- [Google Cloud Documentation](https://cloud.google.com/docs)  
- [Google Pub/Sub](https://cloud.google.com/pubsub/docs)  
- [Google BigQuery](https://cloud.google.com/bigquery/docs)  
- [Google AI Platform](https://cloud.google.com/ai-platform/docs)  
- [Google Data Studio](https://datastudio.google.com/)

**Is it Free?**

- **GCP:** Free tier with limited resources; additional usage incurs costs.

**Python Libraries**

- `google-cloud-pubsub`: `pip install google-cloud-pubsub`  
- `google-cloud-bigquery`: `pip install google-cloud-bigquery`  
- `google-cloud-aiplatform`: `pip install google-cloud-aiplatform`  
- `pandas`: `pip install pandas`  
- `matplotlib`: `pip install matplotlib` (local plotting if needed)

### **Google Cloud Python Client**

**Title**: Ingest bitcoin prices using Google Cloud Python Client

**Difficulty**: 1 (easy)

**Description**  
In this project, students will explore the Google Cloud Python Client, a library designed to simplify interactions with Google Cloud Services. This project aims to provide hands-on experience in ingesting and processing real-time Bitcoin price data using Google Cloud Pub/Sub and Google Cloud Functions. The project is suitable for students with basic Python skills and an interest in cloud-based data processing solutions.

**Describe technology**  
The Google Cloud Python Client is a set of Python libraries that provide access to Google Cloud Services in a simple and efficient manner. For this project, the primary focus will be on Google Cloud Pub/Sub for real-time data ingestion and Google Cloud Functions for creating serverless functions to process data. Pub/Sub is a messaging service that allows for asynchronous data streaming, while Cloud Functions offers a lightweight, event-driven compute solution.

**Describe the project**  
The project involves:

- Setting up a Pub/Sub topic in Google Cloud to receive Bitcoin price data from a public API (e.g., CoinGecko).  
- Creating a Python-based Cloud Function to subscribe to the Pub/Sub topic, process incoming messages, and perform simple transformations, such as filtering data by time intervals or converting prices into different currencies.  
- Storing the processed data in Google Cloud Storage for subsequent analysis and time series visualization.  
- Optionally, students can use basic Python visualization libraries (e.g., Matplotlib) to create time series charts of Bitcoin price trends over time.

This straightforward project will be completed in about one week and provides practical exposure to Google Cloud services and real-time data processing workflows.

**Useful resources**

- [Google Cloud Python Client Documentation](https://googleapis.dev/python/google-api-core/latest/index.html)  
- [Getting Started with Google Cloud Pub/Sub](https://cloud.google.com/pubsub/docs/quickstarts)  
- [Google Cloud Functions Documentation](https://cloud.google.com/functions/docs)  
- [CoinGecko API Documentation](https://www.coingecko.com/en/api)

**Is it free?**  
Google Cloud offers a free tier with limited usage, including Pub/Sub and Cloud Functions. Students can use this for the project, but they should be aware of usage limits to avoid incurring charges.

**Python libraries / bindings**

- `google-cloud-pubsub`: This library is essential for interacting with Pub/Sub topics and subscriptions. Install it using `pip install google-cloud-pubsub`.  
- `google-cloud-functions`: Though Cloud Functions are managed on GCP, students will write their serverless functions in Python. Understanding this service will be key to integrating the solution.  
- Basic Python packages such as `requests` for fetching data from APIs and `matplotlib` for data visualization.

### **Google Cloudpickle**

**Title**: Advanced Time Series Analysis with Google Cloudpickle

**Difficulty**: 3 (Difficult)

**Description**

Google Cloudpickle is a Python library that extends the standard functionalities of Python's built-in `pickle` module. It is utilized to serialize (pickle) Python objects that are otherwise not serializable using the default pickle protocols. This includes functions, classes, and instances that are dynamically defined or involve closures, making Cloudpickle indispensable for distributed computing environments and complex data workflows.

In this project, students will implement a robust time series analysis pipeline for real-time Bitcoin price data using Cloudpickle to manage and serialize complex Python objects involved in the workflow. The project involves ingesting Bitcoin price data from a public API, performing real-time time series analysis, and managing objects throughout distributed nodes, offering insights into Python's distributed computing capabilities using Cloudpickle.

**Describe technology**

- **Serialization**: Cloudpickle enables serialization of complex Python objects, including functions and classes, which are not possible with the standard pickle module.  
- **Distributed Computing**: It facilitates distributed Python computational frameworks by allowing worker nodes to deserialize functions and associated data.  
- **Dynamic Environments**: Supports environments where code is dynamically created or modified, crucial in real-time data processing tasks.

**Describe the project**

1. **Data Ingestion**: Start by setting up a data ingestion pipeline using a Python package like `requests` to fetch Bitcoin price data from a public API such as CoinDesk or CoinGecko.  
     
2. **Data Serialization**: Use Cloudpickle to serialize complex Python objects involved in the data transformation process. This includes serialization of functions and classes that process and analyze Bitcoin price data.  
     
3. **Time Series Analysis**: Implement time series analysis techniques on the ingested data, such as moving averages, trend analysis, or anomaly detection. Use libraries like Pandas for time series manipulation and Matplotlib for visualization.  
     
4. **Distributed Processing**: Simulate a distributed computing environment using Python's `multiprocessing` module, leveraging Cloudpickle to distribute serialized functions and data across processes.  
     
5. **Results and Reporting**: Store results of the analysis and generate dynamic reports. Utilize Cloudpickle to serialize the final objects for persisting analysis results or sharing across different nodes.  
     
6. **Challenges**: Discuss challenges like managing dependencies and ensuring compatibility across different Python environments in a distributed setup.

**Useful resources**

- [Cloudpickle GitHub Repository](https://github.com/cloudpipe/cloudpickle): Official repository with documentation and examples.  
- [Pandas Documentation](https://pandas.pydata.org/pandas-docs/stable/): For understanding and working with time series data in Python.  
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html): To explore various visualization methodologies for time series data.

**Is it free?**

Yes, Google Cloudpickle is open-source and free to use.

**Python libraries / bindings**

- `cloudpickle`: Install using `pip install cloudpickle`. It provides essential serialization functionalities.  
- `requests`: Basic library to fetch API data. Install using `pip install requests`.  
- `pandas`: Used for data manipulation and time series analysis. Install using `pip install pandas`.  
- `matplotlib`: For creating static, animated, and interactive visualizations. Install using `pip install matplotlib`.  
- `multiprocessing`: A Python package for parallel processing using numerous processors. Part of Python’s standard library.

### **Graphene**

**Title**: Analyzing Bitcoin Trends with Graphene

**Difficulty**: 1 (Easy)

**Description**  
This project involves using Graphene, a lightweight and powerful Python framework, to handle and analyze real-time Bitcoin price data. Students will learn about fundamental aspects of Graphene and apply this knowledge to ingest, process, and analyze time-series data related to Bitcoin prices. The project offers a basic introduction to Graphene and data handling in Python, suitable for those new to big data systems and time series analysis.

**Describe Technology**

- **Graphene**:  
  - Graphene is a popular framework for building GraphQL APIs in Python. It's known for its simplicity and ease of use, facilitating the creation of robust data models and query handling.  
  - Core Concepts:  
    - **Schema**: Defines the data model and operations in GraphQL.  
    - **Resolvers**: Functions that fulfill the data fetching requirements.  
    - **Queries and Mutations**: Used for data retrieval and modification, respectively.  
  - Graphene supports Python's native async features for real-time data handling.

**Describe the Project**

- Objective: Use Graphene to implement a mini-server that listens to real-time Bitcoin price updates via a public API, processes this data, and makes it available for analysis through a GraphQL interface.  
- Steps:  
  - Set up a basic Python environment with Graphene.  
  - Implement a data ingestion script to fetch real-time Bitcoin price data from a public API like CoinGecko.  
  - Define a GraphQL schema using Graphene, including types for Bitcoin data such as price, timestamp, and volume.  
  - Create resolvers that handle incoming Bitcoin price data and store it in a Python data structure (like a list or a simple database).  
  - Implement queries that allow users to perform basic time-series analysis on this data, such as retrieving price changes over a specific period or calculating average price.  
  - Test your GraphQL server locally, ensuring it appropriately responds to queries and updates in real-time data.  
- The project culminates in a demonstration of querying real-time Bitcoin price trends via GraphQL.

**Useful Resources**

- [Graphene Documentation](https://docs.graphene-python.org/en/latest/)  
- [Introduction to GraphQL](https://graphql.org/learn/)  
- [Python Requests Library](https://docs.python-requests.org/en/master/)

**Is it free?**  
Yes, Graphene is open-source and freely available. No additional software costs are associated with this project.

**Python Libraries / Bindings**

- **Graphene**: The main library to implement GraphQL APIs. Install with `pip install graphene`.  
- **Requests**: For making HTTP requests to fetch data from external APIs. Install with `pip install requests`.  
- **Asyncio**: A standard Python library for asynchronous programming, used for handling real-time data updates.

This project offers hands-on experience with Graphene and helps students develop skills in building GraphQL APIs and time-series analysis using real-world data.

### **Great Expectations**

Title: Monitoring Bitcoin Prices Using Great Expectations

Difficulty: Medium (2=medium difficulty)

**Description**

Great Expectations is an open-source data validation and documentation framework that helps ensure data quality through automated testing and data profiling. With Great Expectations, data teams can create expectations for their data, validate it against these expectations, and generate detailed data documentation. This framework seamlessly integrates into Python data workflows and can be used to ensure the integrity of both batch and real-time data. This project will involve using Great Expectations to validate and monitor real-time Bitcoin price data from a public API.

**Describe technology**

- **Core Concepts**: Understand how Great Expectations empowers users to define, validate, and document data expectations directly in Python scripts. It provides a framework for writing portable, reusable, and shared data validation tests that help maintain data quality.  
- **Key Features**:  
  - **Expectations**: Specify what the data should look like or behave, such as acceptable ranges or expected distributions.  
  - **Validation**: Automatically test data against the expectations.  
  - **Data Documentation**: Generate human-readable documents detailing expectations of datasets and their validation status.  
- **Integration**: Great Expectations can be integrated with various Python data tools and backend storage options to support comprehensive data validation workflows.

**Describe the project**

In this project, students will leverage Great Expectations to implement a real-time monitoring system for Bitcoin price data:

1. **Data Ingestion**: Use Python to set up ingestion of real-time Bitcoin data from a public API like CoinGecko.  
2. **Expectation Suite**: Develop a suite of data expectations targeting key aspects such as:  
   - Expected price range to flag prices that exceed predefined thresholds.  
   - Time intervals to ensure the data is regularly updated and retrieved.  
3. **Validation Workflow**: Implement a validation pipeline to continuously check ingested data against the expectation suite.  
4. **Documentation and Alerts**: Use Great Expectations to create comprehensive data documentation and set up alerts for expectation failures, providing early warnings for potential data issues.  
5. **Time Series Analysis**: Integrate time series analysis to predict Bitcoin price trends or sudden volatility changes, using data validated by Great Expectations as reliable inputs for analysis.

**Useful Resources**

- [Great Expectations Documentation](https://docs.greatexpectations.io/)  
- [CoinGecko API Documentation](https://www.coingecko.com/en/api)  
- Online tutorials and guides for setting up Great Expectations with real-time data.

**Is it free?**

Yes, Great Expectations is open-source and free to use. However, data source APIs like CoinGecko may have usage limits or require an API key.

**Python libraries / bindings**

To implement this project, the following Python libraries are essential:

- **Great Expectations**: Install via `pip install great_expectations`. This library will be used to create, validate, and document data expectations.  
- **Requests or httpx**: Libraries for making HTTP requests to the Bitcoin price API. Install via `pip install requests` or `pip install httpx`.  
- **Pandas**: For handling and processing data frames, if necessary in the workflow. Install via `pip install pandas`.  
- **Matplotlib or Seaborn**: Optional libraries for visualizing data trends or time series analysis. Install via `pip install matplotlib seaborn`.

This project gives students practical experience with data validation and monitoring, using Python and Great Expectations to ensure real-time data integrity while working with time-series data analysis.

### **Griptape**

**Title**: Analyze Real-time Bitcoin Data with Griptape

**Difficulty**: 2 (Medium)

**Description**  
Griptape is a Python library geared towards simplifying the development of AI workflows by integrating LLMs (Large Language Models). It offers flexibility in defining and handling data processing pipelines, facilitating clean and reusable code structures. The library focuses on modularity, allowing users to construct complex data transformations and integrations easily. This medium-difficulty project involves utilizing Griptape to create a real-time data processing system focusing on Bitcoin prices. Students will learn to ingest data via a public API, perform various time-series analyses, and visualize trends leveraging Griptape’s capabilities.

**Describe technology**

- Griptape enables simplified development of AI workflows with its modular and composable Python components.  
- It is designed for seamless integration with large language models and can be adapted for diverse data processing tasks.  
- The core functionalities include pipeline creation, component plug-ins for data transformations, and visualization supports.  
- By abstracting complex operations, Griptape streamlines workflow creation, which is particularly useful when dealing with real-time data processing and analytics.

**Describe the project**

- The goal is to build a system that ingests real-time Bitcoin price data using a public API like CoinGecko, processes it for time-series analysis, and displays insights.  
- Students will start by setting up a Griptape pipeline to periodically fetch Bitcoin price data. They'll configure this to retrieve the data at regular intervals, ensuring the system supports continuous operation.  
- The next step involves defining and implementing components within the Griptape framework for tasks such as data normalization, anomaly detection, and feature extraction for time-series analysis.  
- Students will focus on key functions like moving averages, volatility indexing, or other common financial indicators, which will serve as inputs to a time-series analysis model.  
- To conclude, students must visualize their findings using Python plotting libraries, displaying trend lines, forecasted prices, and any anomalies detected over time.

**Useful resources**

- [Griptape GitHub Repository](https://github.com/griptape-ai/griptape)  
- [CoinGecko API Documentation](https://www.coingecko.com/en/api)  
- [Time-series Analysis with Python](https://machinelearningmastery.com/time-series-forecasting-supervised-learning/)  
- [Python Data Visualization Libraries](https://matplotlib.org/stable/plot_types/index.html)

**Is it free?**

- Yes, Griptape is open-source and available for free use. CoinGecko’s API also offers free access options with certain limitations.

**Python libraries / bindings**

- Griptape: For pipeline and workflow creation. Installable via pip with `pip install griptape`.  
- Requests: For interacting with the CoinGecko API (`pip install requests`).  
- Pandas: For data manipulation and time-series operations (`pip install pandas`).  
- Matplotlib/seaborn: For data visualization (`pip install matplotlib` and `pip install seaborn`).  
- Scikit-learn: For implementing any additional time-series modeling or processing (`pip install scikit-learn`).

### **Gym**

**Title**: Real-time Bitcoin Analytics with Apache Kafka

**Difficulty**: 1 (easy)

**Description** This project introduces students to Apache Kafka, a distributed streaming platform that is used for building real-time data pipelines and streaming applications. Students will learn how to utilize Kafka's capabilities to ingest Bitcoin price data in real-time and perform basic processing. The project will emphasize understanding Kafka's core components, such as topics, producers, consumers, and brokers. Additionally, the project will focus on using basic Python packages for data handling and analysis.

**Describe technology**

- **Apache Kafka**: An open-source stream-processing software platform developed by LinkedIn and donated to the Apache Software Foundation. Kafka is designed to handle live data feeds, providing a robust messaging system with high throughput, reliability, and low latency.  
  - **Core Components**:  
    - *Topics*: Categories to which records are published.  
    - *Producers*: Send data to topics.  
    - *Consumers*: Subscribe to topics to receive data.  
    - *Brokers*: Servers handling data transfers.  
- The focus will be on understanding and implementing the necessary components to handle real-time data flow efficiently.

**Describe the project**

- **Objective**: Create an application that continuously ingests real-time Bitcoin prices using Apache Kafka and performs basic time series analysis, such as calculating simple moving averages.  
- **Steps**:  
  1. **Kafka Setup**: Set up a local Kafka environment using Docker and configure the necessary components to start ingesting data.  
  2. **Data Ingestion**: Use a Python script to act as a Kafka producer, fetching real-time Bitcoin price data from a public API (e.g., CoinGecko) and sending it to a Kafka topic.  
  3. **Data Consumer**: Develop a Kafka consumer in Python using basic packages like `kafka-python` to read from the Kafka topic.  
  4. **Data Processing**: Implement basic time series analysis, such as calculating moving averages, using Pandas.  
  5. **Visualization**: (Optional) Visualize the processed data using Python libraries like Matplotlib or Plotly.  
- The project will help students gain hands-on experience in setting up a simple real-time data ingestion and processing pipeline.

**Useful resources**

- [Apache Kafka Quickstart Guide](https://kafka.apache.org/quickstart)  
- [CoinGecko API](https://www.coingecko.com/en/api/documentation)  
- [Kafka-Python Documentation](https://kafka-python.readthedocs.io/en/master/)  
- [Pandas Documentation](https://pandas.pydata.org/pandas-docs/stable/)

**Is it free?** Yes, Apache Kafka is free and open-source. The CoinGecko API is also free for simple use cases.

**Python libraries / bindings**

- `kafka-python`: A Python client for Apache Kafka. Install using `pip install kafka-python`.  
- `requests`: For fetching data from the Bitcoin price API. Install using `pip install requests`.  
- `pandas`: To perform time series analysis. Install using `pip install pandas`.  
- `matplotlib` or `plotly`: Optional for data visualization. Install using `pip install matplotlib` or `pip install plotly`.

### **H2O.ai**

**Title**: Analyzing Bitcoin Prices in Real-Time with H2O.ai  
**Difficulty**: 1 (easy)

**Description**  
H2O.ai is an open-source platform that offers a suite of machine learning and data processing tools. Students will gain an understanding of H2O.ai's core functionalities, like its easy-to-use interface for building machine learning models, automated machine learning (AutoML), and deployment capabilities. This project will involve using H2O.ai to ingest and analyze real-time Bitcoin price data, leveraging its time series analysis features to predict future price trends.

**Describe technology**

- **H2O.ai Core Features**:  
  - Open-source, scalable machine learning platform.  
  - Includes support for a variety of machine learning algorithms.  
  - Provides AutoML capabilities to automatically train and tune models.  
  - Offers integration with other tools and libraries like R, Python, and Spark.  
- **Time Series Analysis**:  
  - H2O.ai supports time series tasks including forecasting, anomaly detection, and visualization.  
  - Use H2O's Deep Learning or AutoML ability to build time series models with minimal coding.

**Describe the project**

- **Objective**: Build a time series forecasting model using H2O.ai to predict future Bitcoin prices based on real-time data.  
- **Steps**:  
  1. **Data Ingestion**: Use Python packages like `requests` or `websockets` to fetch real-time Bitcoin price data from a public API such as CoinGecko or Binance.  
  2. **Data Preparation**: Utilize H2O.ai's data frame support to clean and prepare the fetched data for analysis.  
  3. **Modeling**: Employ H2O.ai AutoML to create and tune time series models to forecast Bitcoin prices.  
  4. **Visualization**: Use H2O.ai's visualization utilities to plot actual vs. predicted Bitcoin prices, showcasing the model’s accuracy.  
  5. **Reporting**: Document the model’s performance metrics and any insights derived from the analysis.

**Useful resources**

- [H2O.ai Documentation](https://docs.h2o.ai/)  
- [H2O.ai GitHub](https://github.com/h2oai)  
- [CoinGecko API Documentation](https://www.coingecko.com/en/api)

**Is it free?** Yes, H2O.ai offers a free and open-source platform, but there is also an enterprise version available with additional features.

**Python libraries / bindings**

- **h2o**: The main Python library used to interact with H2O.ai for data ingestion, model building, and data visualization. Install via `pip install h2o`.  
- **requests** or **websockets**: These libraries can be used to fetch real-time data from external sources.  
- **pandas**: Can be used alongside H2O's data frames for basic data manipulation before transferring to H2O.ai's infrastructure. Install via `pip install pandas`.

By completing this project, students will develop a foundational understanding of utilizing H2O.ai in analyzing time series data, while gaining practical experience in working with real-time data ingestion and modeling in Python.

### **Haystack**

**Title**: Real-Time Bitcoin News Analysis & Q\&A System with Haystack  
**Difficulty**: 2 (medium)  
**Description**  
**Describe technology**  
**Haystack** is an open-source NLP framework by deepset for building end-to-end question answering, retrieval, and semantic search systems. Key features:

- **Document Stores**: Integrate databases (Elasticsearch, FAISS) for vector/text storage.  
- **Retrievers/Pipelines**: Fetch relevant documents using BM25/neural models.  
- **QA Models**: Leverage transformers (e.g., BERT, RoBERTa) for context-aware answers.  
  Example: Analyze Bitcoin news articles to answer questions like *"What caused the price drop on May 12?"*

**Describe the project**  
Build a Haystack pipeline to ingest real-time Bitcoin news/articles, analyze sentiment, and answer time-sensitive questions. Steps:

1. **Data Ingestion**:  
   - Scrape Bitcoin news headlines/articles (e.g., CryptoPanic API) or tweets (Twitter API v2) in real-time.  
   - Store in Elasticsearch document store with metadata (timestamp, source).  
2. **Preprocessing**:  
   - Clean text (remove URLs, special characters).  
   - Use Haystack’s `PreProcessor` to split documents into paragraphs.  
3. **Pipeline Setup**:  
   - **Retriever**: Use `BM25Retriever` to find relevant articles for a query.  
   - **Reader**: Fine-tune a RoBERTa model on financial QA data (e.g., FiQA dataset) for precise answers.  
   - **Generator**: Add a `Seq2SeqGenerator` (e.g., T5) for open-ended questions like *"Summarize Bitcoin’s price drivers this week."*  
4. **Sentiment Analysis**:  
   - Integrate a custom Haystack node using `transformers` pipeline to score article sentiment (positive/negative).  
5. **Interface**:  
   - Build a CLI/Streamlit app where users ask questions and get answers with source citations.

**Useful resources**

- Haystack Documentation: [https://haystack.deepset.ai/](https://haystack.deepset.ai/)  
- CryptoPanic News API: [https://cryptopanic.com/developers/api/](https://cryptopanic.com/developers/api/)  
- FiQA Dataset for Financial QA: [https://sites.google.com/view/fiqa/](https://sites.google.com/view/fiqa/)

**Is it free?**  
Yes. Haystack is Apache-2.0 licensed. CryptoPanic API offers 100 free calls/day.

**Python libraries / bindings**

- haystack-core: Core framework for pipelines.  
- elasticsearch: Document storage/retrieval.  
- transformers: QA/sentiment models (e.g., roberta-base, t5-small).  
- requests: Fetch news data.

### **\+**

### **Hex.tech**

**Title**: Collaborative Bitcoin Market Analysis & Forecasting with Hex  
**Difficulty**: 2 (medium)

**Description**  
**Describe technology**  
**Hex** is a modern data workspace for analytics and collaborative data science. Key features:

- **SQL/Python/R integration**: Mix code languages in notebooks.  
- **Data app publishing**: Turn analyses into interactive dashboards.  
- **Data lineage & versioning**: Track changes and dependencies.  
- **Scheduled pipelines**: Automate data refreshes (e.g., hourly Bitcoin prices).

**Describe the project**  
Build a Hex project to analyze Bitcoin market trends, correlate them with external factors (e.g., S\&P 500, gold), and create a price forecast model. Steps:

1. **Data Ingestion**:  
   - Connect to APIs using Hex’s Python cells:  
     - Bitcoin prices (CoinGecko API).  
     - Macroeconomic data (Alpha Vantage API).  
     - Social sentiment (CryptoPanic headlines).  
   - Schedule hourly data refreshes in Hex.  
2. **Time Series Analysis**:  
   - Use SQL in Hex to calculate:  
     - 30-day volatility.  
     - Bitcoin vs. gold correlation (rolling window).  
   - Python-powered anomaly detection (e.g., sudden 5% drops).  
3. **Machine Learning**:  
   - Train a Prophet time-series model on historical data.  
   - Publish predictions as a Hex data app with sliders to adjust forecast horizons.  
4. **Collaboration**:  
   - Add commentary cells to explain market events (e.g., "ETF approval impact").  
   - Share the app with peers for live feedback.

**Useful resources**

- Hex Docs: [https://learn.hex.tech/](https://learn.hex.tech/)  
- CoinGecko API Guide: [https://www.coingecko.com/en/api](https://www.coingecko.com/en/api)  
- Prophet Forecasting: [https://facebook.github.io/prophet/docs/quick\_start.html](https://facebook.github.io/prophet/docs/quick_start.html)

**Is it free?**   
Hex offers a free tier (limited compute hours). APIs may have usage limits.

**Python libraries / bindings**

- pandas: Data manipulation.  
- prophet: Time-series forecasting.  
- requests: API calls.  
- plotly: Interactive dashboards (built into Hex).

### **Huey**

**Title**: Distributed Bitcoin Trade Processing & Anomaly Detection with Huey  
**Difficulty**: 3 (hard)

**Description**  
**Describe technology**  
**Huey** is a lightweight, multi-threaded task queue for Python designed for simplicity and scalability. Key features:

- **Task scheduling**: Execute asynchronous, periodic, or retryable tasks.  
- **Redis/SQLite backend**: Supports distributed task queues with Redis for high throughput.  
- **Task prioritization**: Handle critical tasks (e.g., anomaly alerts) first.  
- **Consumer pools**: Parallelize workers across CPU cores or machines.

**Describe the project**  
Build a fault-tolerant, distributed system to ingest and process high-frequency Bitcoin trade data (10,000+ trades/hour) using Huey. Steps:

1. **Real-Time Ingestion**:  
   - Connect to WebSocket APIs (Binance, Coinbase Pro) to stream trade data.  
   - Use Huey tasks to enqueue each trade with prioritization (e.g., large trades \> small trades).  
2. **Distributed Processing**:  
   - Deploy Huey consumer workers across multiple machines (or Docker containers).  
   - Tasks include:  
     - **Aggregation**: Calculate 1-min/5-min OHLC (Open-High-Low-Close) metrics.  
     - **Anomaly detection**: Flag trades 3σ outside rolling averages (Huey retries failed analysis).  
     - **Sentiment sync**: Correlate trades with Twitter API sentiment data in parallel.  
3. **Fault Tolerance**:  
   - Implement task retries with exponential backoff for API rate limits/errors.  
   - Use Redis as a Huey backend for durability (tasks survive worker crashes).  
4. **Monitoring & Alerts**:  
   - Expose Prometheus metrics for task throughput/latency.  
   - Trigger Slack alerts via Huey hooks on critical anomalies.

**Useful resources**

* Huey Documentation: [https://huey.readthedocs.io/en/latest/](https://huey.readthedocs.io/en/latest/)  
* Binance WebSocket API: [https://binance-docs.github.io/apidocs/spot/en/\#websocket-market-streams](https://binance-docs.github.io/apidocs/spot/en/#websocket-market-streams)  
* Prometheus Python Client: [https://github.com/prometheus/client\_python](https://github.com/prometheus/client_python)

**Is it free?**   
Yes. Huey is MIT-licensed. Redis has a free tier; cloud costs apply for scaling.

**Python libraries / bindings**

* huey: Core task queue library.  
* websockets: Real-time trade ingestion.  
* redis: Distributed task backend (pip install redis).  
* prometheus\_client: Monitoring.  
* pandas: Time-series aggregation (optional).

### **Huggingface**

**Huggingface**: Real-time Bitcoin News Summarization and Trend Prediction with HuggingFace

**Difficulty**

- **Level:** 3 (difficult)

**Description**

- **Technology Overview:**  
  HuggingFace provides the `transformers` library with pre-trained models (e.g., BERT, GPT) for NLP tasks like summarization and sentiment analysis. This project uses it to process Bitcoin news and predict market trends.  
    
- **Project Details:**  
  This project builds a system to:  
    
  - **Ingest Data:** Collect real-time Bitcoin news via NewsAPI and web scraping (e.g., BeautifulSoup).  
  - **Process with HuggingFace:** Use `transformers` to summarize articles and analyze sentiment.  
  - **Time Series Analysis:** Aggregate sentiment and topics into time series data.  
  - **Predictive Modeling:** Train a model (e.g., RNN) to predict Bitcoin prices from news data.  
  - **Visualization:** Create a real-time dashboard for summaries, sentiment, and predictions.  
  - **Performance:** Optimize with GPU acceleration for model inference.


  The complexity arises from handling large text datasets, advanced NLP, and real-time prediction.

**Useful Resources**

- [HuggingFace Transformers](https://huggingface.co/docs/transformers/index)  
- [NewsAPI](https://newsapi.org/docs)  
- [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)  
- [CoinGecko API](https://www.coingecko.com/en/api)  
- [TensorFlow](https://www.tensorflow.org/)  
- [Streamlit](https://docs.streamlit.io/)

**Is it Free?**

- **HuggingFace:** Yes, pre-trained models are free.  
- **NewsAPI & CoinGecko:** Free tiers available.  
- **Web Scraping:** Free, subject to terms.

**Python Libraries**

- `transformers`: `pip install transformers`  
- `requests`: `pip install requests` (API calls)  
- `beautifulsoup4`: `pip install beautifulsoup4` (scraping)  
- `tensorflow` or `pytorch`: `pip install tensorflow` or `pip install torch` (modeling)  
- `pandas`: `pip install pandas` (data handling)  
- `streamlit`: `pip install streamlit` (dashboard)

### **HuggingFace \#2**

**Title**: Bitcoin Event-Driven Price Impact Analysis with Hugging Face NLP  
**Difficulty**: 2 (medium)  
**Description**  
**Describe technology**  
Hugging Face’s `transformers` library provides models for **Named Entity Recognition (NER)** and **text classification**. This project uses a pre-trained NER model (`dslim/bert-base-NER`) to detect Bitcoin-related events (e.g., regulatory changes, hacks) from news articles and measure their delayed price impact.

**Describe the project**  
This project identifies actionable Bitcoin market events from text and quantifies their multi-day price effects, avoiding overlap with sentiment-based approaches:

- **Data collection**:  
  - Scrape **long-form Bitcoin news articles** (not headlines) using `newspaper3k` (Python library).  
  - Fetch daily OHLC (Open-High-Low-Close) Bitcoin data from CoinGecko.  
- **Event extraction**:  
  - Use Hugging Face’s NER pipeline to detect entities like organizations ("SEC"), laws ("MiCA"), and technologies ("Lightning Network").  
  - Classify articles into event types using zero-shot classification (`facebook/bart-large-mnli`):  
    1. Regulatory, Technological, Market Manipulation, Adoption News.  
- **Time series engineering**:  
  - Create a binary event matrix (1=event occurred on day *t*, 0=otherwise) for each category.  
  - Calculate 3-day rolling price volatility (% change from day *t* to *t+3*).  
- **Causal inference**:  
  - Use **propensity score matching** (PSM) with `causalnex` to isolate event impacts from market noise.  
  - Quantify average price volatility increase/decrease per event type.  
- **Reporting**:  
  - Build an automated report showing "High-Impact Events" (e.g., "SEC lawsuits cause \+8% volatility").  
  - Visualize event clusters on a timeline with price overlays using `plotly`.


**Challenges**:

- Distinguishing impactful events from routine news (e.g., "Coinbase listing" vs. "Coinbase routine maintenance")  
- Handling overlapping events in time series analysis  
- Addressing survivorship bias in news scraping


**Useful resources**

- [Hugging Face Zero-Shot Classification Guide](https://huggingface.co/docs/transformers/tasks/zero_shot_classification)  
- [CausalNEX Documentation](https://causalnex.readthedocs.io/)  
- [CoinGecko OHLC API](https://www.coingecko.com/en/api/documentation)


**Is it free?**  
Yes:

- `newspaper3k` and Hugging Face models are open-source  
- CoinGecko API free tier supports daily data  
- `causalnex` is MIT-licensed


**Python libraries / bindings**

- `transformers`: NER and zero-shot classification  
- `newspaper3k`: News article scraping & NLP  
- `causalnex`: Propensity score matching  
- `pandas`/`numpy`: Time series alignment  
- `plotly`: Interactive timeline visualization  
- `requests`: API data fetching

### **Kedro**

**Title:** Real-Time Bitcoin Price Analysis with Kedro

**Difficulty:** 1=easy

**Description**

Kedro is an open-source Python framework for creating reproducible, maintainable, and modular data science code. It is particularly useful for designing data pipelines by applying software engineering best practices. Understand the core concepts of Kedro like nodes, pipelines, and data catalog, which provide a structured approach to managing data, workflow, and experiment tracking.

**Describe technology**

- **Kedro**: This framework is designed to help data scientists and engineers create robust modular pipelines by enforcing a standard way to work with data. It promotes version control, environment management, and testing.  
- **Nodes and Pipelines**: Nodes are the building blocks in Kedro and each node represents a function. Pipelines are made up of nodes and define the order in which nodes should be executed.  
- **Data Catalog**: A feature of Kedro that catalogs all the datasets used and produced by the pipeline, which helps in data tracing and management.

**Describe the project**

- **Objective**: Use Kedro to build a simple pipeline that ingests real-time Bitcoin price data and performs a basic time series analysis, like moving averages.  
- **Data Source**: Pull real-time Bitcoin price data from a public API like CoinGecko.  
- **Steps**:  
  1. Set up a Kedro project and configure the environment.  
  2. Create a data catalog for managing the Bitcoin price data.  
  3. Implement the data ingestion node to fetch and store the Bitcoin price data.  
  4. Build a pipeline that reads the data, applies a moving average calculation as a simple time series analysis, and stores the results.  
  5. Set up version control for your Kedro project and ensure code quality with unit tests.  
- **Outcome**: By the end of the project, students should have a basic understanding of how to use Kedro to design data pipelines, while getting hands-on experience with time series analysis using Python.

**Useful resources**

- [Kedro Documentation](https://kedro.readthedocs.io)  
- [Kedro GitHub Repository](https://github.com/kedro-org/kedro)  
- [Python for Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/)

**Is it free?**

Yes, Kedro is an open-source project and free to use.

**Python libraries / bindings**

- **Kedro**: To create and run your data pipelines. Install it via `pip install kedro`.  
- **pandas**: For data manipulation and analysis, including computing moving averages. Install with `pip install pandas`.  
- **requests**: For fetching data from external APIs. Install with `pip install requests`.  
- **Pytest**: For unit testing your code within Kedro. Install with `pip install pytest`.

By working on this project, students will gain a foundational understanding of Kedro’s capabilities and how it can be used to streamline the data science workflow for real-time data analysis projects.

### **Keras**

**Title**: Analyzing Bitcoin Prices with Keras and Time Series

**Difficulty**: 1 (easy)

**Description**  
This project involves building a simple time series model using Keras to analyze and predict Bitcoin price movements. Keras is a high-level neural networks library written in Python that simplifies the process of building deep learning models. Students will learn the basic functionalities of Keras by constructing a simple neural network and applying it to predict Bitcoin prices using historical data.

**Describe technology**

- **Keras**: An open-source high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano.  
- Designed for easy and fast prototyping.  
- Provides simple, consistent interface optimized for user-friendliness.  
- Supports both convolutional networks and recurrent networks, as well as combinations of the two.  
- Features include customizability, modularity, and ease of extensibility.

**Describe the project**

- **Data Acquisition**: Use Python libraries such as `pandas` and `requests` to fetch historical Bitcoin price data from a public API like CoinGecko.  
- **Data Preprocessing**: Clean the dataset and prepare it for time series analysis. This can include handling missing values, normalization, and splitting the data into training and testing sets.  
- **Model Building with Keras**:  
  - Develop a simple sequential model using Keras.  
  - Implement layers such as LSTM (Long Short-Term Memory) for handling time series data.  
  - Compile the model with an appropriate loss function, optimizer, and metrics.  
- **Training and Evaluation**:  
  - Train the model on the prepared dataset and evaluate its performance.  
  - Visualize results using plots to show actual vs. predicted prices.  
- **Deployment**:  
  - Use the trained model to make future forecasts on Bitcoin prices.  
  - Discuss potential improvements and next steps for more complex models.

**Useful resources**

- [Keras Documentation](https://keras.io/)  
- [TensorFlow Documentation](https://www.tensorflow.org/tutorials)  
- [CoinGecko API Documentation](https://www.coingecko.com/en/api)

**Is it free?**  
Yes, Keras and its dependencies can be used for free. Libraries to fetch data (like `pandas` and `requests`) are also open-source.

**Python libraries / bindings**

- **Keras**: The primary library used for building the neural network models.  
- **TensorFlow**: The backend engine running Keras, facilitating model training and predictions.  
- **pandas**: For data manipulation and preprocessing.  
- **numpy**: For number handling and scientific computing.  
- **matplotlib**: For visualization of data and prediction results.  
- **requests**: For retrieving real-time and historical Bitcoin price data from APIs.

This project allows students to gain practical experience in using Keras for time series analysis and demonstrates how machine learning can be applied in cryptocurrency markets for predictive analytics.

### **Kubeflow**

Title: Real-time Bitcoin Price Analysis Using Kubeflow

**Difficulty**: 2 (medium difficulty)

**Description**:  
This project revolves around using Kubeflow, an open-source Kubernetes-native platform designed to expedite the deployment, orchestration, and scaling of machine learning workflows. Kubeflow is ideal for managing ML pipelines on Kubernetes, offering a variety of components such as Jupyter notebooks, TensorFlow training, and KFServing for model serving. The aim is to ingest and process real-time Bitcoin price data for time series analysis, utilizing Python for data manipulation and analysis tasks. The project will expose students to the basics of Kubeflow and its capabilities in handling big data and machine learning workflows within the Kubernetes ecosystem.

**Describe technology**:

- **Kubeflow Overview**: Understand the core functionalities of Kubeflow, including components like Pipelines for orchestrating complex workflows and KFServing for deploying machine learning models.  
- **Kubernetes Integration**: Leverage Kubernetes for resource scheduling and management, enabling efficient scaling and deployment in containerized environments.  
- **Workflow Automation**: Use Kubeflow Pipelines to automate the end-to-end workflow of ingesting, processing, analyzing, and serving Bitcoin price data.

**Describe the project**:

- **Data Ingestion**: Fetch real-time Bitcoin prices using a public API such as CoinGecko. Implement a continuous data ingestion pipeline using Kubeflow Pipelines to retrieve data at regular intervals.  
- **Data Storage and Processing**: Store the fetched data in a time-series database, such as TimescaleDB, to handle frequent updates and support efficient retrieval for analysis.  
- **Time Series Analysis**: Use Python libraries like Pandas and NumPy to perform exploratory data analysis (EDA) and basic time series forecasting on the historical Bitcoin price data.  
- **Model Deployment and Serving**: Train a simple predictive model using a machine learning library of your choice, and deploy it using KFServing to enable real-time predictions.  
- **Visualization and Reporting**: Use Python libraries such as Matplotlib or Seaborn to visualize the Bitcoin price trends and forecast results, generating insights into price movements.

**Useful resources**:

- [Kubeflow Official Documentation](https://www.kubeflow.org/docs/)  
- [Kubeflow GitHub Repository](https://github.com/kubeflow/kubeflow)  
- [Kubeflow Slack Community](https://kubeflow.slack.com/)  
- [TimescaleDB Documentation](https://www.timescale.com/docs)

**Is it free?**:  
Kubeflow itself is free to use as an open-source project. However, deploying Kubeflow requires a Kubernetes cluster, which may incur costs depending on the cloud provider used (e.g., Google Cloud Platform, Amazon Web Services). Local setup with Minikube or Docker for small-scale testing is free.

**Python libraries / bindings**:

- **Kubernetes Python Client**: For interacting with Kubernetes APIs.  
- **Kubeflow Pipelines SDK**: To create and manage pipelines in Kubeflow.  
- **Pandas**: For data manipulation and analysis.  
- **NumPy**: For numerical computations.  
- **Scikit-learn**: For building basic machine learning models.  
- **Matplotlib / Seaborn**: For data visualization.

### **Kubernetes**

**Title**: Implementing Real-time Bitcoin Data Analysis with Kubernetes

**Difficulty**: 2=medium difficulty (it should take around 10 days to complete)

**Description**: This project involves leveraging Kubernetes to create a scalable and efficient infrastructure for ingesting and processing real-time Bitcoin data. Kubernetes is an open-source platform designed to automate deploying, scaling, and operating application containers. This project will focus on setting up a Kubernetes cluster to handle Bitcoin data ingestion and processing using Python and basic data processing libraries. Students will learn to deploy a containerized application that fetches real-time data from a Bitcoin API, processes the data, and performs time-series analysis.

**Describe technology**:

- **Kubernetes**: An orchestrated container management system initially designed by Google, now run as an open-source project. It allows developers to deploy application containers across a cluster of machines with processes for automation, scaling, and management.  
- **Core Concepts**: Pods (the smallest deployable units), ReplicaSets (ensure specified number of pod replicas), Services (abstract away the pod details), and Deployments (manage deployments).

**Describe the project**:

- **Objective**: Set up a Kubernetes cluster that can scale the ingestion and processing of real-time Bitcoin data, perform essential data transformations, and apply basic time-series analysis.  
- **Steps**:  
  1. **Environment Setup**: Create a Kubernetes cluster using a cloud provider or using Minikube locally.  
  2. **Containerization**: Create a Docker container for a Python application that fetches real-time Bitcoin prices using a public API like CoinGecko.  
  3. **Deployment on Kubernetes**: Deploy the containerized application on the Kubernetes cluster using Kubernetes Deployment and manage the application lifecycle.  
  4. **Real-time Processing**: Use Python libraries like Pandas for basic data manipulations and Matplotlib for visualizations to show trends and patterns over time.  
  5. **Scaling**: Set up Kubernetes Autoscaling to handle increased load during peak times.  
  6. **Monitoring**: Implement monitoring for the deployed application using Kubernetes-native solutions like Prometheus and Grafana.

**Useful resources**:

- [Kubernetes Official Documentation](https://kubernetes.io/docs/home/)  
- [Minikube Official Site](https://minikube.sigs.k8s.io/docs/)  
- [Docker Official Documentation](https://docs.docker.com/)  
- [Bitcoin API from CoinGecko](https://www.coingecko.com/en/api)  
- [Prometheus for Kubernetes](https://prometheus.io/docs/prometheus/latest/getting_started/)  
- [Grafana for data visualization](https://grafana.com/docs/grafana/latest/introduction/)

**Is it free?**:

- Kubernetes itself is free and open-source. However, deploying on the cloud may incur costs based on resources used (e.g., GKE, EKS).  
- Minikube provides a free local setup for learning and experimenting at no cost.  
- Docker offers free tier usage but may have limits based on organizational policies for usage at scale.

**Python libraries / bindings**:

- **Pandas**: For data manipulation and analysis.  
- **Matplotlib**: For plotting and visualizations of time-series data.  
- **Requests**: For HTTP requests to the Bitcoin API.  
- **Docker SDK for Python**: For building and managing Docker images.  
- **Kubernetes Python client**: To interact programmatically with Kubernetes clusters. Installable via pip (`pip install kubernetes`).

This project will give students a practical understanding of deploying applications in a cloud-native environment while gaining hands-on experience with time-series data processing.

### **Kubernetes Python Client**

**Title**: Analyzing Bitcoin Data with Kubernetes Python Client

**Difficulty**: 1 (easy)

**Description**  
In this project, you will use the Kubernetes Python Client to manage and deploy a simple application that ingests real-time Bitcoin price data. Kubernetes is an open-source container orchestration platform that automates many manual processes involved in deploying, managing, and scaling containerized applications. This project will guide you through setting up a small Kubernetes cluster using Python to deploy a simple Bitcoin price tracker that processes time-series data from a public API and outputs basic analytics.

**Describe technology**

- **Kubernetes**: A platform designed to automate deploying, scaling, and operating application containers.  
- **Kubernetes Python Client**: A Python library that interacts with Kubernetes clusters. This client allows developers to manage Kubernetes resources easily, execute commands within containers, and handle cluster-related operations programmatically.

**Describe the project**

- You'll start by setting up a local Kubernetes cluster using Minikube or a similar tool.  
- Using the Kubernetes Python Client, you'll write scripts to deploy a Flask application in Kubernetes. This application will fetch real-time Bitcoin price data from a public API like CoinGecko.  
- You will set up a Kubernetes CronJob that schedules regular data fetching and storage tasks.  
- Implement basic time-series analysis with Python to calculate statistics such as average price over time or percentage change.  
- Extend the deployment with Kubernetes resources like ConfigMaps for configuration management and Persistent Volumes for data storage.  
- Finally, visualize the processed data by deploying a simple frontend service within the Kubernetes cluster, showcasing how Bitcoin prices change over time.

**Useful resources**

- [Kubernetes Official Documentation](https://kubernetes.io/docs/home/)  
- [Kubernetes Python Client Documentation](https://github.com/kubernetes-client/python)  
- [Minikube Quickstart](https://minikube.sigs.k8s.io/docs/start/)

**Is it free?**  
Yes, this project is free, as you can run Kubernetes locally using Minikube or a similar tool without needing a cloud service.

**Python libraries / bindings**

- **Kubernetes Python Client**: Install using `pip install kubernetes`. It allows interaction with Kubernetes clusters and resources.  
- **Requests**: Install using `pip install requests`. Used for making HTTP requests to fetch Bitcoin data from APIs.  
- **Flask**: Install using `pip install flask`. A lightweight WSGI web application framework to create the application for fetching and serving data.  
- **Pandas**: Install using `pip install pandas`. Useful for time-series analysis and handling data structures.

### **Langchain and Neo4j**

**Title**: Real-time Bitcoin Analysis with Langchain and Neo4j   
**Difficulty**: 3 (difficult)

**Description**  
Langchain is a framework for building applications powered by language models. It simplifies the integration of advanced natural language processing (NLP) capabilities into real-time applications. Neo4j is a graph database platform that is especially adept at handling highly interconnected data. This project integrates both Langchain and Neo4j to build a sophisticated system for ingesting and processing real-time Bitcoin data. The project aims to capture and analyze trends, patterns, and correlations in bitcoin transactions using complex queries and NLP-based time series analysis.

**Describe technology**

- **Langchain**:  
    
  - A framework that facilitates the development of applications using large language models.  
  - Provides tools to easily access and manipulate language models for various operations such as summarization, text generation, question answering, etc.  
  - An example would be using Langchain to automatically generate insights or summaries from raw bitcoin transaction data.


- **Neo4j**:  
    
  - A native graph database designed to leverage data relationships as first-class entities.  
  - It allows the representation of intricate networks and supports graph algorithms that traverse these networks efficiently.  
  - Examples include using Neo4j to store transaction data and perform complex network analyses to discover transaction clusters or anomalies.

**Describe the project**

- **Objective**:  
  Implement a system that ingests real-time bitcoin transaction data and stores it in a Neo4j graph database. Use Langchain to perform NLP-based time series analysis, generating insights from the evolving data set.  
    
- **Steps**:  
    
  1. **Data Ingestion**:  
     - Set up a process to fetch real-time Bitcoin transaction data from a public API like CoinGecko using Python.  
     - Insert this data into a Neo4j database in the form of nodes and relationships representing transactions and wallets.  
  2. **Graph Data Modeling**:  
     - Create a schema in Neo4j to optimize storage of time-series bitcoin transaction data.  
     - Design relationships that allow for complex queries like clustering and trend analysis.  
  3. **Real-Time Processing**:  
     - Use Py2neo or Neo4j Python driver to query the database for real-time insights.  
     - Implement periodic data analysis scripts that perform network analysis to identify influential nodes or sudden changes in transaction patterns.  
  4. **NLP Analysis**:  
     - Use Langchain to convert raw transaction data into meaningful narratives or summaries.  
     - Implement a Langchain-based system to analyze transaction patterns using time series techniques and predict future trends or identify anomalies.


- **Outcome**: Students will produce a system demonstrating sophisticated data ingestion, storage, and analysis capabilities combining NLP with graph data processing.

**Useful resources**

- [Langchain Official Documentation](https://docs.langchain.com/)  
- [Neo4j Graph Database and Analytics](https://neo4j.com/docs/)  
- [CoinGecko API Documentation](https://www.coingecko.com/en/api)

**Is it free?**

- **Langchain**: Usage might involve a licensing cost depending on the model used.  
- **Neo4j**: Offers a free version, but larger projects might need a commercial license.  
- **APIs**: CoinGecko API offers a free tier with rate limits.

**Python libraries / bindings**

- **Langchain Python SDK**: Ensure smooth integration of language model capabilities. Implemented via Python's Natural Language Toolkit (NLTK) or similar.  
- **Neo4j Python Driver (Py2neo)**: Used for graph database manipulation and querying. Installable via pip.  
- **Requests**: For handling HTTP requests to APIs.

### **LightGBM**

**Title**: Real-Time Bitcoin Price Analysis using LightGBM

**Difficulty**: 2 (medium)

**Description**  
LightGBM is a highly efficient and scalable gradient boosting framework developed by Microsoft, optimized for speed and accuracy. It is especially effective for training models on large datasets because it uses a histogram-based approach that reduces memory usage and improves training speed. LightGBM's key features include support for categorical features natively, improved accuracy, and faster training. In this project, students will leverage LightGBM to create a real-time prediction system for Bitcoin prices. Using basic Python packages, students will construct a pipeline for ingesting live Bitcoin price data, perform necessary processing, and implement a time-series prediction model to forecast future price movements.

**Describe Technology**

- LightGBM is a gradient boosting framework that uses tree-based learning algorithms.  
- It is designed for distributed systems and excels in handling large datasets.  
- Supports efficient parallel and GPU learning, which accelerates the training process.  
- Natively supports categorical features, which can lead to better model performance without preprocessing.  
- LightGBM is known for its accuracy while maintaining computational efficiency.

**Describe the Project**

- **Data Ingestion**: Set up an environment to ingest real-time Bitcoin price data from a public API such as CoinGecko. Use Python libraries like requests or websocket-client to continuously fetch data every few minutes.  
- **Data Processing**: Preprocess the fetched Bitcoin data to extract relevant features for time-series analysis. This may include engineering time-based features (e.g., rolling averages, time lags).  
- **Model Implementation**: Implement a LightGBM model to perform time-series forecasting. The goal is to predict the next-minute Bitcoin price based on historical data.  
- **Model Evaluation**: Split data into training and test sets. Use evaluation metrics suitable for time-series data, such as Root Mean Square Error (RMSE) or Mean Absolute Error (MAE), to evaluate the model's performance.  
- **Real-Time Prediction**: Integrate the trained model into the data ingestion pipeline to make real-time price predictions and log the results for further analysis.  
- **Visualization**: Utilize a Python data visualization library like matplotlib or seaborn to plot actual vs. predicted Bitcoin prices over time to visually assess the model performance.

**Useful Resources**

- [LightGBM Documentation](https://lightgbm.readthedocs.io/)  
- [Python requests library documentation](https://pypi.org/project/requests/)  
- [WebSocket client for Python](https://pypi.org/project/websocket-client/)  
- [CoinGecko API](https://www.coingecko.com/en/api)  
- [Matplotlib documentation](https://matplotlib.org/stable/contents.html)

**Is it free?**  
Yes, LightGBM and the other Python libraries used (requests, websocket-client, matplotlib) are   
open-source and free to use.

**Python Libraries / Bindings**

- **LightGBM**: The primary library for implementing the gradient boosting model. Install via `pip install lightgbm`.  
- **Requests**: Used for making HTTP requests to retrieve Bitcoin price data from an API. Install via `pip install requests`.  
- **Websocket-client**: For establishing real-time connections to fetch Bitcoin data without polling. Install via `pip install websocket-client`.  
- **Matplotlib/Seaborn**: For visualizing actual vs. predicted Bitcoin prices. Install via `pip install matplotlib seaborn`.

### **LlamaIndex**

**Title**: Enterprise-Scale Bitcoin Data Knowledge Graph with LlamaIndex  
**Difficulty**: 3 (hard)

**Description**  
**Describe technology**  
**LlamaIndex** is a framework for building LLM-powered data applications. It specializes in indexing/retrieving structured and unstructured data for RAG (Retrieval-Augmented Generation). Key features:

- **Data connectors**: APIs, SQL DBs, PDFs, blockchain nodes.  
- **Hierarchical indices**: Optimize LLM context windows via summaries.  
- **Query engines**: Multi-step reasoning over hybrid data sources.  
- **Agents**: Autonomous LLM-driven analysis workflows.

**Describe the project**  
Create a Bitcoin analytics platform using LlamaIndex to ingest, index, and query petabyte-scale blockchain/economic data with LLMs. Challenges:

1. **Multi-Source Ingestion**:  
   - Stream real-time data: Bitcoin node (raw blocks), Glassnode API (on-chain metrics), FRED (macroeconomic indicators).  
   - Build custom LlamaIndex data loaders for blockchain RPC endpoints.  
2. **Knowledge Graph Construction**:  
   - Use LlamaIndex’s `KnowledgeGraphIndex` to link entities (wallets, transactions, macroeconomic events).  
   - Enable queries like *"Show transactions linked to Mt. Gox wallets during 2023-2024 Fed rate hikes"*.  
3. **LLM Agent System**:  
   - Deploy LlamaIndex agents with tools for:  
     - **On-chain forensics**: Trace stolen funds via taint analysis.  
     - **Sentiment synthesis**: Correlate Reddit/Twitter chatter with price action.  
     - **Risk simulation**: *"What if the SEC rejects spot ETFs? Model price impact."*  
4. **Optimization**:  
   - Implement hierarchical indices to handle 10M+ transactions.  
   - Fine-tune open-source LLMs (e.g., Llama-3) on Bitcoin whitepaper/transaction semantics.  
5. **Deployment**:  
   - Serve via FastAPI with auth/rate limiting.  
   - Monitor with Prometheus/Grafana (token/sec, cache hit rates).

**Useful resources**

* LlamaIndex Documentation: [https://docs.llamaindex.ai/](https://docs.llamaindex.ai/)  
* Bitcoin Core RPC API: [https://developer.bitcoin.org/reference/rpc/](https://developer.bitcoin.org/reference/rpc/)  
* FRED Economic Data: [https://fred.stlouisfed.org/docs/api/fred/](https://fred.stlouisfed.org/docs/api/fred/)

**Is it free?**  
LlamaIndex is MIT-licensed. Costs accrue from LLM APIs (OpenAI/Anthropic) and cloud infra.

**Python libraries / bindings**

* llama-index-core: Core indexing/query logic.  
* llama-index-llms-openai: GPT-4/Claude integrations.  
* bitcoinrpc: Bitcoin node interaction.  
* docker: Containerized microservices.

### **LLM**

**Title**: Real-time Bitcoin Sentiment Analysis and Price Prediction with llm

**Difficulty**

- **Level:** 3 (difficult)

**Description**

- **Technology Overview:**  
  The `llm` library is a simple and minimal Python package for working with Large Language Models (LLMs). It allows integration with various LLM providers and models, facilitating tasks such as text generation, sentiment analysis, and topic modeling. In this project, `llm` will be used to process real-time Bitcoin-related text data for sentiment analysis and feature extraction.  
    
- **Project Details:**  
  This project involves building a comprehensive system to:  
    
  - Ingest real-time Bitcoin-related data from sources like Twitter, Reddit, and news APIs.  
  - Use the `llm` library to connect to an LLM (e.g., GPT-3 or a fine-tuned model) for sentiment analysis and topic modeling of the text data.  
  - Aggregate sentiment scores and topic frequencies into time series data.  
  - Develop a predictive model (e.g., LSTM or Prophet) to forecast Bitcoin prices based on the extracted features.  
  - Create a real-time dashboard using Dash or Streamlit to visualize sentiment trends, topic evolution, and price predictions.  
  - Optimize the system for scalability using cloud services like AWS Lambda or Google Cloud Functions.


  The complexity arises from handling real-time data streams, integrating with LLMs, performing time series analysis, and ensuring scalability with high data volumes.

**Useful Resources**

- [llm Python Package](https://pypi.org/project/llm/)  
- [Twitter Streaming API](https://developer.twitter.com/en/docs/twitter-api/tweets/filtered-stream/introduction)  
- [PRAW Documentation](https://praw.readthedocs.io/en/stable/)  
- [NewsAPI](https://newsapi.org/docs)  
- [CoinGecko API](https://www.coingecko.com/en/api)  
- [Dash Documentation](https://dash.plotly.com/)  
- [Streamlit Documentation](https://docs.streamlit.io/)

**Is it Free?**

- **llm:** Yes, open-source.  
- **APIs:** Free tiers available for Twitter, Reddit, NewsAPI, and CoinGecko with limitations.  
- **Cloud Services:** AWS Lambda and Google Cloud Functions have free tiers; costs may apply with heavy usage.

### **llm.datasette**

**Title**: Real-Time Bitcoin Data Processing with Datasette

**Difficulty**: 2 (medium, it should take around 10 days to complete)

**Description**: Datasette is an open-source tool designed for exploring and publishing data, especially useful for working with structured datasets. It's built to make it easy to visualize and query data from SQLite databases. In this project, you will learn how to use Datasette to ingest and process real-time Bitcoin pricing data. The main objectives are to set up an SQLite database to store the incoming data, visualize the data trends, and perform time series analysis. This project allows students to interact with data in a more dynamic and exploratory fashion, leveraging the easy-to-use yet powerful capabilities of Datasette.

**Describe technology**:

- **Datasette**: An open-source tool specifically designed for publishing and exploring data from SQLite databases over the web.  
- Provides a web interface for querying and visualizing data, making it accessible to users without requiring advanced technical skills.  
- Excellent for building publicly accessible databases with strong support for custom plugins and queries.  
- Supports live-updating datasets, making it suitable for real-time data analysis and exploration.

**Describe the project**:

- **Objective**: To develop a real-time data processing system to ingest Bitcoin pricing data and perform basic time series analysis using Datasette.  
- **Data Ingestion**: Use a public API like CoinGecko to fetch real-time Bitcoin price data and store it in an SQLite database.  
- **Database Setup**: Configure Datasette to connect to SQLite, creating a live-updating database that continuously ingests data.  
- **Visualization**: Utilize Datasette's web interface to build interactive dashboards for visualizing trends in Bitcoin prices over time.  
- **Time Series Analysis**: Implement basic time series analysis on the data, such as calculating moving averages and detecting price anomalies.

**Useful resources**:

- [Datasette Documentation](https://docs.datasette.io/)  
- [CoinGecko API Documentation](https://coingecko.com/en/api/documentation)  
- [SQLite Documentation](https://sqlite.org/docs.html)  
- [Time Series Analysis in Python](https://www.analyticsvidhya.com/blog/2021/07/time-series-forecasting-in-python/)

**Is it free?**: Yes, Datasette is open-source and free to use. However, hosting on a cloud service for production may incur costs.

**Python libraries / bindings**:

- **Datasette**: Installable via pip with `pip install datasette`. Provides a platform for exploring and publishing data using SQLite databases.  
- **Requests**: For making HTTP requests to fetch data from the CoinGecko API. `pip install requests` is needed.  
- **SQLite**: Built-in Python library for database interaction. No external installation is required.  
- **Pandas**: Helpful for manipulating and analyzing time series data. Install using `pip install pandas`.  
- **Schedule**: For scheduling regular updates of Bitcoin pricing data. Installable via `pip install schedule`.

### **Luigi**

**Title**: Real-Time Bitcoin Analysis using Apache Flink

**Difficulty**: 3 (Difficult)

**Description**  
Apache Flink is an open-source stream processing framework for processing large volumes of data in real time. It is highly suited for low-latency and high-throughput data streaming applications, allowing for real-time event processing and stateful stream processing. This project involves leveraging Apache Flink to design and implement a real-time data processing pipeline to analyze Bitcoin price data. The aim is to receive live Bitcoin price data, process it, and conduct time series analysis to detect anomalies or trends.

**Describe technology**

- Apache Flink is known for its powerful capabilities in handling both batch and stream processing.  
- Key components include the Flink Runtime, which executes dataflow programs, and the Flink APIs for defining jobs.  
- It supports diverse environments, allowing integration with various data sources such as Kafka for ingesting data.  
- Stateful processing enables complex analytics, including aggregations and windowing operations, essential for time series data.

**Describe the project**

- **Objective**: Set up a robust system to ingest real-time Bitcoin data, process it using Apache Flink, and perform time series analysis.  
- **Data Ingestion**: Utilize a public Bitcoin API, like CoinGecko or Binance, to receive real-time price data. Implement Kafka as a message broker to ingest this streaming data into Flink.  
- **Data Processing**: Create a Flink streaming job that performs real-time analytics. This should include time windowing operations to compute rolling averages or detect significant price changes and anomalies.  
- **Time Series Analysis**: Use Python's capabilities via Flink-Python bindings to perform advanced analytics, such as trend prediction using ARIMA models or anomaly detection with statistical tests.  
- **Output and Visualization**: Store processed data in a time-series database like InfluxDB and visualize findings using a dashboard tool like Grafana.

**Useful resources**

- [Apache Flink Documentation](https://nightlies.apache.org/flink/flink-docs-stable/)  
- [Kafka Connectors for Flink](https://nightlies.apache.org/flink/flink-docs-stable/dev/connectors/kafka/)  
- [Flink-Python API Documentation](https://nightlies.apache.org/flink/flink-docs-release-1.14/docs/dev/python/getting-started/)

**Is it free?**  
Yes, Apache Flink is an open-source project available under the Apache License, Version 2.0. You may need cloud resources or infrastructure for deploying this project, which might incur costs.

**Python libraries / bindings**

- **Flink-Python (PyFlink)**: Provides Python API interfaces for Apache Flink, facilitating the definition and execution of data processing tasks. Install using `pip install apache-flink`.  
- **pandas**: Handy for data manipulation and analysis tasks in Python.  
- **statsmodels**: Useful for performing time series analysis, including ARIMA modeling. Install using `pip install statsmodels`.  
- **kafka-python**: A Python client for the Apache Kafka distributed streaming platform. Install using `pip install kafka-python`.  
- **influxdb**: To interact with InfluxDB for storing and querying time-series data. Install using `pip install influxdb-client`.

### **Luigi Corrected**

**Title**: Real-Time Bitcoin Price Analytics Pipeline with Luigi  
**Difficulty**: 3 (difficult)  
**Description**:  
Design a Luigi-powered pipeline to ingest, process, and analyze real-time Bitcoin price data. The project focuses on building a fault-tolerant system for time series forecasting and anomaly detection.

**Describe technology**:

- **Luigi**: A workflow management system for orchestrating complex data pipelines.  
  - **Key functionalities**:  
    - Task dependency resolution (e.g., `requires()` method for chaining tasks).  
    - Atomic output handling (e.g., `output().exists()` checks).  
    - Parallel execution (e.g., `--workers 4` flag).  
  - Example: A `FetchDataTask` class that triggers `CleanDataTask` only after successful API ingestion.

**Describe the project**:

1. **Real-time ingestion**:  
   - Stream Bitcoin price data from Coinbase Pro WebSocket API (15,000+ requests/day).  
   - Implement error handling for API rate limits and reconnection logic.  
2. **Time series processing**:  
   - Calculate 1-hour rolling volatility and ARIMA-based forecasts.  
   - Detect anomalies using Z-score thresholds (±3σ).  
3. **Pipeline orchestration**:  
   - Create 5+ interdependent Luigi tasks (e.g., Fetch→Clean→Analyze→Visualize→Alert).  
   - Implement S3/MinIO integration for storing processed data.  
4. **Monitoring**:  
   - Generate PyPlot visualizations of price trends and prediction intervals.  
   - Send email alerts for detected anomalies using SMTPLIB.

**Useful resources**:

1. [Luigi: Complex Pipelines Made Easy](https://luigi.readthedocs.io/en/stable/)  
2. [Coinbase Pro WebSocket API Docs](https://docs.pro.coinbase.com/)  
3. [Forecasting: Principles and Practice (ARIMA guide)](https://otexts.com/fpp3/arima.html)

**Is it free?**: Yes (Luigi MIT License, Coinbase API free tier)

**Python libraries / bindings**:

- Core: `luigi`, `websockets`, `numpy`  
- Analysis: `statsmodels`, `pandas`, `scikit-learn`  
- Visualization: `matplotlib`, `seaborn`  
- Storage: `boto3` (for S3), `sqlalchemy`

### **M2Crypto**

**Title**: Real-Time Bitcoin Time Series Analysis using M2Crypto

**Difficulty**: Medium (2 – it should take around 2 weeks to complete)

**Description**  
M2Crypto is a comprehensive Python library that allows developers to work with cryptographic functions. It is built on top of the OpenSSL library, providing high-level functions for encryption, decryption, digital signatures, and more. In this project, students will harness M2Crypto to ensure secure handling and processing of real-time Bitcoin data for a time series analysis task. Essential concepts covered will include digital signature verification, data encryption, and the development of secure communication channels.

**Describe Technology**

- **M2Crypto Overview**: A brief introduction to M2Crypto, its core capabilities, and its role as a binding to OpenSSL in the Python ecosystem.  
- **Cryptographic Functions**: Explanation of M2Crypto’s core functionalities, such as RSA, DSA, and EC operations, SSL connection mechanisms, and X.509 certificates handling.  
- **Installation and Setup**: Guide on installing M2Crypto, including dependencies and environment setup.  
- **Basic Examples**:  
  - Creating and verifying digital signatures.  
  - Encrypting and decrypting data using symmetric and asymmetric keys.

**Describe the Project**

- **Objective**: Implement a real-time data ingestion pipeline to fetch Bitcoin price data via an open API, securely process the data using M2Crypto, and perform exploratory time series analysis.  
- **Scope**:  
  - Set up a secure ingest pipeline using a public Bitcoin price API, such as CoinGecko or Binance.  
  - Use M2Crypto to ensure the integrity and confidentiality of the data by integrating digital signature verification on incoming data and encrypting the data for storage.  
  - Preprocess the data for time series analysis using libraries like `pandas` and `numpy`.  
  - Implement basic time series analysis techniques to analyze trends, seasonality, and volatility.  
  - Design and create visualizations of the time series data using Matplotlib or Seaborn.  
- **~~Expected Deliverables~~**~~:~~  
  - ~~Secure data ingestion pipeline script.~~  
  - ~~Time series analysis script with visual output.~~  
  - ~~Documentation highlighting lessons learned and difficulties overcome using M2Crypto for secure data handling.~~

**Useful Resources**

- [M2Crypto Documentation](https://gitlab.com/m2crypto/m2crypto/tree/master/doc) and [M2Crypto API Reference](https://m2crypto.readthedocs.io/en/latest/)  
- [OpenSSL Documentation](https://docs.openssl.org/master/man7/ossl-guide-libcrypto-introduction/) for understanding underlying cryptographic principles.  
- [Python Pandas Documentation](https://pandas.pydata.org/docs/)  
- [CoinGecko API Documentation](https://docs.coingecko.com/v3.0.1/reference/introduction) or other API documentation if using a different data source.

**Is it Free?**  
M2Crypto is an open-source library available free of charge. However, keep in mind that usage of your API of choice must comply with their respective terms of service, which may involve charges for higher tiers of usage.

**Python Libraries / Bindings**

- **M2Crypto**: For cryptographic operations and secure data handling.  
- **requests**: To fetch Bitcoin prices from publicly accessible APIs.  
- **pandas & numpy**: For data preprocessing and manipulation, especially time series data handling.  
- **matplotlib & seaborn**: To generate visualizations of the Bitcoin price trends.  
- **pytest**: Optional, for testing the data ingestion and processing pipeline to ensure robustness.

### **Mailchimp Marketing**

**Title**: Real-Time Bitcoin Price Analysis with Mailchimp Marketing  
**Difficulty**: Medium (2)

**Description**  
Mailchimp Marketing, widely recognized as a leading all-in-one email marketing and automation platform, offers powerful tools for segmenting audiences, personalizing campaigns, and analyzing results. For this project, you will gain insights into how to leverage Mailchimp's capabilities to facilitate data-driven marketing decisions using real-time Bitcoin price data.

**Describe technology**  
Mailchimp Marketing empowers users to create visually appealing email campaigns, automate workflows, and track campaign insights. It accommodates integration with various data sources and allows triggered campaigns based on specific events. With its intuitive dashboard, users can customize their marketing strategies based on the analytics provided. In this project, you will learn to use Mailchimp's API to ingest external data and trigger targeted email marketing campaigns based on real-time Bitcoin price fluctuations.

**Describe the project**  
In this project, you will implement a system that ingests real-time Bitcoin price data and links it with Mailchimp Marketing to trigger email updates. You will:

- Utilize a public API such as CoinGecko to fetch real-time Bitcoin price data.  
- Integrate this data into Mailchimp Marketing using Mailchimp's API to create segments based on specific price thresholds.  
- Set up automated email campaigns that are triggered when specific Bitcoin price changes occur, targeting users interested in Bitcoin trading or investing.  
- Perform time series analysis on historical Bitcoin price data to predict future trends and use these insights to further tailor marketing strategies.  
- Use Python packages such as `requests` for API calls, `pandas` for data manipulation, and `matplotlib` for visualizing any patterns from the data analysis.

**Useful resources**

- [Mailchimp Developer Documentation](https://developer.mailchimp.com/)  
- [CoinGecko API Documentation](https://docs.coingecko.com/v3.0.1/reference/introduction)  
- [Python Requests Library Documentation](https://docs.python-requests.org/en/master/)  
- [Python Pandas Documentation](https://pandas.pydata.org/docs/)  
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)

**Is it free?**  
Mailchimp offers a free plan with limited features that should suffice for this project. However, accessing advanced features, like certain automation options, might require a subscription. CoinGecko's API is free to use within certain limits.

**Python libraries / bindings**

- **requests**: A simple HTTP library for Python to handle API requests.  
- **pandas**: A robust library for data manipulation and analysis.  
- **matplotlib**: A comprehensive library for creating static, animated, and interactive visualizations in Python.  
- **mailchimp3**: A Python client library for Mailchimp's API, facilitating the interaction with Mailchimp services in a Python environment.

### **Mailgun**

**Title**: Real-Time Bitcoin Data Processing with Mailgun  
**Difficulty**: 1 (easy)

**Description:**  
In this project, students will use Mailgun, a popular email automation service, to set up notifications based on real-time Bitcoin price changes. The project will involve ingesting Bitcoin data using a public API and processing this data to send automated alerts when specific price thresholds are crossed. This project provides a great introduction to using APIs, handling real-time data, and integrating automated notification systems using Python.

**Describe technology:**  
Mailgun is an API-driven email service designed for sending, receiving, and tracking emails. It offers features like email validation, detailed analytics, and email infrastructure optimization. In this project, we will use Mailgun's API to send automated email notifications based on Bitcoin price data.

**Describe the project:**

- **Objective**: To monitor Bitcoin prices in real-time and send email alerts when specific thresholds are crossed.  
- **Steps:**  
  1. **Data Ingestion**: Use a public Bitcoin API (like CoinGecko) to fetch real-time price data in JSON format.  
  2. **Data Processing**: Parse the JSON response to extract the current Bitcoin price.  
  3. **Conditional Logic**: Implement logic to evaluate if the price crosses predetermined thresholds (e.g., a 5% drop from the previous hour).  
  4. **Email Alerts**: Utilize the Mailgun API to send email notifications to the user when the set conditions are met.  
  5. **Automation**: Set up a Python script to automatically execute these steps at regular intervals (e.g., every 10 minutes).

**Useful resources:**

- [Mailgun Documentation](https://documentation.mailgun.com/)  
- [CoinGecko API Documentation](https://docs.coingecko.com/v3.0.1/reference/introduction)

**Is it free?**  
Mailgun offers a free tier, but it has limitations on the number of emails sent per month. For small projects and testing, the free tier is usually sufficient.

**Python libraries / bindings:**

- **Requests**: To make HTTP requests for fetching Bitcoin price data from the API (`pip install requests`).  
- **Mailgun Python SDK**: For easily interacting with the Mailgun API (`pip install mailgun2`).  
- **Schedule**: To assist with running the script at regular intervals (`pip install schedule`).

### **Marquez**

**Title:** Tracking and Visualizing Bitcoin Transaction Lineage with Marquez​

**Difficulty:** 2 (Medium)

**Description:** 

In this project, students will leverage Marquez, an open-source metadata management service, to track and visualize the lineage of Bitcoin transaction data. The project involves ingesting Bitcoin transaction data from a public API, processing and storing this data, and utilizing Marquez to monitor data flow and transformations. This hands-on experience will introduce students to data lineage concepts, metadata management, and the importance of data governance in the context of cryptocurrency transactions.

**Describe Technology:** 

Marquez is an open-source metadata service designed for the collection, aggregation, and visualization of a data ecosystem's metadata. It maintains the provenance of how datasets are consumed and produced, provides global visibility into job runtime and dataset access frequency, and centralizes dataset lifecycle management.Marquez enables highly flexible data lineage queries across all datasets, efficiently associating dependencies between jobs and the datasets they produce and consume. 

**Describe the Project:**

**Objective:** To monitor, process, and visualize the lineage of Bitcoin transaction data using Marquez.​

**Steps:**

1. **Data Ingestion:** Utilize a public Bitcoin API (such as CoinGecko) to fetch real-time transaction data in JSON format.​  
2. **Data Processing:** Parse the JSON response to extract relevant transaction details, including transaction IDs, timestamps, input and output addresses, and amounts.​  
3. **Data Storage:** Store the processed transaction data in a database (e.g., PostgreSQL) for further analysis and lineage tracking.​  
4. **Marquez Integration:**  
   * **Metadata Collection:** Implement Marquez's metadata API to collect metadata about the data sources, transformations, and outputs related to the Bitcoin transaction data.  
   * **Lineage Tracking:** Use Marquez to track the lineage of the transaction data as it flows through various processing stages, ensuring transparency and traceability.​  
5. **Data Visualization:** Utilize Marquez's web user interface to visualize the data lineage, showing the interdependencies between datasets and the transformations applied to the Bitcoin transaction data.​  
6. **Automation:** Develop a Python script to automate the data ingestion, processing, and metadata collection processes, ensuring continuous tracking and updating of data lineage.​

**Useful Resources:**

* [Marquez Documentation](https://marquezproject.ai/)  
* [CoinGecko API Documentation](https://docs.coingecko.com/v3.0.1/reference/introduction)  
* [Marquez GitHub Repository](https://github.com/MarquezProject/marquez)

**Is it Free?** 

Marquez is an open-source project released under the Apache 2.0 license, making it free to use and modify.CoinGecko also offers a free tier for their API, which should be sufficient for educational and small-scale projects.​

**Python Libraries / Bindings:**

* **Requests:** To make HTTP requests for fetching Bitcoin transaction data from the API.​  
* **psycopg2:** To interact with the PostgreSQL database for storing transaction data.​  
* **Marquez Python Client:** For interacting with the Marquez API to collect and manage metadata.​  
* **Schedule:** To assist with running the script at regular intervals for continuous data ingestion and processing.

This project offers students practical experience in data governance and lineage tracking within the cryptocurrency domain, highlighting the significance of metadata management in ensuring data quality and transparency.​

### **Metaflow**

**Title**: Real-Time Bitcoin Analysis using Metaflow

**Difficulty**: Medium (2)

**Description**:  
Metaflow is a human-centric framework that makes it easy to build and manage real-life data science projects. Developed by Netflix, it allows data scientists to focus on data processing and insights while abstracting away the complexities of infrastructure. It integrates seamlessly with existing Python environments and provides enhanced scalability and reproducibility in data science workflows. Students will delve into the core functionalities of Metaflow through the implementation of a real-time Bitcoin data processing pipeline.

**Describe technology**:  
Metaflow focuses on easing the development of data science applications by offering features like version control of data, parameters, and code, built-in scalability via infrastructure abstraction, and enhanced workflow management with step functions.

- **Key Features**:  
  - Workflow Definitions: Easily define workflows using Python.  
  - Data Artifacts: Automatically version and store data artifacts.  
  - Scalability: Scale operations by leveraging cloud infrastructure.  
  - Fail-Safe Execution: Manage retries and error handling with ease.

**Describe the project**:  
This project involves developing a real-time Bitcoin price tracking and analysis system using Metaflow. The tasks include:

1. **Data Ingestion**:  
     
   - Set up a Python script to ingest real-time Bitcoin prices from a public API (e.g., CoinGecko).  
   - Define a Metaflow flow to manage the data ingestion pipeline.

   

2. **Data Processing**:  
     
   - Implement step functions in Metaflow to process the ingested data. This can involve transformation tasks such as cleaning, normalization, and generating new time series features.

   

3. **Time-Series Analysis**:  
     
   - Perform basic time series analysis to identify trends and patterns in Bitcoin's price movements.  
   - Use libraries such as NumPy and Pandas for analysis within Metaflow steps.

   

4. **Visualization**:  
     
   - Generate real-time visualizations of Bitcoin price trends using data processed by Metaflow. Consider using matplotlib for chart generation.

**Useful resources**:

- [Metaflow Documentation](https://docs.metaflow.org)  
- [CoinGecko API Documentation](https://docs.coingecko.com/v3.0.1/reference/introduction)  
- [NumPy Documentation](https://numpy.org/doc/stable/)  
- [Pandas Documentation](https://pandas.pydata.org/docs/)  
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)

**Is it free?**  
Yes, Metaflow is open-source and free to use. However, utilizing cloud infrastructure for executing flows at scale may incur costs depending on the cloud provider.

**Python libraries / bindings**:

- **Metaflow SDK**: Install Metaflow using `pip install metaflow` to define and execute workflows.  
- **Requests**: Utilize for API calls to fetch real-time Bitcoin data (`pip install requests`).  
- **NumPy & Pandas**: Core libraries for data manipulation and analysis within Metaflow steps.  
- **Matplotlib**: Use for creating visualizations (`pip install matplotlib`).

### **Metasfresh**

**Title:** Real-Time Sales Data Processing with metasfresh ERP​

**Difficulty:** 2 (Medium)​

**Description:** This project introduces students to real-time data processing within the metasfresh ERP system. By the end of this project, students will have set up metasfresh, simulated real-time sales data, and implemented automated processes to respond to specific sales events as they occur.​

**Describe technology:**

* **metasfresh ERP:** An open-source ERP solution offering functionalities like sales, purchasing, and inventory management. It provides APIs for integration with external systems.​  
* **Python:** A versatile programming language used here to simulate real-time data generation and processing.​

**Describe the project:**

* **Objective:** To simulate real-time sales transactions, ingest them into metasfresh ERP, and implement automated responses to specific sales events.​  
* **Steps:**  
  1. **System Setup:** Install and configure metasfresh ERP on a local or cloud-based server.​  
  2. **Data Simulation:** Develop a Python script to simulate real-time sales transactions, generating data continuously to mimic a live sales environment.​  
  3. **Real-Time Data Ingestion:** Use Python's capabilities to send simulated sales data directly to metasfresh via its REST API.​  
  4. **Data Processing:** Implement logic within metasfresh to process incoming sales data, such as updating inventory levels or generating sales reports.​  
  5. **Automated Responses:** Configure metasfresh to trigger specific actions (e.g., inventory updates, sales reports) based on incoming sales data.​  
  6. **Monitoring:** Implement logging and monitoring within metasfresh to ensure data is processed correctly and to facilitate troubleshooting.​

**Useful resources:**

* [metasfresh Documentation](https://docs.metasfresh.org/)​  
* [Python Requests Library Documentation](https://docs.python-requests.org/en/latest/)​

**Is it free?** Yes, metasfresh is an open-source platform and free to use. Python is also open-source and freely available.​

**Python libraries / bindings:**

* `requests`: To make HTTP requests for interacting with metasfresh's API (install via `pip install requests`).​  
* `pandas`: For data manipulation and analysis, if needed (install via `pip install pandas`).​

This project provides students with practical experience in setting up an ERP system, simulating real-time data, and integrating automated processes, all within a controlled environment using familiar technologies.​

### **Metpy**

**Title:** Real-Time Weather Data Analysis with MetPy​

**Difficulty:** 2 (Medium)​

**Description:** This project introduces students to MetPy, an open-source Python library tailored for meteorological data analysis and visualization. Participants will set up a real-time data processing pipeline to fetch, analyze, and visualize live weather data, gaining hands-on experience with MetPy's capabilities.

**Describe technology:** MetPy is a collection of tools in Python designed for reading, visualizing, and performing calculations with weather data. Built on top of the scientific Python ecosystem—including libraries like NumPy, SciPy, Matplotlib, and xarray—MetPy provides functionalities such as unit-aware calculations, support for various meteorological file formats, and specialized plotting routines like Skew-T and station plots. 

**Describe the project:**

* **Objective:** To develop a real-time weather data analysis system that fetches live meteorological data, performs essential analyses, and visualizes the results using MetPy.​  
* **Steps:**  
  1. **Data Acquisition:** Utilize public APIs (e.g., OpenWeatherMap) to fetch real-time weather data, including parameters like temperature, humidity, wind speed, and atmospheric pressure.​  
  2. **Data Processing:** Employ MetPy's unit-aware calculations to process and analyze the retrieved data, such as computing dew point, wind chill, or other derived meteorological quantities.​  
  3. **Visualization:** Create visual representations of the data using MetPy's plotting capabilities, including time series plots, Skew-T diagrams, or station plots to depict the spatial distribution of weather parameters.  
  4. **Automation:** Develop a Python script that automates the data fetching, processing, and visualization steps at regular intervals (e.g., every hour) to maintain an up-to-date analysis.​

**Useful resources:**

* [MetPy Documentation](https://unidata.github.io/MetPy/latest/index.html)​  
* [OpenWeatherMap API Documentation](https://openweathermap.org/api)​  
* [Python's Requests Library Documentation](https://docs.python-requests.org/en/latest/)​

**Is it free?** Yes, MetPy is an open-source library and free to use. OpenWeatherMap offers a free tier for its API with certain limitations, which should suffice for educational purposes.​

**Python libraries / bindings:**

* `metpy`: For meteorological data analysis and visualization (install via `pip install metpy`).​  
* `requests`: To make HTTP requests for fetching data from APIs (install via `pip install requests`).​  
* `pandas`: For data manipulation and analysis (install via `pip install pandas`).​  
* `matplotlib`: For creating static, animated, and interactive visualizations (install via `pip install matplotlib`).​

This project offers students practical experience in handling real-time data, performing meteorological analyses, and creating visualizations, all within the Python ecosystem.

### **Microsoft Power BI**

**Title**: Real-Time Bitcoin Data Ingestion and Time Series Analysis using Microsoft Power BI

**Difficulty**: 3 (difficult)

**Description**:  
This project focuses on leveraging Microsoft Power BI to ingest, visualize, and analyze Bitcoin price data in real-time. Microsoft Power BI is a suite of business analytics tools that deliver insights by analyzing datasets and generating interactive reports and dashboards. The project involves setting up a data ingestion pipeline and creating a sophisticated Power BI dashboard for time series analysis of Bitcoin's market trends.

**Describe Technology**:

- **Microsoft Power BI**: A powerful business analytics tool that facilitates the visualization and sharing of data insights. It is designed to handle large datasets and supports connections to a wide range of data sources. Key features include customizable dashboards, real-time data stream processing, and integrated artificial intelligence capabilities. Power BI empowers users to build advanced reports and dashboards and offers robust integration capabilities with Python for data analysis.

**Describe the Project**:

- **Objective**: Set up a data ingestion system to collect Bitcoin price data in real-time from a public API (such as CoinGecko or CryptoCompare), feed this data into Microsoft Power BI for visualization, and perform time series analysis.  
- **Steps**:  
  1. **API Data Ingestion**: Implement a Python-based solution to fetch real-time Bitcoin price data from a public API. Use libraries like `requests` to acquire the data in JSON format.  
  2. **Data Preparation**: Transform the raw data to ensure it is suitable for input into Power BI. This can include cleaning, normalization, and conversion to CSV or other compatible formats.  
  3. **Power BI Setup**: Utilize Power BI's data flow capabilities to import the transformed data. Configure a scheduled refresh to ensure data is regularly updated.  
  4. **Dashboard Creation**: Design a comprehensive dashboard to include key metrics such as current price, historical trends, moving averages, and volatility indexes. Incorporate Python scripts for advanced time series analysis via Power BI’s Python integration.  
  5. **Real-Time Analytics**: Implement streaming datasets in Power BI to visualize data in real-time, providing continuous updates to the dashboard.  
  6. **Time Series Analysis**: Conduct a detailed time series analysis using Python integrated within Power BI. Employ models to analyze trends, seasonality, and fluctuations, and predict future price movements.

**Useful Resources**:

- Microsoft Power BI Documentation: [Link](https://docs.microsoft.com/en-us/power-bi/)  
- Python Support in Power BI: [Link](https://learn.microsoft.com/en-us/power-bi/connect-data/desktop-python-visuals)  
- CoinGecko API Documentation: [Link](https://docs.coingecko.com/v3.0.1/reference/introduction)

**Is it Free?**  
Microsoft Power BI offers a free version with limited functionalities. However, for real-time streaming and advanced features, a Power BI Pro subscription may be required.

**Python Libraries / Bindings**:

- `pandas`: For data manipulation and transformation before feeding into Power BI. Install using `pip install pandas`.  
- `requests`: For extracting data from APIs. Install using `pip install requests`.  
- `matplotlib` & `seaborn`: Optional for preliminary visualization and analysis before uploading data to Power BI. Install using `pip install matplotlib seaborn`.

### **MLflow Computational framework**

**Title**: Real-time Bitcoin Data Processing with MLflow  
**Difficulty**: 1 (easy)

**Description**  
This project is designed to give you a hands-on introduction to MLflow, an open-source platform to manage the machine learning lifecycle, which includes experimentation, reproducibility, and deployment. By the end of this project, you will have set up a simple real-time data processing pipeline for Bitcoin prices and explored basic MLflow functionalities to document and manage your work effectively.

**Describe technology**

- **MLflow** is a tool that helps manage the machine learning lifecycle, encompassing four key functions:  
  - **Tracking experiments** to record and compare parameters and results (MLflow Tracking).  
  - **Packaging code** into reproducible runs using Conda and Docker (MLflow Projects).  
  - **Managing and deploying models** from various ML libraries (MLflow Models).  
  - **Creating REST APIs** for the deployed models for easy access (Model Registry).

**Describe the project**

- **Objective**: Implement a real-time data ingestion pipeline for Bitcoin prices and apply basic time series analysis in Python, using MLflow to track the experiment and results.  
- **Steps**:  
  1. **Data Collection**: Use a public API like CoinGecko to ingest real-time Bitcoin data every few minutes.  
  2. **Data Processing**: Write a basic Python script to clean and prepare the collected data for analysis.  
  3. **Time Series Analysis**: Implement a simple moving average algorithm to analyze Bitcoin price trends.  
  4. **MLflow Integration**: Use MLflow to:  
     - Track the parameters and metrics of your analysis, such as time interval and average price.  
     - Log the versions of the software packages used in your project.  
     - Document and save the results for future reference or comparison.  
  5. **Visualization**: Optional step to visualize the time series analysis using a library like Matplotlib.

**Useful resources**

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)  
- [CoinGecko API Documentation](https://www.coingecko.com/en/api)  
- [Python's Requests Library Documentation](https://docs.python-requests.org/en/latest/)

**Is it free?**  
Yes, MLflow is open-source and can be used for free. CoinGecko provides free access to its API with rate limits.

**Python libraries / bindings**

- **MLflow**: Install MLflow using `pip install mlflow`.  
- **Requests**: For API calls, use Python's requests library, installable via `pip install requests`.  
- **Pandas**: For data manipulation, use Pandas via `pip install pandas`.  
- **Matplotlib**: For optional data visualization, install via `pip install matplotlib`.

### **mock**

**Title:** Unit Testing Cryptocurrency Applications with Python's `unittest.mock`​

**Difficulty:** 2 (Medium)​

**Description:** This project introduces students to Python's `unittest.mock` library, a powerful tool for creating mock objects and conducting unit tests. Participants will develop a cryptocurrency price alert application and utilize `unittest.mock` to simulate external API responses, ensuring the application's reliability without relying on real-time data.​

**Describe technology:** `unittest.mock` is a library for testing in Python. It allows developers to replace parts of their system under test with mock objects and make assertions about how they have been used. This is particularly useful for isolating the code under test and controlling its environment during testing.

**Describe the project:**

* **Objective:** To develop a cryptocurrency price alert application and implement unit tests using `unittest.mock` to simulate API responses.  
* **Steps:**  
  1. **Application Development:**  
     * Create a Python application that fetches cryptocurrency prices from a public API (e.g., CoinGecko) and sends alerts when prices cross certain thresholds.  
  2. **Mocking External APIs:**  
     * Use `unittest.mock` to simulate responses from the cryptocurrency API, allowing testing of various price scenarios without making actual API calls.  
  3. **Writing Unit Tests:**  
     * Develop unit tests to verify that the application correctly processes API data and triggers alerts as expected.  
  4. **Assertion Checks:**  
     * Implement assertions to ensure that the application behaves correctly under different simulated conditions, such as price increases, decreases, or API errors.

**Useful resources:**

* [unittest.mock Documentation](https://docs.python.org/3/library/unittest.mock.html)​  
* [Real Python: Understanding the Python Mock Object Library](https://realpython.com/python-mock-library/)​  
* [CoinGecko API Documentation](https://docs.coingecko.com/v3.0.1/reference/introduction)

**Is it free?** Yes, `unittest.mock` is part of Python's standard library and is free to use. CoinGecko provides free access to its API with certain rate limits, suitable for educational purposes.​

**Python libraries / bindings:**

* `unittest.mock`: For creating mock objects and patching dependencies during testing.  
* `requests`: To make HTTP requests for fetching data from APIs.​  
* `pandas`: For data manipulation and analysis.​

This project offers students practical experience in developing applications that interact with external APIs and implementing unit tests using mock objects to ensure code reliability without depending on live data.​

### **Modin**

**Title**: Process Real-Time Bitcoin Data with Modin

**Difficulty**: 1 (easy)

**Description**  
Modin is a parallel DataFrame library compatible with pandas that accelerates data processing by making use of all available CPU cores. It maintains full compatibility with pandas, allowing for a seamless transition of existing pandas code to take advantage of Modin's improved performance. Modin provides substantial speedup in data processing tasks, making it an excellent choice for handling large datasets or processing data in real-time without having to change existing code dramatically.

**Describe technology**

- Modin parallelizes operations on pandas DataFrames by distributing the computation across all available CPU cores, thus speeding up the data processing.  
- It supports a majority of pandas APIs without requiring changes to the existing pandas codebase.  
- Modin can be integrated with Dask or Ray as backends for parallel processing, thus providing scalability beyond a single machine.

**Describe the project**

- **Objective**: Implement a real-time data processing pipeline using Modin to conduct time series analysis on Bitcoin price data obtained from a public API (e.g., CoinGecko).  
- **Steps**:  
  1. Fetch real-time Bitcoin price data from a chosen API using basic Python libraries (e.g., `requests`).  
  2. Initialize a Modin DataFrame to ingest and preprocess the data. This can include data cleaning and transformation into a time series format.  
  3. Perform basic time series analysis using Modin, such as calculating moving averages, price variances, and detecting patterns over time.  
  4. Visualize the results using a compatible library (e.g., matplotlib) to plot time series trends.  
  5. Conclude with a simple report summarizing findings from the analysis.

**Useful resources**

- [Official Modin Documentation](https://modin.readthedocs.io/en/latest/)  
- [Getting Started with Modin](https://modin.readthedocs.io/en/latest/getting_started/index.html)  
- [CoinGecko API Documentation](https://docs.coingecko.com/v3.0.1/reference/introduction)

**Is it free?**  
Yes, both Modin and the CoinGecko API are free to use, with the latter having limitations on the number of requests you can make within a certain period.

**Python libraries / bindings**

- **Modin**: Core library to accelerate DataFrame operations, compatible with pandas. Install via `pip install modin[dask]` or `pip install modin[ray]` depending on the chosen backend or `pip install modin[all]`.  
- **Requests**: To fetch real-time Bitcoin data: `pip install requests`.  
- **Matplotlib**: To visualize the analyzed data: `pip install matplotlib`.

### **msgpack**

**Title**: Real-Time Bitcoin Price Analysis with MsgPack

**Difficulty**: 1 (easy)

**Description**:  
MsgPack (or MessagePack) is an efficient binary serialization format that is analogous to JSON, but its compact nature makes it more suitable for transferring large volumes of data quickly. It is ideal for real-time data processing tasks where bandwidth and speed are crucial, such as ingesting and processing real-time Bitcoin price data. This project will allow students to understand the core functionalities of MsgPack in Python and then use it in a practical scenario focusing on time series analysis of Bitcoin pricing data.

**Describe technology**:

- MsgPack is a lightweight and efficient binary serialization format.  
- Unlike JSON, MsgPack stores data in a binary format, reducing file sizes and optimizing data throughput.  
- It enables fast encoding/decoding of data, which is particularly beneficial in scenarios involving real-time data processing or network communication.  
- Python's `msgpack` library can easily serialize and deserialize Python data structures using MessagePack format.

**Describe the project**:

- Students will gather Bitcoin price data from a public API such as CoinGecko or CryptoCompare.  
- Convert the retrieved JSON formatted Bitcoin price data into MsgPack format.  
- Store the serialized data locally or transmit it to another endpoint to mock real-time data streaming.  
- Develop a basic system to deserialize the data back into a Python-readable format for time series analysis.  
- Implement simple time series analysis techniques such as moving averages to understand recent price trends.  
- Visualize the price trends over time using a plotting library like Matplotlib or Seaborn for deeper insights.

**Useful resources**:

- [MsgPack Official Website](https://msgpack.org/)  
- [msgpack-python Documentation](https://pypi.org/project/msgpack/)  
- [CoinGecko API Documentation](https://docs.coingecko.com/v3.0.1/reference/introduction)

**Is it free?**:  
Yes, MsgPack is open-source and freely available for use. The Bitcoin data retrieval from public APIs like CoinGecko or CryptoCompare also offers free tiers with rate limits.

**Python libraries / bindings**:

- `msgpack`: The main Python library for encoding and decoding data in the MessagePack format. Install via `pip install msgpack`.  
- `requests`: For making HTTP requests to fetch Bitcoin price data. Install via `pip install requests`.  
- `matplotlib / seaborn`: For data visualization to plot time series analysis results. Install via `pip install matplotlib seaborn`.

This project provides an approachable way to explore real-time data processing using MsgPack, coupled with practical experience in ingesting and analyzing financial time-series data.

### **NLTK**

**Title:** Real-Time Bitcoin Sentiment Analysis Using NLTK and Selenium​

**Difficulty:** 3 (Difficult)​

**Description:** 

This project involves utilizing the Natural Language Toolkit (NLTK) in conjunction with Selenium-based Twitter scraping tools to perform real-time sentiment analysis on Bitcoin-related tweets. Students will collect tweets mentioning Bitcoin without relying on paid APIs, process the text data, and analyze sentiment trends over time. This analysis will provide insights into public sentiment fluctuations toward Bitcoin and their potential influence on its price movements.​

**Describe Technology:** 

NLTK, or Natural Language Toolkit, is a comprehensive Python library designed for natural language processing (NLP) tasks. It offers tools for classification, tokenization, stemming, tagging, parsing, and semantic reasoning, making it ideal for sentiment analysis. Key components include:​

* **Tokenizers:** To split text into words or sentences.​  
* **POS Tagging:** To assign parts of speech to each word.​  
* **Sentiment Analysis:** Functions like VADER (Valence Aware Dictionary and sEntiment Reasoner) to determine the sentiment polarity of the text.​  
* **Support for Training Custom Models:** For specific NLP tasks.​

Selenium is a powerful tool for automating web browsers, enabling the scraping of web content without relying on APIs.In this project, Selenium-based Twitter scrapers, such as the [selenium-twitter-scraper](https://github.com/godkingjay/selenium-twitter-scraper), will be used to collect Bitcoin-related tweets. This scraper automates the extraction of tweets from specified Twitter profiles or search results, facilitating data collection without the need for API access.

**Describe the Project:**

**Objective:** To analyze the sentiment of real-time Bitcoin-related tweets and perform time series analysis on sentiment trends.​

**Steps:**

1. **Data Ingestion:**  
   * **Collect Tweets:** Utilize Selenium-based Twitter scrapers to gather real-time tweets mentioning Bitcoin.  
2. **Preprocessing:**  
   * **Cleaning:** Remove noise such as URLs, mentions, hashtags, and special characters from the text data.​  
   * **Tokenization:** Break down text into individual words or tokens using NLTK's tokenizers.​  
   * **Stop-word Removal:** Eliminate common words that do not contribute to sentiment (e.g., 'is', 'and', 'the').​  
3. **Sentiment Analysis:**  
   * **VADER Sentiment Analyzer:** Apply NLTK’s VADER sentiment analyzer, which is well-suited for social media texts, to determine the sentiment polarity of each tweet.​  
   * **Custom Models:** Optionally, train a custom sentiment analysis model using labeled datasets for more tailored analysis.​  
4. **Real-Time Processing:**  
   * **Automation:** Develop a Python script to automate data collection and processing at regular intervals (e.g., every 10 minutes) to maintain real-time sentiment tracking.​  
5. **Time Series Analysis:**  
   * **Visualization:** Utilize libraries such as Matplotlib to plot sentiment scores over time.​  
   * **Exploratory Data Analysis:** Identify trends, patterns, and outliers in sentiment data.​  
6. **Outcome Analysis:**  
   * **Correlation Analysis:** Compare sentiment trends to real-time Bitcoin price changes to analyze potential correlations.​

**Useful Resources:**

* [NLTK Documentation](http://www.nltk.org/)  
* [VADER Sentiment Analysis](https://github.com/cjhutto/vaderSentiment)  
* [selenium-twitter-scraper GitHub Repository](https://github.com/godkingjay/selenium-twitter-scraper)  
* [Pandas Documentation](https://pandas.pydata.org/docs/)  
* [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)

**Is it Free?**

Yes, all the suggested tools and libraries are free and open-source. NLTK, Selenium, and the selenium-twitter-scraper can be used without any associated costs. However, when scraping data, it's essential to comply with Twitter's terms of service and ensure ethical data collection practices.​

**Python Libraries / Bindings:**

* **NLTK:** Install with `pip install nltk`. Essential for text processing and sentiment analysis tasks.​  
* **Selenium:** Install with `pip install selenium`. Used for automating web browser interactions to scrape tweets.  
* **Pandas:** Install with `pip install pandas`. For managing and analyzing time series data.​  
* **Matplotlib:** Install with `pip install matplotlib`. For visualizing data trends and analysis results.​

### **Numba**

**Title**: Real-Time Bitcoin Price Analysis with Numba

**Difficulty**: 1 (Easy)

**Description**  
This project involves using Numba, a just-in-time compiler that translates a subset of Python and NumPy code into fast machine code, to ingest and process real-time Bitcoin price data. The task will focus on implementing basic functionalities of Numba to optimize computational parts of a Python-based time series analysis on Bitcoin prices.

**Describe Technology**

- Numba is designed to accelerate numerical Python functions, making them nearly as fast as compiled languages like C or FORTRAN.  
- Key functionalities include:  
  - Just-In-Time (JIT) Compilation: Numba compiles Python functions “just-in-time” to improve runtime performance.  
  - Easy integration with NumPy: Numba can efficiently handle NumPy operations, improving array processing speed.  
  - Parallel Computing: Supports GPU and multi-core CPUs to enable parallel computations.

**Describe the Project**

- **Objective**: Implement a real-time data processing system for Bitcoin prices with optimized performance using Numba.  
    
- **Steps**:  
    
  1. **Data Ingestion**: Use a basic Python library, like `requests`, to fetch real-time Bitcoin price data from a public API (e.g., CoinGecko).  
  2. **Optimization with Numba**: Write a function to analyze the time series data, focusing on tasks like simple moving averages or returns. Use Numba's JIT decorator to optimize these functions.  
  3. **Real-time Processing**: Implement a loop to fetch data at regular intervals and process it using the Numba-optimized functions.  
  4. **Visualization**: Use matplotlib to plot the real-time price changes and computed metrics, providing a visual insight into Bitcoin's price trends.


- **Outcome**: Students will gain experience in boosting Python performance with Numba, applying it to time-sensitive cryptocurrency data analysis.

**Useful Resources**

- [Numba Documentation](https://numba.readthedocs.io/en/stable/index.html)  
- [CoinGecko API Documentation](https://docs.coingecko.com/v3.0.1/reference/introduction)  
- [Matplotlib for Data Visualization](https://matplotlib.org/)

**Is it Free?**  
Yes, Numba is an open-source Python library freely available for use. Access to cryptocurrency APIs like CoinGecko can also be used for free within certain limits.

**Python Libraries / Bindings**

- **Numba**: For optimizing numerical computations (`pip install numba`)  
- **Requests**: To fetch real-time data from APIs (`pip install requests`)  
- **NumPy**: For numerical operations (`pip install numpy`)  
- **Matplotlib**: To visualize the data (`pip install matplotlib`)

### **Ollama**

**Title:** Developing a Local AI-Powered Document Search Engine with Ollama​

**Difficulty:** 3 (Difficult)​

**Description:** This project guides students through building a local, AI-driven search engine using Ollama, enabling efficient and secure querying of personal documents without relying on external servers. Participants will develop a Python application that leverages Large Language Models (LLMs) to understand natural language queries and retrieve relevant information from local files.

**Describe technology:** Ollama is a platform that facilitates running Large Language Models (LLMs) locally, allowing for advanced AI functionalities without the need for cloud-based services. This ensures data privacy and security, as all processing occurs on the user's machine.

**Describe the project:**

* **Objective:** To create a local search engine capable of understanding natural language queries and retrieving pertinent information from personal documents using Ollama's LLM capabilities.  
* **Steps:**  
  1. **Set Up the Development Environment:**  
     * Install Ollama on your local machine to enable LLM functionalities.  
     * Install necessary Python libraries, such as `faiss-cpu` for similarity search, `sentence-transformers` for embedding generation, and `streamlit` for building the user interface.  
  2. **Document Processing:**  
     * Develop scripts to parse and extract text from various document formats (e.g., PDFs, Word documents).  
     * Use `sentence-transformers` to convert document text into embeddings, facilitating efficient similarity searches.  
  3. **Indexing Documents:**  
     * Utilize `faiss-cpu` to index document embeddings, enabling rapid similarity searches based on user queries.  
  4. **Building the User Interface:**  
     * Create an interactive web interface using `streamlit` where users can input natural language queries.  
     * Display search results with relevant document snippets and links to the original files.  
  5. **Implementing the Search Functionality:**  
     * Process user queries by generating embeddings and performing similarity searches against the indexed documents.  
     * Leverage Ollama's LLM to interpret queries and enhance search accuracy.  
  6. **Testing and Optimization:**  
     * Conduct thorough testing to ensure accurate search results and optimize performance for large document collections.

**Useful resources:**

* [Ollama GitHub Repository](https://github.com/ollama/ollama)​  
* [Building an AI-Driven Local Search Engine with Ollama](https://adasci.org/hands-on-guide-to-build-an-ai-driven-local-search-engine-with-ollama/)​[adasci.org](https://adasci.org/hands-on-guide-to-build-an-ai-driven-local-search-engine-with-ollama/)  
* [Python Code Recipes for Ollama](https://mljar.com/docs/ollama-python/)​

**Is it free?** Yes, Ollama is open-source and free to use. The required Python libraries are also open-source.

**Python libraries / bindings:**

* `faiss-cpu`: For efficient similarity search and clustering of dense vectors.  
* `sentence-transformers`: To generate embeddings for sentences and documents.  
* `streamlit`: For building interactive web applications.  
* `ollama`: To interact with Ollama's LLM capabilities.​

This project provides students with practical experience in natural language processing, information retrieval, and building AI-powered applications that prioritize data privacy by operating entirely on local machines.

### **Ollama Python**

**Title**: Custom Bitcoin Chatbot with Ollama Python

**Difficulty**: Difficult

**Description**

- **Technology Overview:**  
  Ollama Python is the Python SDK for Ollama, enabling integration of locally deployed LLMs into Python applications. This project uses Ollama Python to build a chatbot providing real-time Bitcoin insights.  
    
- **Project Details:**  
  This project involves:  
    
  - Deploying an LLM locally with Ollama.  
  - Using Ollama Python to integrate the LLM into a chatbot application.  
  - Ingesting real-time Bitcoin data and news.  
  - Enabling the chatbot to answer questions about current prices, trends, sentiment, and provide time series-based forecasts.  
  - Implementing natural language understanding to interpret user queries and generate responses.  
  - Ensuring the chatbot handles multiple users and delivers timely information.


  The complexity lies in building a responsive chatbot, integrating real-time data, and ensuring accurate LLM responses.

**Useful Resources**

- [Ollama Python GitHub](https://github.com/ollama/ollama-python)  
- [CoinGecko API Documentation](https://docs.coingecko.com/v3.0.1/reference/introduction)  
- [NewsAPI](https://newsapi.org/docs)

**Is it Free?**

- **Ollama Python:** Yes, open-source.  
- **APIs:** Free tiers available.

**Python Libraries**

- `Ollama-python`: Enables integration of locally deployed LLMs into the Bitcoin chatbot for real-time insights.  
- `requests`: `pip install requests:` Facilitates API calls to fetch real-time Bitcoin data and news.  
- `pandas`: `pip install pandas:` Provides data manipulation and analysis tools for processing Bitcoin price trends and forecasts.

### **Ollama Python \#2**

**Title**: Real-Time Bitcoin Price Analysis and Forecasting with Ollama-Python  
**Difficulty**: 3 (difficult)  
**Description**  
**Describe technology**  
Ollama-Python is a Python library for interacting with Ollama, a framework for deploying and running large language models (LLMs) locally. Ollama simplifies running models like LLaMA, Mistral, or CodeLLaMA on your machine, enabling text generation, data analysis, and fine-tuning. It supports REST API integration and real-time inference, making it suitable for combining LLMs with data pipelines.

**Describe the project**  
This project involves building a real-time Bitcoin price analysis system that integrates Ollama-Python to generate insights and forecasts from streaming data. Students will:

1. **Ingest real-time Bitcoin price data** from a public API (e.g., CoinGecko or Binance) using Python’s `requests` or `websockets`.  
2. **Process time series data** with `pandas` to calculate metrics (e.g., moving averages, volatility) and detect anomalies.  
3. **Integrate Ollama-Python** to run an LLM (e.g., Mistral-7B) for two tasks:  
   - Generate natural language summaries of price trends (e.g., "Bitcoin surged 5% in the last hour").  
   - Forecast short-term price movements by fine-tuning the LLM on historical data.  
4. **Build a real-time dashboard** using `Plotly` or `Dash` to visualize raw data, metrics, and LLM-generated insights.  
5. **Implement a streaming pipeline** to ensure low-latency processing (e.g., using `Faust` for stream processing).

**Challenges**:

- Optimizing Ollama’s inference speed for real-time use.  
- Structuring prompts to extract meaningful insights from time series data.  
- Handling computational constraints when running LLMs alongside data pipelines.

**Useful resources**

- [Ollama-Python Documentation](https://github.com/ollama/ollama-python)  
- [CoinGecko API Guide](https://www.coingecko.com/en/api)  
- [Time Series Forecasting with Machine Learning](https://otexts.com/fpp3/) (Chapter 11\)

**Is it free?**  
Ollama is open-source and free, but running large LLMs (e.g., 7B+ parameter models) requires significant RAM/GPU resources. Students may need cloud credits for GPU instances.

**Python libraries / bindings**

- `ollama-python`: Interact with Ollama’s local API to run LLMs.  
- `pandas`/`numpy`: Time series processing.  
- `requests`/`websockets`: Fetch real-time Bitcoin data.  
- `plotly`/`dash`: Visualization dashboard.  
- `faust`/`kafka-python`: Stream processing (optional for advanced pipelines).

### **OpenAI Gym / Gymnasium**

**Title:** Real-Time Bitcoin Price Analysis with Gymnasium​

**Difficulty:** 3 (Difficult)​

**Description:** Gymnasium is a maintained fork of OpenAI's Gym library, designed to support the development and comparison of reinforcement learning algorithms. In this project, students will leverage Gymnasium's flexible environment-creation capabilities to ingest, process, and perform time series analysis on real-time Bitcoin price data. By simulating a dynamic environment where Bitcoin prices are treated as state information, students can develop and test predictive models for future price trends.​

**Technology Overview:**

* **Gymnasium:** An open-source Python library that provides a standard API for reinforcement learning environments. It simplifies the process of creating and modifying simulated environments where different algorithms can be tested. ​  
  * **Core Concepts:**  
    * **Environments:** Instances of scenarios where agents operate.  
    * **Agents:** Algorithms that interact with these environments.  
    * **Interaction Loop:** Continuous process where states, actions, rewards, and observations are processed.  
* **Utility for the Project:** Leveraging Gymnasium’s environments to simulate and interact with live Bitcoin price data, treating each incoming data point as part of a continuously evolving state.​

**Project Description:**

**Objective:** Develop a system using Gymnasium that ingests real-time Bitcoin price data and simulates a dynamic environment for time series analysis.​

**Phases:**

1. **Data Collection:**  
   * Utilize public APIs such as CoinGecko or Binance to continuously fetch real-time Bitcoin price data. Implement a data ingestion pipeline using Python.​  
2. **Environment Setup:**  
   * Create a custom Gymnasium environment where the state is defined by real-time Bitcoin price, volume, and other relevant metrics.​  
3. **Interaction Loop:**  
   * Develop an agent that interacts with this environment, implementing strategies for predicting price movements and performing time series analyses.​  
4. **Analysis:**  
   * Apply time series analysis techniques like ARIMA, moving averages, or Fourier transforms using Python libraries to forecast price movements or detect anomalies.​  
5. **Reporting:**  
   * Visualize and report insights from the time series analysis, providing interpretations of trends and forecasting future price directions.​

**Useful Resources:**

* [Gymnasium Documentation](https://gymnasium.farama.org/index.html)  
* [CoinGecko API Documentation](https://docs.coingecko.com/v3.0.1/reference/introduction)  
* [pandas Documentation](https://pandas.pydata.org/docs/)  
* [Statsmodels Documentation](https://www.statsmodels.org/stable/index.html)  
* [matplotlib Documentation](https://matplotlib.org/stable/contents.html)

**Is it Free?** 

Yes, Gymnasium is open-source and freely available. Public APIs like CoinGecko and Binance offer free tiers for accessing real-time market data.​

**Python Libraries / Bindings:**

* **Gymnasium:** Install using `pip install gymnasium`. Enables the creation and simulation of custom environments.​  
* **pandas:** Useful for handling and structuring Bitcoin price data fetched from APIs. Install via `pip install pandas`.​  
* **statsmodels:** Provides tools for performing statistical modeling and time series analysis. Install using `pip install statsmodels`.​  
* **requests:** Facilitates HTTP requests to public APIs for data retrieval. Install with `pip install requests`.​  
* **matplotlib:** For visualizing time series data, install with `pip install matplotlib`.

### **OpenRefine**

**Title**: Real-time Bitcoin Price Analysis with OpenRefine

**Difficulty**: 2 (medium)

**Description**  
OpenRefine is a powerful data manipulation tool designed to clean and transform large datasets efficiently. Originally known as Google Refine, it provides a simple yet robust platform for data wrangling activities such as cleaning messy data, transforming data from one format to another, and extending data sets with web services. OpenRefine allows users to explore huge data sets with ease by providing features such as faceted browsing, clustering of data to identify patterns, and the ability to script custom transformations.

**Describe technology**

- OpenRefine is a desktop application that runs on Java and offers a user-friendly web interface.  
- It handles larger datasets with the potential for real-time data integration through its API capabilities.  
- Key functionalities include data cleaning, data transformation, reconciliation with external datasets, and data exploration.  
- For our project, we will utilize OpenRefine's capabilities to handle and clean real-time bitcoin price data, making it ready for time series analysis.  
- OpenRefine allows for the automation of tasks such as data fetching, cleaning, and transformation through its scripting capabilities.

**Describe the project**

- The objective of this project is to ingest real-time Bitcoin price data provided by a public API (e.g., CoinGecko) and prepare it for time series analysis.  
- Begin by using basic Python scripts to retrieve exactly one week's worth of Bitcoin price data at regular intervals of every 15 minutes and save it to a CSV file.  
- Import this data into OpenRefine to perform cleaning operations such as correcting inconsistencies, handling missing values, and ensuring uniform data types.  
- Use OpenRefine to transform the data where necessary (e.g., converting timestamps into a user-friendly format and aggregating prices to desired intervals like hourly or daily).  
- Leverage OpenRefine's reconciliation features to enrich the data by linking it with other datasets, if needed.  
- Export the cleaned and transformed data back into a suitable format for further analysis in Python, such as performing time series forecasting using libraries like Pandas and Matplotlib.

**Useful resources**

- [OpenRefine Documentation](https://docs.openrefine.org/)  
- [Introduction to OpenRefine](https://openrefine.org/)  
- [CoinGecko API Documentation](https://docs.coingecko.com/v3.0.1/reference/introduction)  
- [Python Pandas Documentation](https://pandas.pydata.org/docs/)  
- [Python Matplotlib Documentation](https://matplotlib.org/)

**Is it free?**  
Yes, OpenRefine is an open-source project and is freely available under a BSD license. You can download and use it without any cost.

**Python libraries / bindings**

- `requests`: Utilize this library to fetch Bitcoin price data from public APIs. Install using `pip install requests`.  
- `pandas`: Use this library for data manipulation and preparation in Python before and after using OpenRefine. Install with `pip install pandas`.  
- `matplotlib`: This library will help visualize the time series data for analysis. Install it with `pip install matplotlib`.

This project grants hands-on experience with data cleaning and transformation, data enrichment, and time series analysis using open-source technologies like OpenRefine alongside basic Python packages.

### **OpenStack Python SDK**

**Title**: Ingest Bitcoin Prices using OpenStack Python SDK

**Difficulty**: 1 (easy)

**Description**  
The OpenStack Python SDK offers a comprehensive set of tools to interact with OpenStack services using Python, facilitating operations like provisioning and managing cloud resources. For this project, students will explore how to use the OpenStack Python SDK to ingest and process real-time Bitcoin price data. Through this hands-on experience, students will learn about integrating cloud services using OpenStack and basic time series data analysis in Python.

**Describe technology**

- **OpenStack Python SDK**: A Python library that simplifies the interaction with OpenStack services. It provides object-oriented APIs to work with OpenStack clouds, covering services like compute, storage, and networking.  
- **Key Features**:  
  - Manage OpenStack cloud resources programmatically.  
  - Simplifies the integration with OpenStack services using high-level Python APIs.  
  - Facilitates automation and orchestration of cloud resources.

**Describe the project**  
In this project, students will:

2. **Set up an OpenStack Environment**: Use OpenStack to set up a simple cloud environment where they can manage resources.  
3. **Data Ingestion**: Use the OpenStack Python SDK to launch a cloud instance that runs a script to fetch real-time Bitcoin prices from a public API like CoinGecko.  
4. **Data Storage**: Store the raw Bitcoin price data in an OpenStack Object Storage service (Swift) for later analysis.  
5. **Real-time Processing**: Implement a simple time series analysis using Python to calculate and visualize trends in Bitcoin pricing data.  
6. **Data Visualization**: Use a basic plotting library like Matplotlib or Seaborn to create visualizations of the Bitcoin price trends.

**Useful resources**

- [OpenStack Python SDK Documentation](https://docs.openstack.org/openstacksdk/)  
- [CoinGecko API Documentation](https://docs.coingecko.com/v3.0.1/reference/introduction)  
- [Matplotlib](https://matplotlib.org/stable/contents.html)  
- [Python Time Series Analysis](https://www.machinelearningplus.com/time-series/time-series-analysis-python/)

**Is it free?**  
OpenStack itself is an open-source platform and can be set up on local systems without any cost. However, if students opt to use a professional cloud service or specific OpenStack distributions, they might incur costs.

**Python libraries / bindings**

- **OpenStack SDK**: The key library to interact with OpenStack resources. Install it with `pip install openstacksdk`.  
- **Requests**: A library for making HTTP requests to fetch data from APIs. Install with `pip install requests`.  
- **Pandas**: For data manipulation and analysis, especially useful for time series data. Install with `pip install pandas`.  
- **Matplotlib**: For creating static, animated, and interactive visualizations in Python. Install with `pip install matplotlib`.

### **Papermill**

**Title**: Stream Processing of Bitcoin Data with Papermill

**Difficulty**: 2 (medium)

**Description**:  
Papermill is an open-source tool that allows users to parameterize and execute Jupyter Notebooks. It is designed for data science workflows, enabling automated execution of notebook tasks in a flexible and scalable manner. This project aims to teach students how to use Papermill for ingesting and processing real-time Bitcoin price data, focusing on the time series analysis of price changes. Students will configure notebooks to fetch and store Bitcoin data from a public API, then use Papermill to automate the running of this workflow in regular intervals, effectively creating a real-time data processing pipeline.

**Describe technology**:

- Papermill: A tool for parameterizing and executing Jupyter Notebooks. It's often used to run batch jobs in data science projects or set up repeatable and scheduled workflows for analytics tasks.  
- Key functionalities of Papermill:  
  - Parameterization: Allows you to define parameters within your notebooks, enabling the reuse of notebook templates with varying inputs.  
  - Execution: Automates the execution of notebooks, making it possible to schedule and repeatedly run them with different parameters.  
  - Input/Output: Supports managing the input and output notebook files, making it efficient to capture and store results from execution.

**Describe the project**:

- Step 1: Configure a Jupyter Notebook to fetch Bitcoin price data from a public API, such as CoinGecko, using Python libraries like `requests` or `http.client`.  
- Step 2: Implement basic data processing in the notebook, such as cleaning the data and performing initial exploratory data analysis (EDA).  
- Step 3: Introduce time series analysis within the notebook to analyze price trend patterns, like moving averages and volatility.  
- Step 4: Use Papermill to parameterize the notebook, setting parameters to adjust API request intervals and analysis time windows.  
- Step 5: Create a script using Papermill that schedules the periodic execution of the notebook with defined parameters, thereby building a near real-time updating data analysis dashboard.  
- Step 6: Explore logging and result storage features of Papermill to facilitate data persistence and track execution results.

**Useful resources**:

- [Papermill Documentation](https://papermill.readthedocs.io/)  
- [CoinGecko API Documentation](https://docs.coingecko.com/v3.0.1/reference/introduction)  
- [Jupyter Notebooks Introduction](https://jupyter.org/documentation)  
- [Time Series Analysis in Python](https://www.analyticsvidhya.com/blog/2018/02/time-series-forecasting-methods/)

**Is it free?**  
Yes, Papermill is an open-source tool and can be used freely. However, students need access to a Python environment, and possibly Jupyter Notebooks, which are also freely available.

**Python libraries / bindings**:

- `papermill`: The primary library for executing parameterized Jupyter Notebooks. Install via `pip install papermill`.  
- `requests`: A simple HTTP library for fetching data from web APIs. Install via `pip install requests`.  
- `pandas`: Used for data manipulation and analysis within your notebooks. Install via `pip install pandas`.  
- `matplotlib` and/or `seaborn`: Libraries for data visualization to graphically represent the results of your analysis. Install via `pip install matplotlib seaborn`.

### **Petastorm**

**Title:** Batch Processing of Bitcoin Price Data with Petastorm​

**Difficulty:** 3 (Difficult)​

**Description:** 

This project involves developing a system to ingest, store, and analyze Bitcoin price data using Petastorm. Students will collect Bitcoin price data at regular intervals, store it in Parquet format, and utilize Petastorm's capabilities to efficiently process and analyze these datasets. This approach provides insights into Bitcoin price trends over time and demonstrates the integration of big data processing with machine learning workflows.​

**Technology Overview:**

* **Petastorm:** An open-source data access library developed by Uber that facilitates the use of Parquet datasets in TensorFlow, PyTorch, and other machine learning frameworks. It enables efficient reading and writing of large-scale datasets, bridging the gap between big data storage formats and machine learning applications.

**Project Description:**

**Objective:** To develop a system that ingests Bitcoin price data at regular intervals, stores it using Petastorm in the Parquet format, and performs batch processing for time series analysis and forecasting.

**Steps:**

1. **Data Ingestion:**  
   * Develop a Python script to fetch Bitcoin price data from a public API (e.g., CoinGecko) at fixed intervals (e.g., hourly).​  
   * Accumulate the data over a defined period (e.g., 24 hours) to create a batch for processing.​  
2. **Data Storage:**  
   * Define a schema for the dataset using Petastorm's `Unischema` to structure the data appropriately.​  
   * Utilize Petastorm's capabilities to write the accumulated data batches to Parquet files, ensuring efficient storage and fast retrieval.  
3. **Data Processing:**  
   * Implement a data processing pipeline in Python to read the stored Parquet files using Petastorm's `make_batch_reader`.  
   * Perform transformations and computations, such as calculating moving averages, volatility, and other relevant metrics.​  
4. **Time Series Analysis:**  
   * Use Python libraries like Pandas and Matplotlib to analyze the processed data.​  
   * Visualize time series trends, patterns, and anomalies in Bitcoin prices.​  
5. **Machine Learning Integration:**  
   * Develop predictive models using TensorFlow or PyTorch to forecast future Bitcoin price trends based on historical data.​  
   * Train and evaluate the models using the processed datasets stored in Parquet format.​

**Outcome:** By completing this project, students will gain practical experience in:​

* Handling batch data ingestion and storage using Petastorm.​  
* Integrating big data processing with machine learning workflows for predictive analytics.​

**Useful Resources:**

* [Petastorm Documentation](https://petastorm.readthedocs.io/en/latest/)​  
* [Apache Parquet Documentation](https://parquet.apache.org/docs/)​  
* [CoinGecko API Documentation](https://docs.coingecko.com/v3.0.1/reference/introduction)  
* [Pandas Documentation](https://pandas.pydata.org/docs/)​  
* [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)​  
* [TensorFlow Documentation](https://www.tensorflow.org/guide)​  
* [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)​

**Is it Free?** 

Yes, Petastorm is an open-source library and free to use. Accessing Bitcoin price data through public APIs like CoinGecko is also free, although rate limits may apply. Utilizing additional services or infrastructure (e.g., cloud storage or computing resources) may incur costs depending on the provider.​

**Python Libraries / Bindings:**

* **Petastorm:** Install with `pip install petastorm`.​  
* **Pandas:** Install with `pip install pandas`.​  
* **Matplotlib:** Install with `pip install matplotlib`.​  
* **TensorFlow:** Install with `pip install tensorflow`.​  
* **PyTorch:** Install with `pip install torch`.​

### **Petl**

**Title**: Real-Time Bitcoin Price Analysis with Petl

**Difficulty**: 3

**Description**  
This project involves setting up a real-time data ingestion and processing system for Bitcoin prices, utilizing the 'Petl' library in Python to manage and manipulate incoming datasets. Petl (Python ETL) is a lightweight library that simplifies the process of extracting, transforming, and loading data, making it an excellent choice for handling data workflows, especially in data science and big data analytics. The project’s complexity lies in the need for continuous data ingestion, time series analysis, and dynamic transformation processes, providing a comprehensive learning experience in data handling and processing.

**Describe technology**

- **Petl:**  
  - Petl is short for Python ETL (Extract, Transform, Load) and provides a suite of utilities to simplify the processes of data extraction, transformation, and loading.  
  - It offers a wide range of functions for manipulating tabular data, such as filtering, transforming, and joining tables, among many others.  
  - With its simple syntax, Petl presents an excellent starting point for managing data workflows in Python, particularly in the context of small to medium-sized data volumes.  
  - Example functionalities include reading from various data sources (e.g., CSV, Excel, databases), applying transformations (e.g., converting data types, formatting), and writing the cleaned data to desired destinations.

**Describe the project**

- **Objective:** Develop a real-time Bitcoin price ingestion and processing system using Petl and Python. The system will fetch live Bitcoin price data from a public API and perform time series analysis to monitor and report price trends.  
- **Steps:**  
  1. **Data Ingestion:** Utilize Python to fetch real-time Bitcoin price data from a public API, such as CoinGecko.  
  2. **Data Storage:** Store the incoming data temporarily using a file-based or database storage system.  
  3. **Data Transformation:** Use Petl to process the raw data, including:  
     - Filtering specific time intervals for analysis.  
     - Converting currency formats and standardizing timestamps.  
  4. **Time Series Analysis:**  
     - Perform basic time series analysis using Python libraries such as pandas or statsmodels in conjunction with Petl-transformed data.  
     - Analyze price trends and calculate metrics like moving averages and volatility.  
  5. **Visualization and Reporting:**  
     - Generate visualizations using libraries like matplotlib or seaborn to illustrate price trends over time.  
     - Create a reporting mechanism to summarize findings and potential alerts when specific market conditions are met.

**Useful resources**

1. Petl Documentation: [Petl Documentation](https://petl.readthedocs.io/en/stable/)  
2. Public APIs for Bitcoin Prices: [CoinGecko API Documentation](https://docs.coingecko.com/v3.0.1/reference/introduction)  
3. Time Series Analysis with Python: [Pandas Documentation](https://pandas.pydata.org/pandas-docs/stable/), [Statsmodels Time Series Analysis](https://www.statsmodels.org/stable/tsa.html)  
4. Python Visualization Libraries: [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/)

**Is it free?**  
Yes, Petl is an open-source library. Access to public APIs like CoinGecko and Binance is usually free, but check for any usage limitations.

**Python libraries / bindings**

- **petl:** An essential library for performing ETL operations effortlessly with tabular data in Python. Install it using `pip install petl`.  
- **requests:** For making HTTP requests to fetch data from APIs. Install using `pip install requests`.  
- **pandas:** Useful for additional data analysis operations and handling time series data. Install using `pip install pandas`.  
- **matplotlib and seaborn:** For data visualization. Install using `pip install matplotlib seaborn`.

### **Petl**

**Title**: Ingesting and Processing Bitcoin Prices using Petl  
**Difficulty**: 1 (easy)

**Description**  
Petl (Python ETL) is a lightweight Python library designed for extract, transform, and load (ETL) tasks. It provides simple tools for working with data tables, which are ideal for small-scale data tasks and educational environments. Petl focuses on ease of use and flexibility, making it perfect for handling data from various sources and performing fundamental operations such as filtering, transforming, and joining datasets.

This introductory project involves using Petl to ingest real-time Bitcoin price data from a public API, such as CoinGecko, and perform simple time series analysis. By utilizing basic Python packages alongside Petl, students will learn to manipulate data and extract meaningful insights with ease.

**Describe technology**

- **Petl**:  
  - Aimed at simple ETL tasks using data tables.  
  - Offers intuitive functions for loading data from various sources, including CSV, JSON, Excel, and more.  
  - Facilitates straightforward data transformations, such as filtering, sorting, and aggregating.  
- **Core functionalities include**:  
  - Loading data with `fromcsv()`, `fromjson()`, etc.  
  - Transforming data using functions like `select()`, `cut()`, `convert()`.  
  - Storing or outputting data back to files or other formats.

**Describe the project**

- Create a Python script that uses Petl to fetch real-time Bitcoin price data from a public API.  
- Load the API data into a Petl table.  
- Transform the data to focus on key metrics such as price, time, and market capitalization.  
- Implement time series analysis to calculate simple moving averages or other metrics over specified time windows.  
- Optionally, visualize the results using a basic Python plotting library, like Matplotlib, to provide insights into Bitcoin's price trends.

**Useful resources**

- [Petl Documentation](https://petl.readthedocs.io/en/stable/)  
- [CoinGecko API Documentation](https://docs.coingecko.com/v3.0.1/reference/introduction)  
- [Python Requests Library Documentation](https://docs.python-requests.org/en/master/)

**Is it free?**  
Yes, Petl is an open-source library available to use for free. Access to the CoinGecko API is also free with no authentication required for basic usage.

**Python libraries / bindings**

- **Petl**: Essential for ETL tasks, installed via `pip install petl`.  
- **Requests**: Used to make HTTP requests to the Bitcoin price API, installed via `pip install requests`.  
- **Matplotlib**: Optional, for plotting time series data, installed via `pip install matplotlib`.

This project will guide students through the fundamental concepts of ETL processes with a focus on Python's Petl library, preparing them for more complex data manipulation tasks.

### **Pika**

**Title**: Real-Time Bitcoin Data Ingestion with Pika

**Difficulty**: 1 (easy)

**Description**:  
Pika is a robust Python client library for interacting with RabbitMQ, a widely-used message broker that facilitates efficient message queuing and handling across distributed systems. Pika provides a straightforward, user-friendly interface to connect, publish, and consume messages within RabbitMQ, making it an excellent tool for implementing real-time data ingestion and processing systems. In this project, students will explore Pika's basic functionalities, leading to the development of a time series analysis on Bitcoin price data.

**Describe technology**:

- **Core Concept**: Pika is a pure Python implementation for connecting with RabbitMQ, which supports advanced message queuing protocol (AMQP).  
- **Basic Operations**: Students will learn how to establish a channel, declare a queue, and perform basic publishing and consuming of messages.  
- **Use Cases**: Primarily used in applications that require asynchronous message processing, load balancing, and implementing task queues.  
- **Example**: Students will explore a Pika-based example to publish and consume simple text messages using a RabbitMQ instance.

**Describe the project**:

- **Objective**: Develop a basic real-time data ingestion system using Pika to handle Bitcoin price data.  
- **Step 1**: Set up RabbitMQ locally or use a cloud-based RabbitMQ service.  
- **Step 2**: Create a producer script in Python using Pika to fetch Bitcoin price data from a public API such as CoinGecko and publish it to a RabbitMQ queue at regular intervals.  
- **Step 3**: Develop a consumer script that retrieves data from the queue and stores it in a local database using a simple Python database library like SQLite.  
- **Step 4**: Implement basic time series analysis, such as time plotting or moving average calculations, with the ingested data using libraries like Matplotlib and NumPy.  
- **Learning Outcome**: Understand the foundations of working with message brokers and real-time data systems, and gain practical experience in processing financial data.

**Useful resources**:

- [Pika GitHub Repository](https://github.com/pika/pika)  
- [RabbitMQ Official Website](https://www.rabbitmq.com/)  
- [RabbitMQ Tutorials](https://www.rabbitmq.com/getstarted.html)  
- [CoinGecko API Documentation](https://docs.coingecko.com/v3.0.1/reference/introduction)

**Is it free?**  
Yes, Pika is an open-source library, and RabbitMQ is available as open-source software. A local RabbitMQ setup can be done for free, while a cloud-based RabbitMQ service may involve costs depending on the provider.

**Python libraries / bindings**:

- **Pika**: Python client library for RabbitMQ, used for establishing connection and message queue handling. Install using `pip install pika`.  
- **SQLite**: Python built-in library for lightweight database management.  
- **Matplotlib**: Visualization library for plotting time series data, installable via `pip install matplotlib`.  
- **NumPy**: Essential library for numerical computations, installable via `pip install numpy`.

### **Plotly**

**Title**: Real-Time Bitcoin Price Analysis using Plotly

**Difficulty**: 3 (difficult)

**Description**:  
In this project, you will implement a real-time data visualization system for Bitcoin prices using Plotly in Python. Your task is to fetch data from a public API, process it, and visualize it to perform time series analysis. Plotly is an interactive graphing library for Python that makes generating visually appealing and informative plots straightforward. Students will gain experience in building advanced interactive plots that provide insights into Bitcoin price trends and fluctuations over time.

**Describe technology**:  
Plotly is a versatile graphing library that supports interactive plotting and a wide range of plot types, including 3D plots, maps, and scientific types. It provides a simple, yet powerful interface that integrates seamlessly with Jupyter notebooks and other development environments. Plotly supports various languages, but for this project, we will focus on Plotly for Python, which allows users to create both static and interactive plots conveniently.

**Describe the project**:

- Implement a system that fetches real-time Bitcoin price data from a public API such as CoinGecko or Binance at regular intervals.  
- Use basic Python packages such as Requests for HTTP requests and Pandas for data manipulation and structuring.  
- Process and clean the data to handle missing values, anomalies, or duplicate entries.  
- Employ Plotly functionalities to visualize the Bitcoin price data. Create various time series plots, including line charts, candlestick charts, and area plots, to display different aspects of Bitcoin price variation over time.  
- Implement interactive features, such as zooming, panning, and updating graphs in real-time when new data is fetched.  
- Use additional Plotly features, like annotations and trend lines, to perform and highlight time series analysis on the data.

**Useful resources**:

- [Plotly Documentation](https://plotly.com/python/): Comprehensive guide and examples for various types of plots and plots customization.  
- [Requests Documentation](https://requests.readthedocs.io/en/master/): Reference for using Requests package to make HTTP requests.  
- [Pandas Documentation](https://pandas.pydata.org/docs/): Guide on data manipulation, processing, and analysis using Pandas.  
- [CoinGecko API Documentation](https://docs.coingecko.com/v3.0.1/reference/introduction): Public API to access real-time Bitcoin prices.

**Is it free?**  
Plotly offers both free and commercial versions. The open-source version is entirely free and sufficient for the project's requirements.

**Python libraries / bindings**:

- **Plotly**: Install using `pip install plotly`. Essential for creating interactive plots.  
- **Requests**: Install using `pip install requests`. Needed to fetch real-time data from APIs.  
- **Pandas**: Install using `pip install pandas`. Important for data manipulation and structuring into a suitable format for visualization.

### **Plotly \#2**

**Title**: Real-Time Bitcoin Blockchain Metrics Visualization and Time Series Analysis with Plotly  
**Difficulty**: 3 (difficult)

**Description**  
This project focuses on visualizing and analyzing Bitcoin blockchain metrics (e.g., transaction volume, block size, hash rate) in real-time using Plotly. Instead of Dash, you will use Plotly’s native interactive plotting capabilities to build a self-updating Jupyter Notebook or standalone HTML dashboard. The goal is to ingest live blockchain data, process it, and generate time series visualizations with statistical analysis.

**Describe technology**

- **Plotly**: A library for creating interactive, publication-quality graphs. It supports animations, subplots, and dynamic updates without Dash.  
- **Key Features**: Use Plotly’s `FigureWidget` for live updates in Jupyter Notebooks, or auto-refreshing HTML files with `plotly.offline`.

**Describe the project**

1. **Data Ingestion**:  
     
   - Fetch Bitcoin blockchain metrics (e.g., transaction count, hash rate) from APIs like Blockchain.com or Mempool.space.  
   - Use `requests` for REST API calls or `websockets` for streaming data.

   

2. **Data Processing**:  
     
   - Clean data with Pandas (handle missing values, resample time series).  
   - Compute rolling averages, transaction rates, and anomaly scores (Z-scores).  
   - Decompose time series into trend/seasonality/residuals using `statsmodels`.

   

3. **Visualization & Analysis**:  
     
   - **Real-Time Plots**: Use Plotly’s `FigureWidget` to create auto-updating line charts for live metrics.  
   - **Statistical Charts**: Build subplots showing decomposed time series, histograms, and correlation heatmaps.  
   - **Anomaly Highlighting**: Add annotations to flag unusual events (e.g., spikes in block size).  
   - Export visualizations as standalone HTML files with `plotly.offline.plot()`.

   

4. **Auto-Refresh Workaround**:  
     
   - Schedule periodic data fetches and plot updates using Python’s `time.sleep()` or `threading.Timer`.  
   - For Jupyter: Use `FigureWidget`’s `add_trace()` and `relayout()` to refresh plots.

**Useful resources**

- Plotly Time Series Tutorial: [https://plotly.com/python/time-series/](https://plotly.com/python/time-series/)  
- Blockchain.com API Docs: [https://www.blockchain.com/api](https://www.blockchain.com/api)  
- Statsmodels Decomposition Guide: [https://www.statsmodels.org/stable/examples/notebooks/generated/tsa\_decompose.html](https://www.statsmodels.org/stable/examples/notebooks/generated/tsa_decompose.html)

**Is it free?**  
Yes: Plotly’s open-source library and Blockchain.com’s API are free for non-commercial use.

**Python libraries / bindings**

- `plotly` (install: `pip install plotly`)  
- `pandas`, `numpy` (data processing)  
- `requests`, `websockets` (data ingestion)  
- `statsmodels` (time series decomposition)

### **Polars**

**Title**: Analyze Bitcoin Prices with Polars in Real-Time

**Difficulty**: 1 (easy)

**Description**  
Polars is a fast DataFrame library implemented in Rust for Python, known for its speed and ability to handle large datasets efficiently. This project will involve using Polars to ingest real-time Bitcoin price data and perform basic time series analysis. Students will gain experience with Polars' core functionalities, including DataFrame manipulation, querying, and aggregation. The goal is to demonstrate how to use Polars for efficient data processing, especially for time-sensitive financial data.

**Describe technology**

- **Polars** is a DataFrame library designed for high-performance, parallel processing of data.  
- Built using Rust but offers Python bindings, ensuring both speed and ease of use.  
- Supports both eager and lazy evaluation, providing flexibility for various data processing needs.  
- Efficiently handles large datasets and complex operations like grouping, joining, and aggregating with minimal memory usage.  
- Known to outperform traditional pandas for large datasets due to its parallel processing capabilities.

**Describe the project**

- **Objective**: Use Polars to ingest real-time Bitcoin price data from an API (e.g., CoinGecko).  
- **Step 1**: Set up a function to fetch Bitcoin price data from the API every few minutes.  
- **Step 2**: Use Polars to create a DataFrame and store the fetched data.  
- **Step 3**: Implement basic time series analysis methods such as calculating moving averages and price change over time.  
- **Step 4**: Visualize the insights using simple plotting libraries like Matplotlib.  
- **Outcome**: Students will understand how to ingest and process real-time data effectively using Polars and gain insights into the dynamic nature of Bitcoin price movements.

**Useful resources**

- [Polars Official Documentation](https://docs.pola.rs)  
- [CoinGecko API Documentation](https://docs.coingecko.com/v3.0.1/reference/introduction)  
- [Polars GitHub Repository](https://github.com/pola-rs/polars)

**Is it free?**  
Yes, Polars is an open-source library and free to use. Access to the CoinGecko API is also free for basic usage.

**Python libraries / bindings**

- **Polars**: The main Python library for this project, used for data manipulation and analysis.  
- **Requests**: To seamlessly retrieve data from APIs.  
- **Matplotlib**: For plotting and visualizing the analyzed Bitcoin price data.  
- **Datetime**: Useful for handling dates and times, essential in time series analysis.

### **Postmark**

**Title**: Ingest Bitcoin Prices Using Postmark

**Difficulty**: 1 (easy)

**Description**:  
Postmark is primarily known for its email delivery services, but for this project, we'll delve into its less-discussed functionality of real-time data processing. Although not typically used for data ingestion, students will learn how Postmark's webhook support can be adapted to ingest real-time Bitcoin price data into a Python-based system.

The project involves setting up a simple webhook server using Python that interacts with the Postmark API. By leveraging Postmark's capability to push real-time updates, students will integrate it with a public Bitcoin price API, such as CoinGecko, to acquire and process current price data.

**Describe technology**:

- **Postmark**: Primarily an email delivery service, it provides webhook features that can be repurposed for real-time data ingestion.  
- **Webhooks**: Postmark's webhooks send HTTP requests to a user-defined URL endpoint, which in this project will read and process Bitcoin price data.  
- **Python**: Basic Python packages will be used to handle HTTP requests and data manipulation.

**Describe the project**:

- **Phase 1**: Familiarize with Postmark's webhook setup. Register an account and read the documentation on creating webhooks.  
- **Phase 2**: Create a basic Python server to handle incoming HTTP POST requests from a public Bitcoin price API.  
- **Phase 3**: Configure Postmark to send notifications via a webhook upon significant changes in Bitcoin prices.  
- **Phase 4**: Implement time series analysis on the ingested data. Use Python's basic data processing libraries like Pandas and Matplotlib to visualize trends and perform elementary analysis such as moving averages.  
- **Phase 5**: Configure Postmark to send notifications via a webhook upon analyzed data.

**Useful resources**:

- [Postmark Documentation](https://postmarkapp.com/developer)  
- [Python Requests Library](https://docs.python-requests.org/en/master/)  
- [CoinGecko API Documentation](https://docs.coingecko.com/v3.0.1/reference/introduction)  
- [Pandas Documentation](https://pandas.pydata.org/pandas-docs/stable/)  
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)

**Is it free?**  
Postmark offers a free trial with limited usage for exploring its features, including webhooks. Additional requests may incur fees.

**Python libraries / bindings**:

- `requests`: To handle HTTP requests and ingest data from APIs. Installable via `pip install requests`.  
- `Flask` or `Django`: To create a simple server to receive webhook data. Installable via `pip install Flask` or `pip install Django`.  
- `Pandas`: For processing and analyzing time series data. Installable via `pip install pandas`.  
- `Matplotlib`: For visualizing the price trends and analysis results. Installable via `pip install matplotlib`.

This project provides a practical introduction to using Postmark's webhook capability and simple Python libraries to create a system for ingesting and analyzing real-time data.

### **Prefect**

**Title**: Real-Time Bitcoin Price Analysis using Prefect

**Difficulty**: 3 (difficult)

**Description**  
This project involves using Prefect, a modern workflow orchestration tool, to ingest, process, and analyze real-time Bitcoin price data. Prefect’s core features revolve around the design, monitoring, and scheduling of data workflow tasks. In this project, students will learn how to harness Prefect to implement a dynamic ETL (Extract, Transform, Load) pipeline for Bitcoin data and perform time series analysis. The project simulates a real-world scenario where continuous data ingestion, data quality checks, and advanced data processing are integral for generating actionable insights from financial data.

**Describe technology**

- **Prefect**: Prefect is a workflow management system that allows developers to build and deploy data pipelines with ease and flexibility. It is designed to improve data reliability by providing visibility over data workflows and allowing for complex dependency management. Prefect includes features like dynamic scheduling, ad-hoc parameterized runs, error notifications, and flow visualization.  
- **Core Concepts**:  
  - **Flow**: The top-level object representing your entire Prefect process, which consists of tasks.  
  - **Task**: The building block of a workflow that performs a single unit of work.  
  - **Executor**: Executes tasks within a flow, allowing for parallel and distributed execution.  
- **Example Use Cases**:  
  - Defining and scheduling tasks for executing Python functions.  
  - Triggering conditional workflows based on task outputs.  
  - Handling retries and rollbacks for failed tasks.

**Describe the project**

1. **Ingest Real-Time Bitcoin Data**:  
   - Utilize a Bitcoin API (e.g., CoinGecko) to fetch live Bitcoin price data.  
   - Set up a Prefect flow to poll this API every few minutes.  
2. **Data Collection**:  
   - Use Prefect tasks to systematically extract, transform, and load Bitcoin data into a database (e.g., PostgreSQL).  
   - Implement data validation checks within Prefect to ensure data quality.  
3. **Data Processing and Analysis**:  
   - Transform the raw data into a format suitable for analysis using basic Python packages such as pandas.  
   - Implement Prefect tasks for time series analysis, such as calculating moving averages or volatility.  
4. **Monitoring and Alerts**:  
   - Create Prefect sensors to monitor significant price movements and notify stakeholders via email or Slack.  
5. **Visualization**:  
   - Integrate with visualization tools (e.g., Matplotlib or Plotly) to graph Bitcoin price trends and analysis results.  
6. **Testing and Debugging**:  
   - Utilize Prefect's debugging features to test and troubleshoot the workflows.

**Useful resources**

- [Prefect Documentation](https://docs.prefect.io/)  
- [CoinGecko API Documentation](https://docs.coingecko.com/v3.0.1/reference/introduction)  
- [Pandas Documentation](https://pandas.pydata.org/docs/)  
- [Plotly Documentation](https://plotly.com/python/)

**Is it free?**

- Prefect Core is open-source and free to use. Prefect Cloud, which provides additional capabilities (e.g., a hosted dashboard), may incur costs.

**Python libraries / bindings**

- **Prefect**: `pip install prefect` \- For designing and executing workflows.  
- **Requests**: `pip install requests` \- For making requests to the Bitcoin API.  
- **Pandas**: `pip install pandas` \- For data manipulation and time series analysis.  
- **SQLAlchemy**: `pip install sqlalchemy` \- For interfacing with databases.  
- **Matplotlib/Plotly**: `pip install matplotlib plotly` \- For data visualization.

### **Presto**

**Title:** Real-Time Bitcoin Data Analysis with Presto​

**Difficulty:** 2 (Medium)​

**Description:** 

In this project, students will utilize Presto, an open-source distributed SQL query engine, to analyze real-time Bitcoin transaction data. The project involves setting up a data pipeline that ingests live Bitcoin transaction data, stores it in a queryable format, and uses Presto to perform interactive analyses. This project offers hands-on experience with big data processing, SQL querying, and real-time data analytics.​

**Describe Technology:** 

Presto is a high-performance, distributed SQL query engine designed for large-scale data analytics. Originally developed by Facebook, it allows users to run interactive analytic queries against data sources of all sizes. Presto supports querying data from multiple sources, including Hadoop, S3, Cassandra, and traditional relational databases, enabling federated queries across various data systems.

**Describe the Project:**

**Objective:** To set up a system that ingests real-time Bitcoin transaction data and utilizes Presto to perform interactive SQL queries for data analysis.​

**Steps:**

1. **Data Ingestion:**  
   * Set up a data ingestion pipeline to collect real-time Bitcoin transaction data. This can be achieved by connecting to public APIs or using services that provide live Bitcoin transaction streams such as CoinGecko.​  
2. **Data Storage:**  
   * Store the ingested data in a format compatible with Presto, such as JSON or Parquet files, in a distributed storage system like HDFS or AWS S3.​  
3. **Presto Setup:**  
   * Install and configure Presto on a local or cloud-based server. Ensure that Presto is connected to the data storage system where the Bitcoin data resides.​  
4. **Data Analysis:**  
   * Use Presto to run SQL queries on the Bitcoin transaction data. Perform analyses such as transaction volume over time, average transaction value, and identifying the most active addresses.  
5. **Visualization:**  
   * Integrate Presto with a visualization tool or use Presto's output to create visual representations of the analysis results, such as graphs and charts, to identify trends and patterns in Bitcoin transactions.​

**Useful Resources:**

* [Presto Documentation](https://prestodb.io/docs/current/)​  
* [CoinGecko API Documentation](https://docs.coingecko.com/v3.0.1/reference/introduction)  
* [Apache Parquet Documentation](https://parquet.apache.org/docs/)​

**Is it Free?** 

Yes, Presto is an open-source project and free to use. However, depending on the data ingestion method and storage solutions chosen, there may be associated costs. For example, storing data in AWS S3 incurs storage costs. It's advisable to use free tiers or local storage solutions for educational purposes to minimize costs.​

**Python Libraries / Bindings:**

* **Requests:** To make HTTP requests for fetching Bitcoin transaction data from APIs.​  
* **Pandas:** For data manipulation and analysis.​  
* **SQLAlchemy:** To facilitate interaction between Python and Presto.​  
* **Matplotlib or Plotly:** For data visualization.​

### **Prophet**

**Title:** Time Series Forecasting of Bitcoin Prices using Prophet

**Difficulty:** Medium (2)

**Description**  
Prophet is an open-source forecasting tool developed by Facebook (now Meta) designed to handle large-scale time series data with ease. It provides straightforward APIs for accurate and fast time series forecasting, particularly when dealing with data exhibiting weekly and yearly seasonality, holidays, or missing data. The core strength of Prophet is its ability to fit complex nonlinear trends with daily seasonality over various periods.

In this project, students will explore Prophet's main functionalities and their application to real-time Bitcoin price data. The use of basic Python packages alongside Prophet will be a requirement, enabling students to solidify their understanding of time series forecasting, data ingestion, and initial data processing for trend analysis.

**Describe technology**

- **Prophet**: Developed by Facebook (Meta), Prophet is a procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects if needed.  
- **Key features**: It is particularly robust to missing data and shifts in the trend, and typically handles outliers well. Its straightforward procedure allows for quick iterations, which is vital for students learning dynamic forecasting methods.  
- **Example functionalities**:  
  - Import and structure data for time series analysis.  
  - Fit a model using historical data and make forecasts.  
  - Plot the results, including components such as trend, weekly, and yearly seasonality.

**Describe the project**

- **Objective**: Implement a real-time data processing pipeline using Prophet to forecast Bitcoin prices.  
- **Steps**:  
  1. **Data Collection**: Use a public API (e.g., CoinGecko or Binance) to ingest real-time Bitcoin price data into a local environment. This involves setting up a script to periodically fetch data and store it in a CSV format.  
  2. **Data Preprocessing**: Use basic Python libraries (Pandas) to clean and preprocess the data. Ensure the data is formatted correctly for time series analysis with Prophet. Handle missing timestamps, any anomalies, or outliers effectively to maintain the quality of input data.  
  3. **Apply Prophet**:  
     - Load the preprocessed data into the Prophet model.  
     - Fit the model to the historical data to understand the underlying patterns and trends.  
     - Forecast future Bitcoin prices for a specified period.  
  4. **Visualization and Analysis**:  
     - Plot the forecasts and evaluate the model's performance.  
     - Analyze the impact of seasonal components and provide insights derived from the forecast results.

**Useful resources**

- [Prophet Official Documentation](https://facebook.github.io/prophet/)  
- [CoinGecko API Documentation](https://docs.coingecko.com/v3.0.1/reference/introduction)  
- [Pandas Documentation](https://pandas.pydata.org/docs/)

**Is it free?**  
Yes, Prophet is open-source and free to use. However, API access might be subject to limits or require subscriptions for higher usage levels. It's essential to verify the terms of use for the API chosen for data collection.

**Python libraries / bindings**

- **Prophet**: Install via pip using `pip install prophet`. This library will be used for building and evaluating the forecasting model.  
- **Pandas**: For data manipulation and preprocessing. Install via pip using `pip install pandas`.  
- **Requests**: For fetching data from public APIs. Install via pip using `pip install requests`.  
- **Matplotlib/Seaborn**: For plotting and visualizing the forecasted results. Install via pip using `pip install matplotlib seaborn`.

By the end of this project, students will gain valuable experience in handling time series data with real-world applications, enriching their understanding of data science models specifically geared for forecasting.

### **Protocol Buffers (protobuf)**

**Title**: Real-time Bitcoin Price Analysis with Protocol Buffers

**Difficulty**: 3 (difficult)

**Description**:  
Protocol Buffers (protobuf) is a language-neutral, platform-neutral, extensible mechanism for serializing structured data, developed by Google. It is mainly used to facilitate data communication across different systems by defining data structures in a protocol buffer file, compiled into a form that can be used by your application code to read and write protocol messages. The efficiency of protobuf in serializing structured data makes it an ideal tool for handling large-scale data communication in a big data system. This project involves the use of protobuf in conjunction with basic Python packages to ingest and process real-time Bitcoin price data, leveraging its serialization capabilities to manage the data throughput and efficient storage.

**Describe Technology**:

- **Serialization Efficiency**: Protobuf offers compact binary serialization, enabling efficient communication and storage.  
- **Cross-language Compatibility**: Protobuf files are language-neutral and can be shared between systems developed in different languages.  
- **Versioning and Extensibility**: Protobuf supports message evolution by allowing fields to be deprecated or added without affecting existing deployments.

**Describe the Project**:

1. **Objective**: Implement a system that ingests real-time Bitcoin price data, processes it using Protocol Buffers, and performs time series analysis to detect trends and anomalies.  
     
2. **Steps**:  
     
   - Define a protobuf schema to describe the Bitcoin price data structure (e.g., timestamp, price, volume).  
   - Use Python to create a real-time data pipeline that fetches Bitcoin prices from a public API like CoinGecko.  
   - Serialize the incoming data using Protocol Buffers for efficient storage and transfer.  
   - Store the serialized data in a suitable storage system (e.g., a file system or a database).  
   - Deserialize the data and perform time series analysis to derive insights such as price trends and anomalies using libraries like `pandas` and `statsmodels`.  
   - Visualize the analysis results using libraries like `matplotlib`.

   

3. **Outcome**: Students will gain hands-on experience in managing real-time data pipelines, understand the efficiency of protobuf in handling large-scale serialization, and apply analytical techniques to derive meaningful business insights.

**Useful Resources**:

- [Protocol Buffers Documentation](https://developers.google.com/protocol-buffers)  
- [CoinGecko API Documentation](https://docs.coingecko.com/v3.0.1/reference/introduction)  
- [Python Protobuf Library on PyPI](https://pypi.org/project/protobuf/)

**Is it free?**  
Yes, Protocol Buffers is open-source and can be freely used. The CoinGecko API provides free access to cryptocurrency pricing data.

**Python Libraries / Bindings**:

- `protobuf`: Install via `pip install protobuf` to compile and manage protocol buffer files within Python.  
- `requests`: A simple library for making HTTP requests, useful for fetching Bitcoin data from APIs.  
- `pandas`: For data manipulation and time series analysis.  
- `matplotlib`: For data visualization.  
- `statsmodels`: For advanced statistical analysis and time series analysis in Python. Install using `pip install statsmodels`.

### **Py2neo**

**Title:** Modeling Bitcoin Transaction Networks with Py2neo​

**Difficulty:** 3 (Difficult)​

**Description:** This project involves using Py2neo, a comprehensive Python library for interacting with the Neo4j graph database, to model and analyze Bitcoin transaction networks. Students will construct a system that ingests Bitcoin transaction data, represents it as a graph in Neo4j, and performs analyses to uncover patterns and insights within the transaction network. This project provides hands-on experience with graph databases, network analysis, and the application of graph theory to real-world financial data.​

**Describe Technology:**

* **Py2neo:** A user-friendly Python library that facilitates interaction with Neo4j databases. It allows for the execution of Cypher queries, management of database connections, and seamless integration of graph data into Python applications.​  
* **Neo4j:** A leading graph database management system designed for handling highly connected data. It efficiently stores and queries data structured as nodes and relationships, making it ideal for applications like social networks, recommendation systems, and financial transaction networks.​

**Describe the Project:**

**Objective:** To develop a system that models Bitcoin transactions as a graph using Py2neo and Neo4j, enabling analysis of the transaction network to identify patterns, detect anomalies, and gain insights into the flow of Bitcoin.​

**Steps:**

1. **Data Ingestion:**  
   * Develop a Python script to fetch Bitcoin transaction data from a public API or dataset. Each transaction should include details such as sender and receiver addresses, transaction amount, and timestamp.​  
2. **Graph Modeling:**  
   * Design a graph schema where each node represents a unique Bitcoin address, and each directed relationship (edge) represents a transaction from one address to another. Include properties on nodes and relationships to capture relevant transaction details.​  
3. **Data Insertion:**  
   * Utilize Py2neo to insert the transaction data into the Neo4j database, creating nodes for addresses and relationships for transactions. Ensure that duplicate nodes are not created for the same address by implementing appropriate checks or constraints.​  
4. **Network Analysis:**  
   * Perform analyses on the transaction network using Cypher queries and Py2neo. Examples include identifying the most active addresses, detecting clusters of addresses with frequent transactions among them, and finding patterns indicative of fraudulent activity.​  
5. **Visualization:**  
   * Optionally, use graph visualization tools compatible with Neo4j to visualize the transaction network, highlighting key nodes and relationships to illustrate findings from the analysis.​

**Useful Resources:**

* [Py2neo Documentation](https://neo4j-contrib.github.io/py2neo/)​  
* [Neo4j Official Site](https://neo4j.com/)​  
* [CoinGecko API Documentation](https://docs.coingecko.com/v3.0.1/reference/introduction)  
* [Neo4j Cypher Manual](https://neo4j.com/docs/cypher-manual/current/)​  
* [Graph Algorithms in Neo4j](https://neo4j.com/developer/graph-data-science/)​

**Is it Free?**

Neo4j offers a free Community Edition suitable for this project. Py2neo is an open-source library and free to use. Accessing Bitcoin transaction data may be free depending on the chosen data source; it's advisable to verify any usage limitations or costs associated with the data provider.​

**Python Libraries / Bindings:**

* **Py2neo:** Install via `pip install py2neo`. Essential for interacting with the Neo4j database.​  
* **Requests:** Install via `pip install requests`. Used for making HTTP requests to fetch Bitcoin transaction data from APIs.​  
* **Pandas:** Install via `pip install pandas`. Useful for initial data manipulation and preparation before inserting into Neo4j.​  
* **Matplotlib or Plotly:** Install via `pip install matplotlib` or `pip install plotly`. These libraries can be used for visualizing analysis results.​

This project aligns with Py2neo's capabilities by focusing on modeling and analyzing a Bitcoin transaction network within a graph database context, leveraging the strengths of Neo4j and Py2neo for handling and querying connected data.

### **PyArrow**

**Title**: Real-Time Bitcoin Data Processing with PyArrow

**Difficulty**: 3

**Description**  
PyArrow is an actively developed library providing a high-performance, cross-language solution to managing large data sets efficiently using Arrow memory format and zero-copy reads. It allows for seamless integration across data science tools and efficient data processing, storage, and interchange capabilities. This project entails utilizing PyArrow's functionalities to ingest real-time Bitcoin price data, transforming the data into a feasible structure for complex time series analysis, and gaining valuable insights from it.

**Describe technology**

- PyArrow: An open-source, cross-language development platform for in-memory data.  
- Focused on high performance and productivity with support for zero-copy reads for Arrow-optimized systems.  
- Offers an interface to read and write Arrow data from various sources.  
- Supports efficient conversion to and from popular formats like Parquet, which is ideal for big data system analytics.

**Describe the project**

- **Objective**: Ingest and process Bitcoin price data effectively using PyArrow for real-time analysis.  
    
- **Phase 1**: Fetch Bitcoin price data continuously using the CoinGecko API and PyArrow.  
    
  - Write scripts to collect Bitcoin price data in real-time and store it using PyArrow's memory handling.


- **Phase 2**: Structure the data using PyArrow tables, enabling efficient storage and access by transforming the raw streamed data into Arrow batches.  
    
  - Use PyArrow's conversion capabilities to format the live data into Arrow tables, optimizing memory usage.


- **Phase 3**: Perform time series analysis to derive insights from the processed data.  
    
  - Develop analytical scripts to manipulate the Arrow tables for calculating metrics like moving average, volatility, and anomaly detection.


- **Phase 4**: Store the structured data in Parquet format for efficient querying.  
    
  - Use PyArrow to write the data in Parquet files, enabling fast reading and future analytics with reduced computational costs.

**Useful resources**

- [PyArrow Official Documentation](https://arrow.apache.org/docs/python/)  
- [Apache Arrow Project](https://arrow.apache.org/)  
- [CoinGecko API Documentation](https://docs.coingecko.com/v3.0.1/reference/introduction)  
- [Parquet Format Documentation](https://parquet.apache.org/docs/)

**Is it free?**  
Yes, PyArrow is part of the Apache Arrow project, which is licensed under the Apache License 2.0, and the CoinGecko API offers free access with rate limits.

**Python libraries / bindings**

- PyArrow: The primary library for managing Arrow data structures and I/O operations (`pip install pyarrow`).  
- Pandas: Useful for auxiliary data manipulation and conversion operations involving PyArrow (`pip install pandas`).  
- Requests or HTTP libraries: For fetching real-time data from the CoinGecko API (`pip install requests`).

This project guides you through advanced aspects of processing and structuring real-time data within the Arrow ecosystem. It provides an in-depth experience in managing, transforming, and deriving insights from large datasets.

### **PyCaret**

**Title**: Real-Time Bitcoin Data Analysis with PyCaret

**Difficulty**: Medium (2=medium difficulty)

**Description**  
This project focuses on using PyCaret, an open-source, low-code machine learning library in Python, to perform real-time data ingestion and time series analysis of Bitcoin price data. The goal is to gain hands-on experience with PyCaret's capabilities in building and deploying machine learning models efficiently, emphasizing its application in time series forecasting.

**Describe technology**

- **PyCaret**: PyCaret simplifies machine learning tasks, including data preparation, model training, and deployment. It supports a wide range of models through a unified API and is especially valuable for quickly prototyping and testing different algorithms.  
- PyCaret provides modules tailored for various machine learning types, including classification, regression, clustering, anomaly detection, natural language processing, and time series analysis.

**Describe the project**

- **Data Ingestion**: Utilize Python packages such as `requests` to connect to a public Bitcoin price API (e.g., CoinGecko, CoinMarketCap) to fetch real-time Bitcoin data. Use `pandas` to organize the fetched data.  
- **Data Processing**: Cleanse and preprocess the acquired data using `pandas` to prepare it for analysis. This involves handling missing values, formatting timestamps, and normalizing data.  
- **Time Series Analysis with PyCaret**:  
  - Leverage PyCaret's time series module to explore different forecasting techniques.  
  - Focus on developing a model that predicts future Bitcoin prices based on real-time data.  
  - Measure the performance of various models and select the optimal one based on evaluation metrics.  
- **Deployment and Visualization**: Implement a visualization step using libraries such as `matplotlib` or `plotly` to visualize historical and predicted data. Optionally, prepare a simple dashboard showing the real-time updates of the Bitcoin price and model predictions.  
- **Results Analysis**: Document the findings, challenges faced, and insights gained from the time series analysis process.

**Useful resources**

- [PyCaret Official Documentation](https://pycaret.gitbook.io/docs)  
- [PyCaret Time Series Module](https://pycaret.gitbook.io/docs/get-started/modules)  
- [CoinGecko API Documentation](https://docs.coingecko.com/v3.0.1/reference/introduction)

**Is it free?**  
Yes, PyCaret is open-source and free to use. Access to some Bitcoin APIs may have free usage tiers or require registration.

**Python libraries / bindings**

- **PyCaret**: The primary library for implementing machine learning workflows, specifically time series analysis for this project.  
- **pandas**: For data manipulation and handling.  
- **requests**: To establish connections with Bitcoin APIs and ingest data.  
- **matplotlib/plotly**: For creating visualizations that represent model predictions and real-time data trends.

### **PyCryptodome**

**Title**: Real-Time Bitcoin Data Analysis Using PyCryptodome

**Difficulty**: 1 (Easy)

**Description**

PyCryptodome is a self-contained Python package of low-level cryptographic primitives, designed for implementing cryptographic operations in Python. Its easy-to-use interface allows students to perform cryptographic tasks such as encryption, decryption, hashing, and authentication without delving into complex cryptographic algorithms. This makes it an excellent choice for projects requiring basic security implementations.

This project involves using PyCryptodome to securely ingest real-time Bitcoin price data from an API, showcasing its cryptographic functionalities. The project will focus on securely handling time-series Bitcoin data, ensuring data integrity through hashing, and enhancing privacy through encryption before storage.

**Describe technology**

* PyCryptodome is a library focused on cryptographic algorithms and primitives.  
* It provides functionalities for encryption/decryption, cryptographic hashes, digital signatures, and random number generation.  
* PyCryptodome is a drop-in replacement for the PyCrypto library and uses a simple API for cryptographic operations.

**Describe the project**

* Retrieve real-time Bitcoin price data from a public API like CoinGecko.  
* Use PyCryptodome to hash the incoming data to ensure integrity and detect any data tampering.  
* Encrypt the data using symmetric encryption techniques before storing it.  
* Implement a simple script to decrypt and display Bitcoin prices securely.  
* Analyze the securely stored Bitcoin data to identify trends and patterns using basic Python libraries such as Pandas for time series analysis.  
* This project not only helps students understand time series data handling but also gives them hands-on experience with basic cryptographic concepts using PyCryptodome.

**Useful resources**

* [PyCryptodome Documentation](https://pycryptodome.readthedocs.io/en/latest/)  
* [CoinGecko API Documentation](https://docs.coingecko.com/v3.0.1/reference/introduction)  
* [Pandas Documentation](https://pandas.pydata.org/docs/)

**Is it free?**

Yes, PyCryptodome is an open-source library and can be used freely. You only need access to a Python environment to run the code.

**Python libraries / bindings**

* PyCryptodome: For cryptographic functions like hashing and encryption. Install via pip:  pip install pycryptodome.  
* Requests: To fetch real-time Bitcoin data from APIs. Install via pip:  pip install requests.  
* Pandas: For data analysis and manipulation of time series data. Install via pip:  pip install pandas.

### **Pydantic**

**Title**: Real-time Bitcoin Data Processing with Pydantic

**Difficulty**: 1 (easy)

**Description**  
Pydantic is a data validation and settings management library in Python that uses Python type annotations. It assists in defining and validating data models, ensuring that data is accurate and conforms to expected formats. Pydantic is particularly useful when dealing with API data, as it allows for seamless parsing and validation of JSON objects into Python objects. This project will involve leveraging Pydantic to ingest real-time Bitcoin price data, validate its structure, and perform basic time series analysis on the data.

**Describe technology**

- Pydantic enables you to define clean and maintainable data models using Python's type hints, which simplifies data validation and error handling.  
- It automatically converts input data into specified Python types and throws validation errors if the data does not conform to the model.  
- Pydantic is lightweight and fast, making it ideal for projects involving frequent data validation, like real-time data processing.

**Describe the project**

- **Objective**: To ingest Bitcoin price data from a public API, validate it with Pydantic, and perform basic time series analysis to observe price trends.  
- **Step 1**: Set up a Python script to fetch real-time Bitcoin price data from an API like CoinGecko.  
- **Step 2**: Utilize Pydantic to define data models that represent the structure of the incoming JSON data. Validate the data against these models to ensure accuracy and consistency.  
- **Step 3**: Implement a basic time series analysis using Python libraries like pandas. Analyze trends, such as average price changes over specific time intervals.  
- **Step 4**: Output the analysis results to the console or a simple visualization using a library like matplotlib.

**Useful resources**

- [Pydantic Documentation](https://pydantic-docs.helpmanual.io/)  
- [CoinGecko API Documentation](https://docs.coingecko.com/v3.0.1/reference/introduction)  
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)  
- [Pandas Documentation](https://pandas.pydata.org/docs/)

**Is it free?**  
Yes, both Pydantic and the suggested public API for Bitcoin data (e.g., CoinGecko) are free to use.

**Python libraries / bindings**

- **Pydantic**: The main library for data validation and management.  
- **Requests**: To fetch data from the Bitcoin API (install using `pip install requests`).  
- **Pandas**: For handling and analyzing the time series data (install using `pip install pandas`).  
- **Matplotlib**: For optional data visualization (install using `pip install matplotlib`).

### **PyDrive**

**Title**: Real-Time Bitcoin Data Ingestion with PyDrive

**Difficulty**: 1 (easy)

**Description**

This project involves implementing a simple system to ingest and process real-time Bitcoin price data using Python with a focus on utilizing PyDrive. PyDrive is a wrapper for Google Drive API that provides an interface to interact with files in Google Drive using Python. Students will explore fundamental tasks of connecting to Google Drive, uploading and downloading files, and creating folders to manage datasets.

**Describe technology**

- **PyDrive**: PyDrive is a Python library that simplifies the process of authenticating and interacting with Google Drive. It provides easy access to Google Drive with minimal setup and allows easy manipulation of files and directories in your Drive account.  
    
  * Example Basic Functionalities:  
      
    - Authenticate Google Drive account using OAuth2.  
    - List files and directories in Google Drive.  
    - Upload, download, and delete files.  
    - Create and manage folders.

    

  * Typical setup involves setting up OAuth2.0 credentials through Google Cloud Console and using them to authorize access.

**Describe the project**

- **Objective**: Students will develop a simple project to continuously fetch Bitcoin price data from a public API like CoinGecko and store it in a Google Drive folder for easy access and sharing.  
    
- **Steps Involved**:  
    
  1. Set up a Google Cloud Project and enable the Google Drive API.  
  2. Use PyDrive to authenticate and connect to Google Drive.  
  3. Create a folder in Google Drive to store the Bitcoin price data.  
  4. Write a Python script to ingest Bitcoin price data at regular intervals and save it to Google Drive as CSV files.  
  5. Use basic Python libraries like pandas for data handling and simple time-series transformations (e.g., converting timestamps, calculating price averages).


- **Outcome**: Students will gain hands-on experience in using PyDrive to manage files with Google Drive, practicing how to combine it with real-time data ingestion and basic data processing.

**Useful resources**

- [PyDrive Documentation](https://pythonhosted.org/PyDrive/)  
- [Google Cloud Console](https://console.cloud.google.com/)  
- [CoinGecko API Documentation](https://docs.coingecko.com/v3.0.1/reference/introduction)

**Is it free?**

Yes, using Google Drive for basic needs with PyDrive is free, subject to Google Drive's standard storage quotas and API request limits. No cost is associated with installing and using PyDrive.

**Python libraries / bindings**

- **PyDrive**: A Python library to interact with Google Drive easily. Install using `pip install PyDrive`.  
- **pandas**: Used for data manipulation and analysis. Install using `pip install pandas`.  
- **requests**: A Python package used for making HTTP requests to fetch data from APIs. Install using `pip install requests`.

This project provides experience in handling real-time data ingestion, understanding basic time-series data manipulation, and using Google Drive for seamless data storage and access.

### **Pygrametl**

**Title**: Building a Bitcoin Real-time ETL Pipeline with Pygrametl  
**Difficulty**: 1 (easy)

**Description**  
In this project, students will learn how to use Pygrametl, a lightweight library designed for ETL purposes in Python, to ingest and process real-time Bitcoin price data. Pygrametl is particularly suitable for this project due to its ability to seamlessly integrate with a variety of data sources and destinations, making it a great tool for handling ETL tasks with Python. The objective of this project is to use Pygrametl to extract Bitcoin price data from a public API, transform the data into a desired format for time series analysis, and load it into a local database for storage and further exploration.

**Describe technology**

- **Pygrametl** is a Python library specifically tailored for ETL processes. It simplifies the task of integrating data from various sources while providing basic functionalities to perform data cleaning, transformation, and loading operations.  
- Originally intended for smaller-scale ETL tasks, its flexibility and ease of use make it a suitable choice for educational purposes and for projects that do not require heavy-duty data processing tools.  
- Pygrametl supports various databases and can be easily configured to interact with different types of data, without requiring extensive knowledge in complex ETL frameworks.

**Describe the project**

- **Objective**: Create a simple Pygrametl-based pipeline to ingest, transform, and load Bitcoin price data for time series analysis.  
- **Steps**:  
  1. **Data Ingestion**: Use a public API like CoinGecko to fetch real-time Bitcoin price data.  
  2. **Data Transformation**: Clean the data to remove any unnecessary fields and perform operations such as converting timestamps to a desired format.  
  3. **Data Loading**: Store the processed data into a local SQLite database to maintain historical records for time series analysis.  
  4. **Time Series Analysis**: Perform basic analysis on the stored data, such as plotting price trends over time or calculating moving averages.  
- **Expected Outcome**: Students will successfully learn to set up a simple ETL pipeline using Pygrametl, gain hands-on experience with real-time data processing, and apply basic time series analysis techniques.

**Useful resources**

- [Pygrametl Documentation](http://pygrametl.org/index.html)  
- [SQLite Documentation](https://www.sqlite.org/docs.html)  
- [CoinGecko API Documentation](https://docs.coingecko.com/v3.0.1/reference/introduction)  
- [Python datetime module](https://docs.python.org/3/library/datetime.html) 

**Is it free?**  
Yes, Pygrametl is an open-source library and can be freely used for educational and personal projects. All tools needed for this project, including SQLite and Python, are also free.

**Python libraries / bindings**

- **Pygrametl**: The primary library used for building the ETL pipeline. Install using `pip install pygrametl`.  
- **Requests**: A library for making HTTP requests, used to fetch data from the API. Install using `pip install requests`.  
- **SQLite**: No installation is needed as it is included in the Python standard library.  
- **Matplotlib**: For plotting and visualizing time series data. Install using `pip install matplotlib`.  
- **Pandas** (optional): Can be used for additional data manipulation and transformation tasks. Install using `pip install pandas`.

Through this project, students will develop a foundational understanding of ETL processes using Pygrametl and gain practical experience with real-time data ingestion and time series analysis.

### **Pyjanitor**

**Title**: Process Bitcoin Data Using Pyjanitor  
**Difficulty**: 1 (easy)

**Description**  
Pyjanitor is a Python library designed to simplify data cleaning tasks by providing convenient and intuitive functions. It extends the functionality of Pandas DataFrames, making it easier to perform data wrangling and preprocessing activities. The library includes methods for cleaning column names, filtering data, and handling missing values, which are commonly encountered challenges in data cleaning.

**Describe technology**

- Pyjanitor provides easy-to-use methods to clean and organize raw datasets, extending Pandas capabilities.  
- It allows you to perform operations such as:  
  - Removing and filling missing data  
  - Correcting string formatting in column headers  
  - Filtering and selecting data using expressive syntax  
  - Concatenating and merging DataFrames effortlessly  
- Pyjanitor is useful for data scientists and analysts who need a straightforward approach to data preprocessing.

**Describe the project**  
This project involves ingesting real-time Bitcoin price data from a public API, such as CoinDesk or CoinGecko, and using Pyjanitor to clean and process the data. The task consists of the following steps:

1. **Data Ingestion:**  
     
   - Use Python's `requests` library to fetch Bitcoin prices from the chosen API every 30 seconds to simulate real-time updates.  
   - Store the raw data in a Pandas DataFrame.

   

2. **Data Cleaning and Processing:**  
     
   - Use Pyjanitor to clean and organize the incoming data:  
     - Clean column names to ensure a consistent and readable format.  
     - Handle any missing values or invalid data points using Pyjanitor methods.  
     - Filter the dataset to focus on specific time intervals and relevant fields (e.g., timestamps and prices).

   

3. **Time Series Analysis:**  
     
   - Perform a simple time series analysis to visualize Bitcoin price trends.  
   - Implement basic statistics such as moving averages using Python's scientific libraries like `pandas` and `matplotlib`.

   

4. **Output and Visualization:**  
     
   - Output the cleaned data and basic statistical calculations to a CSV file.  
   - Generate visualizations to display trends in Bitcoin prices over time.

**Useful resources**

- [Pyjanitor Documentation](https://pyjanitor-devs.github.io/pyjanitor/)  
- [CoinGecko API Documentation](https://docs.coingecko.com/v3.0.1/reference/introduction)  
- [Pandas Documentation](https://pandas.pydata.org/docs/)  
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)

**Is it free?**  
Yes, Pyjanitor is an open-source library and can be freely used and modified. The CoinGecko API is also freely accessible for basic queries, although rate limits apply.

**Python libraries / bindings**

- `pyjanitor`: Install using pip with `pip install pyjanitor`  
- `pandas`: Essential for DataFrame operations, install using `pip install pandas`  
- `requests`: For making HTTP requests to fetch data, install using `pip install requests`  
- `matplotlib`: For visualization, install using `pip install matplotlib`

This project offers hands-on experience with data cleaning and analysis using Pyjanitor, providing a practical understanding of handling real-time data in an accessible way.

### **PyKafka**

**Title**: Real-Time Bitcoin Data Ingestion and Analysis using PyKafka

**Difficulty**: 3 (Difficult)

**Description**:  
This project aims to tackle the challenge of real-time data ingestion and processing with a focus on analyzing Bitcoin price fluctuations. We will leverage PyKafka, a powerful Python client for Apache Kafka, to build a robust system capable of handling streaming data efficiently. This project is designed for advanced students who have a strong understanding of Python and an interest in big data systems.

**Describe Technology**:

- PyKafka is a Python client for Apache Kafka, an open-source stream-processing software platform developed by the Apache Software Foundation, used for building real-time data pipelines and streaming apps.  
- Understand the key features of PyKafka:  
  - **Producer**: Streams data into Kafka topics.  
  - **Consumer**: Consumes data from Kafka topics.  
  - **BalancedConsumer**: Distributes the load between multiple consumers for efficiency and fault tolerance.  
- Learn about the Kafka ecosystem and how PyKafka interacts with it, focusing on message serialization, partitions, and offset management.

**Describe the Project**:

- **Objective**: Create a system that ingests real-time Bitcoin price data and performs time series analysis.  
- **Step-by-Step Implementation**:  
  1. **Kafka Setup**: Configure Apache Kafka on a local or cloud server.  
  2. **Data Ingestion**: Use PyKafka to create a producer that fetches Bitcoin price data from a public API (like CoinGecko) and publishes it to a Kafka topic.  
  3. **Data Consumption**: Develop consumers with PyKafka that read from the Kafka topic, processing the incoming data stream. Implement BalancedConsumers for optimized load balancing.  
  4. **Real-Time Analysis**:  
     - Apply basic time series analysis on the consumed data to detect price trends and anomalies.  
     - Use simple moving averages or exponential smoothing for forecasting Bitcoin price movement.  
  5. **Data Visualization**: Integrate Python libraries like matplotlib or Plotly to provide real-time visual insights into Bitcoin price trends.

This project will take approximately 14 days to complete, given its complexity, and requires a thorough understanding of real-time data processing concepts.

**Useful Resources**:

- PyKafka Documentation: [GitHub Repository](https://github.com/Parsely/pykafka)  
- Kafka Documentation: [Apache Kafka Official Documentation](https://kafka.apache.org/documentation/)  
- Bitcoin API: [CoinGecko API Documentation](https://docs.coingecko.com/v3.0.1/reference/introduction)  
- Data Visualization in Python: [matplotlib](https://matplotlib.org/stable/index.html) | [Plotly](https://plotly.com/python/)

**Is it free?**

- PyKafka and Apache Kafka are open-source and free to use.  
- Public Bitcoin APIs like CoinGecko offer free tiers, subject to rate limits.  
- Visualization tools like matplotlib and Plotly offer basic functionalities for free.

**Python Libraries / Bindings**:

- **PyKafka**: A native Python client for Kafka, enabling high-throughput, fault-tolerant, and scalable data streams. Install via pip: `pip install pykafka`.  
- **matplotlib**: A comprehensive library for static, interactive, and animated visualizations in Python. Install via pip: `pip install matplotlib`.  
- **Plotly**: A graphing library that makes interactive, publication-quality graphs online. Install via pip: `pip install plotly`.

This challenging project will provide hands-on experience with PyKafka and expose students to real-time data ingestion, processing, and time series analysis, simulating real-world scenarios of working with streaming data.

### **Pyllms**

**Title:** Real-Time Bitcoin Sentiment Analysis Using PyLLMs​

**Difficulty:** 2 (Medium)​

**Description:**

In this project, students will utilize PyLLMs, a minimal Python library designed to connect to various Large Language Models (LLMs), to perform real-time sentiment analysis on Bitcoin-related news articles. The project involves fetching real-time news data, processing it using LLMs accessed through PyLLMs, and determining the sentiment to assess potential impacts on Bitcoin prices. This project offers an excellent opportunity to learn about integrating LLMs into data processing pipelines using Python.​

**Describe technology:**

PyLLMs is a lightweight Python library that simplifies connections to multiple LLMs, including those from OpenAI, Anthropic, Google, and others. It offers a unified interface to interact with these models, enabling functionalities such as text completion, sentiment analysis, and more. Key features include:​

* **Multi-Model Support:** Easily switch between different LLM providers without altering the core codebase.​  
* **Benchmarking:** Built-in tools to evaluate model performance across various parameters.​  
* **Asynchronous and Streaming Support:** Facilitates efficient data processing with compatible models.​

**Describe the project:**

**Objective:** To analyze real-time Bitcoin-related news articles using LLMs accessed through PyLLMs to assess sentiment and potential impacts on Bitcoin prices.​

**Steps:**

1. **Data Ingestion:**  
   * Use a news API (such as NewsAPI) to fetch real-time news articles related to Bitcoin.​  
2. **Data Processing:**  
   * Extract relevant information from the news articles, such as headlines and content.​  
3. **Sentiment Analysis:**  
   * Utilize PyLLMs to connect to an LLM capable of performing sentiment analysis on the extracted news content.​  
4. **Impact Assessment:**  
   * Analyze the sentiment data to determine potential impacts on Bitcoin prices.​  
5. **Automation:**  
   * Set up a Python script to automate the data ingestion and analysis process at regular intervals (e.g., every hour).​

**Useful resources:**

* [PyLLMs GitHub Repository](https://github.com/kagisearch/pyllms)​  
* [NewsAPI Documentation](https://newsapi.org/docs)​

**Is it free?**

PyLLMs is an open-source library and free to use. However, accessing certain LLMs through PyLLMs may require API keys, which could have associated costs depending on the provider. Similarly, some news APIs offer free tiers with limitations, so it's essential to review their pricing structures.​

**Python libraries / bindings:**

* **PyLLMs:** Install via `pip install pyllms`.​  
* **Requests:** To make HTTP requests for fetching news data from the API (`pip install requests`).​  
* **Schedule:** To assist with running the script at regular intervals (`pip install schedule`).​

This project provides students with hands-on experience in integrating LLMs into data processing workflows using PyLLMs, focusing on real-time sentiment analysis of Bitcoin-related news.​

### **PyMC3**

**Title**: Real-time Bitcoin Price Analysis Using PyMC3

**Difficulty**: 3 (difficult)

**Description** Explore the advanced capabilities of PyMC3, a probabilistic programming library for Python that allows users to build complex Bayesian models quickly. Students will employ PyMC3 to perform probabilistic time series analysis on real-time Bitcoin data, incorporating Bayesian inference and time series forecasting techniques.

**Describe Technology**

- PyMC3 is a Python library for probabilistic programming focused on Bayesian statistical models.  
- It leverages advanced algorithms such as the No-U-Turn Sampler (NUTS), a variant of the Hamiltonian Monte Carlo (HMC) method, to efficiently sample from probabilistic models.  
- The library's user-friendly and expressive API makes it easy to define complex statistical models, conduct inference, and predict future outcomes using Bayesian principles.  
- The technology is particularly useful for uncertain environments where the goal is to infer unknown parameters or predict future states, making it well-suited for real-time data processing.

**Describe the Project**

- **Objective**: Develop a system to continuously ingest real-time Bitcoin price data and perform Bayesian time series analysis to forecast future price trends.  
- **Data Ingestion**: Use a basic Python package like `requests` or `websockets` to pull data from a public Bitcoin API such as CoinGecko or Binance.  
- **Data Processing**: Transform the raw JSON data to a time series format, ensuring it's ready for analysis with PyMC3.  
- **Modeling with PyMC3**:  
  - Define a probabilistic model for time series forecasting. A common choice is to use a Bayesian ARIMA model or a state-space model adapted for Bayesian inference.  
  - Use PyMC3 to perform inference on the model parameters, allowing for uncertainty quantification and robustness in forecasts.  
  - Implement real-time updating of the time series model as new Bitcoin price data is ingested.  
- **Forecasting and Analysis**: Generate probabilistic forecasts of future Bitcoin prices and visualize the results using libraries like `Matplotlib` or `Seaborn`.  
- **Outcome**: Students will gain hands-on experience with Bayesian modeling, learn to handle real-time data streams, and develop proficiency in probabilistic forecasting.

**Useful Resources**

- PyMC3 Documentation: [PyMC3 Documentation](https://docs.pymc.io/)  
- PyMC3 GitHub Repository: [PyMC3 GitHub](https://github.com/pymc-devs/pymc3)  
- CoinGecko API Documentation: [CoinGecko API Documentation](https://docs.coingecko.com/v3.0.1/reference/introduction)  
- Basic tutorial on Bayesian time series analysis: [Time Series Analysis Blog](https://charlescopley.medium.com/conducting-time-series-bayesian-analysis-using-pymc-22269aeb208b)

**Is it Free?** 

Yes, PyMC3 is an open-source library and free to use. Data ingestion from public Bitcoin APIs like CoinGecko is generally free, although certain features or higher data request rates may require payment or registration.

**Python Libraries / Bindings**

- `pymc3`: For building and estimating Bayesian models. Install via `pip install pymc3`.  
- `requests` or `websockets`: For retrieving real-time Bitcoin price data from APIs. Install using `pip install requests` or `pip install websockets`.  
- `pandas`: For data manipulation and transformation into time series format. Install using `pip install pandas`.  
- `numpy`: For numerical computations necessary in model definitions and transformations. Install via `pip install numpy`.  
- Visualization tools like `matplotlib` or `seaborn`: For graphing real-time price and forecast results. Install using `pip install matplotlib seaborn`.

### **PyNaCl**

**Title**: Real-time Bitcoin Price Analysis using PyNaCl

**Difficulty**: 3 (difficult)

**Description**

PyNaCl (Python Network and Cryptographic Library) is a Python binding for the Network and Cryptographic library (NaCl) which provides a high-level API for cryptographic operations, including secret-key encryption, public-key encryption, signatures, password hashing, and more. This library is vital for ensuring data integrity and security in systems that handle sensitive information, such as financial data transactions. In this project, you'll leverage PyNaCl to implement secure methods in handling real-time Bitcoin price data, focusing on how to use Python to process, encrypt, and analyze time-series data from a cryptocurrency exchange.

**Describe technology**

- PyNaCl offers functionalities such as:  
  - Secret-key encryption for encrypting data that can only be decrypted with the same key.  
  - Public-key encryption allowing secure data transmission.  
  - Digital signatures for verifying data authenticity and integrity.  
  - Password hashing, which is essential for securely storing password data.

**Describe the project**

- **Objective**: Develop a secure and efficient system to ingest, encrypt, and analyze real-time Bitcoin prices.  
- **Step 1: Data Ingestion**: Set up a live data feed from a public Bitcoin API (e.g., CoinGecko or CryptoCompare) to collect real-time Bitcoin prices.  
- **Step 2: Secure Data Transmission**: Use PyNaCl's public-key encryption methods to securely transmit and store the incoming data.  
- **Step 3: Time Series Analysis**: Implement a Python script using basic libraries like Pandas for time-series analysis. Possible analyses could include moving averages, volatility measurements, or predictive modeling.  
- **Step 4: Data Verification**: Implement digital signatures to ensure the transmitted data has not been altered.  
- **Step 5: Analysis Output**: Visualize the analyzed data in a secure manner using Matplotlib or Plotly to create real-time price movement graphs.

**Useful resources**

- [PyNaCl Documentation](https://pynacl.readthedocs.io/en/latest/)  
- [NaCl original documentation](https://nacl.cr.yp.to/)  
- [CoinGecko API Documentation](https://docs.coingecko.com/v3.0.1/reference/introduction)

**Is it free?**

Yes, PyNaCl is an open-source library available for free. Most public cryptocurrency price APIs, like CoinGecko, offer a free tier for data usage, but it is crucial to review their documentation for any potential limitations on usage.

**Python libraries / bindings**

- **PyNaCl**: Provides the cryptographic functionalities required for the project. Install via pip: `pip install pynacl`.  
- **Requests**: A simple HTTP library for Python used for making API requests to fetch Bitcoin price data. Install via pip: `pip install requests`.  
- **Pandas**: For time-series data manipulation and analysis. Install via pip: `pip install pandas`.  
- **Matplotlib / Plotly**: Libraries for data visualization. Install via pip: `pip install matplotlib` or `pip install plotly`.

### **PyOD**

**Title**: Real-Time Bitcoin Anomaly Detection with PyOD

**Difficulty**: 2 (medium)

**Description**  
PyOD (Python Outlier Detection) is an open-source Python library that leverages a wide range of anomaly detection algorithms to identify outliers in a given dataset. In the context of data science and machine learning, anomaly detection is crucial for identifying unexpected behaviors in data—such as fraudulent activities, spikes, or anomalies in time series datasets. PyOD is particularly valuable because it supports various detection models, from classical algorithms like Isolation Forest to neural network architectures such as AutoEncoders, allowing flexible applications across different domains and datasets.

**Describe technology**

- **PyOD** is a comprehensive library for detecting outliers in multivariate data.  
- It supports more than 20 detection algorithms, facilitating extensive experimentation.  
- Allows integration with scikit-learn's pipeline and model selection tools for enhanced machine learning workflows.  
- Comes with utility functions for model evaluation, visualization, and comparison.  
- Can handle both individual data points and entire system behaviors as anomalies.

**Describe the project**  
This project involves developing a real-time bitcoin price anomaly detection system using PyOD. By integrating a publicly available Bitcoin price API, students will ingest and process real-time Bitcoin prices to identify unusual price movements (anomalies) over time. The project components are:

- **Data Ingestion**:  
  Use a simple Python script to consume live Bitcoin price data from a source like CoinGecko or Blockchain.info API.  
- **Data Storage**:  
  Temporarily store the incoming data using a lightweight in-memory database or directly aggregate it for processing.  
- **Anomaly Detection**:  
  Employ PyOD to analyze the Bitcoin price stream for anomalies. Start by applying simple algorithms, such as Z-Score or Isolation Forest, to detect price spikes or drops. Assess the effectiveness by comparing multiple PyOD models.  
- **Time Series Analysis**:  
  Extend the basic model to include time series analysis, identifying patterns over hourly or daily intervals and establishing thresholds for what constitutes "anomalous" behavior in the Bitcoin price dataset.  
- **Real-Time Alerts**:  
  Implement a basic notification system to alert users when significant anomalies are detected. This could be a command-line printout or an email notification for significant events.

**Useful resources**

- [PyOD Official Documentation](https://pyod.readthedocs.io/en/latest/)  
- [Time Series Anomaly Detection Toolkits from PyOD](https://github.com/yzhao062/pyod)  
- [CoinGecko API Documentation](https://docs.coingecko.com/v3.0.1/reference/introduction)

**Is it free?**  
Yes, PyOD is free and open-source. Bitcoin price APIs often have free tiers for basic usage.

**Python libraries / bindings**

- **PyOD**: For anomaly detection processes.  
- **Requests**: To make HTTP requests to the Bitcoin price API.  
- **Pandas**: For data manipulation and time series analysis.  
- **NumPy**: For numerical operations and data preparation.  
- **Scikit-learn**: For data preparation and integrating PyOD models into a machine-learning pipeline.

### **PySpark**

**Title**: Real-time Bitcoin Data Processing with PySpark

**Difficulty**: 3 (difficult)

**Description**

This project involves using PySpark to implement a big data system aimed at ingesting and processing real-time Bitcoin price data. Students will gain hands-on experience with PySpark, focusing on its robust capabilities for handling large-scale data processing in a distributed computing environment. The project will cover ingesting real-time data using a public API and applying time series analysis techniques to derive insights from the data.

**Describe technology**

- **PySpark**: PySpark is an interface for Apache Spark in Python. It allows you to write Spark applications using Python APIs and provides the ability to analyze large-scale data using distributed computing. It supports functional programming and provides an easy-to-use API for large-scale data processing.  
  - **Resilient Distributed Datasets (RDDs)**: The core data structure of PySpark that provides fault tolerance and distributed data processing capabilities.  
  - **DataFrames and Datasets**: Higher-level abstractions built on top of RDDs, providing additional features such as schema enforcement and SQL capabilities.  
  - **Spark Streaming**: A component for processing real-time data streams, allowing for scalable and high-throughput data processing.  
  - **Machine Learning Library (MLlib)**: Built-in library for machine learning tasks, leveraging the scale of Spark.

**Describe the project**

- **Objective**: Implement a real-time data processing system to ingest and analyze Bitcoin price data using PySpark.  
- **Setup**: Configure a Spark environment using PySpark for Python. Acquire real-time Bitcoin price data from an API like CoinDesk or CoinGecko.  
- **Data Ingestion**: Use Spark Streaming to fetch Bitcoin data at regular intervals. Design and implement a Spark Streaming job to ingest JSON data streams.  
- **Data Processing**: Parse and transform the raw data into a structured format (DataFrame). Implement aggregation and filtering operations to clean and prepare the data for analysis.  
- **Time Series Analysis**: Develop a time series analysis model using PySpark's MLlib to forecast future Bitcoin prices. Analyze trends and detect patterns over time.  
- **Output Storage**: Store processed and analyzed data in a distributed file system like HDFS or a cloud-based storage solution.  
- **Visualization**: Use Python's plotting libraries (such as Matplotlib) to visualize the results of the analysis, providing insights into Bitcoin price trends.

**Useful resources**

- [PySpark Documentation](https://spark.apache.org/docs/latest/api/python/index.html)  
- [PySpark Streaming Guide](https://spark.apache.org/docs/latest/streaming-programming-guide.html)  
- [Apache Spark: Machine Learning Library (MLlib) Guide](https://spark.apache.org/docs/latest/ml-guide.html)  
- [CoinGecko API Documentation](https://docs.coingecko.com/v3.0.1/reference/introduction)

**Is it free?**

Yes, Apache Spark and PySpark are free and open-source. However, running PySpark at scale may require infrastructure costs if using cloud services.

**Python libraries / bindings**

- **pyspark**: The core library for PySpark, providing APIs for RDD, DataFrame, Spark Streaming, and MLlib. Install using `pip install pyspark`.  
- **requests**: A library for making HTTP requests in Python, useful for accessing Bitcoin APIs. Install using `pip install requests`.  
- **matplotlib**: A popular plotting library for visualizing data in Python. Install using `pip install matplotlib`.

### **pySparkling**

**Title**: Ingest and Process Bitcoin Prices Using PySparkling

**Difficulty**: 1 (easy)

**Description**  
PySparkling is a Python interface for using the H2O Sparkling Water platform that combines the user-friendly nature of Python with the scalable machine learning capabilities of H2O.ai and Apache Spark. It allows you to efficiently process large datasets using Spark's distributed computing capabilities while also leveraging H2O's machine learning algorithms.

**Describe technology**

- **PySparkling**: An interface to integrate H2O.ai Machine Learning with Spark, allowing distributed computations on large data volumes.  
- **Apache Spark**: An open-source, distributed processing system used for big data workloads, providing built-in modules for streaming, SQL, machine learning, and graph processing.  
- **H2O.ai**: Provides scalable machine learning algorithms, such as Gradient Boosting, Random Forest, and K-Means, which can be applied to big data problems.

**Describe the project**  
This project involves using PySparkling to ingest real-time Bitcoin price data from a public API (e.g., CoinGecko) for time series analysis. You will:

1. **Data Ingestion**:  
     
   - Utilize Python's `requests` library to set up an automated data retrieval system fetching Bitcoin price data at regular intervals.  
   - Integrate the collected data with PySpark and convert it to a format suitable for analysis.

   

2. **Data Processing**:  
     
   - Utilize PySparkling to perform basic transformations on the ingested data, such as parsing JSON data, filtering unnecessary fields, and handling missing values.

   

3. **Time Series Analysis**:  
     
   - Implement a simple time series analysis using PySparkling's machine learning functionalities, for instance, using H2O's AutoML to predict future Bitcoin prices based on historical data.

   

4. **Data Storage**:  
     
   - Store the processed data back into a storage solution like a local file system or a cloud storage service in CSV or Parquet format for future retrieval and analysis.

**Useful resources**

- [CoinGecko API Documentation](https://docs.coingecko.com/v3.0.1/reference/introduction)  
- [PySparkling Documentation](https://docs.h2o.ai/sparkling-water/2.3/latest-stable/doc/pysparkling.html)  
- [Apache Spark Documentation](https://spark.apache.org/docs/latest/)  
- [H2O.ai Documentation](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/index.html)

**Is it free?**  
Yes, PySparkling is open-source and free to use. While integrating with other platforms such as cloud storage might incur costs, using local resources is entirely free.

**Python libraries / bindings**

- **PySparkling**: Used for combining Apache Spark's processing power with H2O.ai's machine learning capabilities.  
- **requests**: To fetch real-time data from Bitcoin price APIs.  
- **pandas**: For data manipulation and pre-processing.  
- **numpy**: For numerical operations and handling time series indices.

This project will give students hands-on experience with setting up a basic big data pipeline in Python using PySparkling, allowing them to gain practical skills in real-time data ingestion, processing, and analysis.

### **PySparkling**

**Title**: Real-time Bitcoin Analysis with PySparkling

**Difficulty**: 2 (medium difficulty, should take around 10 days to complete)

**Description**: PySparkling is a Python package that provides support for using H2O.ai's machine learning algorithms with PySpark. It enables seamless integration of Spark's data processing capabilities with H2O's advanced machine learning algorithms. This project will introduce you to data ingestion and real-time processing using PySparkling for analyzing Bitcoin prices, focusing on time series analysis to predict future price trends.

**Describe technology**:

- **PySparkling**: A bridge between Apache Spark and H2O.ai, enabling distributed data processing using Apache Spark combined with H2O's robust machine learning algorithms.  
- **Key Components**:  
  - **Spark DataFrames**: Efficiently manage large datasets in a distributed fashion.  
  - **H2O Machine Learning Models**: Use algorithms like Gradient Boosting, Random Forest, and Deep Learning for advanced predictive capabilities.  
  - **Real-time Data Handling**: Ingest and process streaming data in near real-time.  
- **Integration**: PySparkling allows users to run H2O's machine learning algorithms on Spark DataFrames, making it easier to scale machine learning projects.

**Describe the project**:

- **Objective**: Implement a system to ingest real-time Bitcoin price data and analyze it using time series techniques to forecast future trends.  
- **Steps Involved**:  
  1. **Data Ingestion**:  
     - Use a public API (e.g., CoinGecko) to continuously fetch real-time Bitcoin price data.  
     - Utilize Spark Streaming to process and store this data in a structured format (e.g., Parquet) for analysis.  
  2. **Data Processing**:  
     - Use Spark DataFrames to clean and transform the incoming data.  
     - Implement H2O's time series algorithms through PySparkling for forecasting.  
  3. **Model Training and Evaluation**:  
     - Train a predictive model using historical data to anticipate future Bitcoin prices.  
     - Evaluate the model's performance using common metrics (e.g., RMSE, MAE).  
  4. **Real-time Predictions**:  
     - Deploy the trained model to generate predictions on incoming data, adapting to new trends in real-time.

**Useful resources**:

- [CoinGecko API Documentation](https://docs.coingecko.com/v3.0.1/reference/introduction): API to fetch real-time Bitcoin price data.  
- [H2O.ai Documentation](https://docs.h2o.ai/): Explore H2O's machine learning capabilities and examples.  
- [PySparkling Documentation](https://docs.h2o.ai/sparkling-water/2.3/latest-stable/doc/pysparkling.html): Official documentation and tutorials for getting started with PySparkling.  
- [Apache Spark Documentation](https://spark.apache.org/docs/latest/): Spark's extensive documentation for data and stream processing.

**Is it free?**:

- PySparkling is open-source and free to use. H2O.ai's open-source offerings include the algorithms necessary for this project.

**Python libraries / bindings**:

- **PySparkling**: Install using `pip install h2o-pysparkling-x` (replace `x` with the appropriate version for compatibility).  
- **Apache Spark**: Install using `pip install pyspark`.  
- **H2O**: Install using `pip install h2o`.  
- **Other Python Packages**: Use basic Python packages like `requests` for API calls, `pandas` for data manipulation, and `matplotlib` or `seaborn` for visualization.

This project provides practical experience with combining PySpark and H2O.ai's machine learning capabilities to process and analyze real-time data streams, specifically focusing on time series analysis for predictive insights into Bitcoin price trends.

### **PyStan**

**Title**: Real-Time Bitcoin Analysis with PyStan

**Difficulty**: 1 \= easy

**Description**: The project explores the use of PyStan, a Python interface to the Stan statistical modeling software, to analyze real-time Bitcoin price data. PyStan allows you to implement advanced statistical models seamlessly, enabling you to perform complex analyses with ease. This project will guide you to use PyStan for time series analysis on Bitcoin price data, helping you uncover trends, predict future prices, and understand the underlying patterns in the data.

**Describe technology**:

- PyStan is a Python interface to Stan, which is an open-source platform for statistical modeling, primarily used for Bayesian analysis.  
- PyStan provides the ability to fit models using Markov Chain Monte Carlo (MCMC) techniques, making it powerful for complex statistical analysis.  
- With PyStan, you can define models in a C++-like language, compile them to C++, and then fit the compiled model using Python.  
- PyStan is useful for statistical modeling and Bayesian inference, allowing you to work with probabilistic programming models.

**Describe the project**:

- The project involves writing a Python script to ingest Bitcoin price data from a public API (e.g., CoinGecko).  
- Use PyStan to construct a time series model to analyze the Bitcoin price data. An ARIMA or a more complex Bayesian model could be used for this purpose.  
- Perform exploratory data analysis (EDA) to understand the characteristics of the Bitcoin price data, such as volatility and trends.  
- Implement a Bayesian time series analysis using PyStan, focusing on modeling and forecasting Bitcoin prices.  
- Visualize the results using Python visualization libraries, such as Matplotlib or Seaborn, to display time series plots, trend lines, and forecasted values.  
- Document the process and obtain insights into how effective PyStan is for real-world time series analysis.

**Useful resources**:

- [PyStan Documentation](https://pystan.readthedocs.io/en/latest/)  
- [Stan Documentation](https://mc-stan.org/users/documentation/)  
- [CoinGecko API Documentation](https://docs.coingecko.com/v3.0.1/reference/introduction)  
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)

**Is it free?**: Yes, both PyStan and the CoinGecko API are free to use.

**Python libraries / bindings**:

- `pystan`: Python interface to Stan for statistical modeling and inference.  
- `requests`: To handle the HTTP requests for fetching Bitcoin data from APIs.  
- `numpy`: For numerical operations and data manipulation in Python.  
- `pandas`: For data manipulation and analysis, useful for handling and processing time series data.  
- `matplotlib` & `seaborn`: For data visualization and plotting time series graphs.  
- Use `pip install pystan requests numpy pandas matplotlib seaborn` to install the necessary packages.

### **PyTables**

**Title**: Real-time Bitcoin Data Processing with PyTables

**Difficulty**: 3 (difficult)

**Description** PyTables is a package for managing hierarchical datasets and designed to efficiently and easily cope with extremely large amounts of data. It provides functionalities to organize and process large datasets in a columnar format, which is highly beneficial for both querying and compressing the data. For this project, students will leverage PyTables to ingest and process real-time data about Bitcoin from a reliable API source. The focus will be on implementing a time-series analysis of Bitcoin prices.

**Describe Technology**

- PyTables is built on top of the HDF5 library, allowing for storage and manipulation of complex data structures while providing fast I/O operations.  
- It supports the creation and management of dictionaries of data (nodes), enabling structured storage and hierarchical queries.  
- PyTables efficiently compresses data on the fly and is optimized for both importing large datasets and performing fast queries.  
- It includes various options for querying, writing, and reading data using the NumPy interface.

**Describe the Project**

- The primary goal is to ingest Bitcoin price data from a public API such as CoinGecko or Coinbase in real time.  
- Students will configure a scheduler, such as `schedule` module or `APScheduler`, to fetch data at regular intervals and store it in a PyTables database.  
- Work on organizing the incoming data into hierarchical structures, storing it in a way optimal for time-series analysis.  
- Implement time-series analysis to identify trends, patterns, and potential predictive insights using the stored data. This can involve analyzing moving averages, price volatility, or other relevant metrics.  
- Finally, visualize the time-series data using a package like Matplotlib or Plotly to demonstrate trends.

**Useful Resources**

- PyTables Documentation: [PyTables Documentation](http://www.pytables.org/)  
- HDF5 Format Specification: [HDF5](https://support.hdfgroup.org/documentation/hdf5/latest/_s_p_e_c.html)  
- CoinGecko API: [CoinGecko API Documentation](https://docs.coingecko.com/v3.0.1/reference/introduction)  
- Python Schedule Library: [Schedule Package](https://schedule.readthedocs.io/)

**Is it free?** 

Yes, PyTables is an open-source library and free to use under the BSD license. However, any costs associated with accessing the Bitcoin data API should be considered.

**Python Libraries / Bindings**

- **PyTables**: Core library for data handling and storage.  
- **NumPy**: For data manipulation within PyTables.  
- **schedule/APScheduler**: To implement real-time data fetching functionality.  
- **requests**: For making HTTP requests to access Bitcoin price data from APIs.  
- **Matplotlib/Plotly**: For visualizing the results of time-series analysis.

### **Pytest-mock**

**Title**: Analyzing Bitcoin Prices with Pytest-mock

**Difficulty**: 1 (easy)

**Description**  
This project focuses on using the `pytest-mock` library to facilitate testing in Python applications. `pytest-mock` enables developers to replace parts of their system under test and make assertions on how they are used. This is particularly useful for testing components that interact with external systems, such as APIs or databases.

In this project, students will build an application that ingests real-time Bitcoin price data from a public API and performs basic time series analysis. The goal is to create a test-driven development (TDD) process using `pytest-mock`, allowing students to simulate different scenarios and validate their data processing logic effectively.

**Describe technology**

- `pytest-mock` is a plugin for the pytest framework that provides powerful mocking capabilities, making it easier to write unit tests for Python applications.  
- Students will learn how to use `pytest-mock` to create mock objects which can simulate the behavior of complex systems or external dependencies.  
- The library allows tests to assert whether particular actions were performed on a mocked object, making it ideal for scenarios where real-time data changes are simulated.

**Describe the project**

- **Data Ingestion**: Set up a basic Python script to fetch real-time Bitcoin price data from a public API, such as CoinGecko or CryptoCompare.  
- **Simulate Real-time Processing**: Using simple Python scripts, simulate the ingestion of data at regular intervals.  
- **Time Series Analysis**: Implement basic operations on the ingested data such as calculating moving averages or detecting trends.  
- **Testing Using Pytest-mock**: Develop a suite of unit tests using pytest and `pytest-mock` to:  
  - Mock API responses to test how the system behaves with varying Bitcoin prices.  
  - Simulate network failures or API downtimes and verify system resilience.  
  - Ensure that time series analysis logic performs correctly by mocking various input datasets.

**Useful resources**

- [pytest-mock Documentation](https://pytest-mock.readthedocs.io/en/latest/)  
- [pytest Official Documentation](https://docs.pytest.org/en/stable/)  
- [CoinGecko API Documentation](https://docs.coingecko.com/v3.0.1/reference/introduction)

**Is it free?**  
Yes, both `pytest-mock` and pytest are open-source and free to use. The public APIs for Bitcoin price data typically offer free tiers, but they may have limitations on the number of requests.

**Python libraries / bindings**

- `pytest-mock`: Install via pip using `pip install pytest-mock`. It extends pytest with powerful mocking features.  
- `requests`: Used for making HTTP requests to the Bitcoin price API. Install using `pip install requests`.  
- Optionally, `pandas`: For handling time series data. Install using `pip install pandas`. This library is useful for more advanced data manipulation and analysis, aiding the processing of Bitcoin price data.

### **PyTorch**

**Title**: Real-time Bitcoin Price Analysis with PyTorch

**Difficulty**: 1 (Easy)

**Description**:  
This project involves leveraging PyTorch, a popular open-source machine learning library, to perform time series analysis on real-time Bitcoin price data. Students will create a simple LSTM (Long Short-Term Memory) network using PyTorch to model Bitcoin prices over time. This project introduces the basic functionalities of PyTorch and its utility for handling dynamic computational graphs and deep learning models.

**Describe Technology**:

- **PyTorch**: PyTorch is an open-source machine learning library developed by Facebook's AI Research lab. It is widely used for tasks involving deep learning due to its ease of use, flexibility, and support for dynamic computational graphs.  
- **Core Features**:  
  - **Tensor Computation** similar to NumPy with strong GPU acceleration.  
  - **Autograd**: Automatic differentiation for building and training neural networks.  
  - **TorchScript**: A way to create serializable and optimizable models from PyTorch code.  
  - Supports both CPU and GPU computations.

**Describe the Project**:

- **Objective**: Implement a simple LSTM model in PyTorch to analyze real-time Bitcoin price data for time series prediction.  
- **Steps**:  
  - Use a Python package like `requests` to fetch real-time Bitcoin prices from a public API such as CoinGecko.  
  - Preprocess the data using Python libraries like `pandas` for handling time series data.  
  - Implement an LSTM model in PyTorch:  
    - Define the LSTM model architecture.  
    - Train the model on the fetched Bitcoin price data.  
    - Evaluate the model's performance over time.  
  - Use Python to visualize the real-time predictions compared to actual prices using a library like `matplotlib`.

**Useful Resources**:

- [PyTorch Official Documentation](https://pytorch.org/docs/stable/index.html)  
- [PyTorch Tutorials](https://pytorch.org/tutorials/)  
- [CoinGecko API Documentation](https://docs.coingecko.com/v3.0.1/reference/introduction)

**Is it free?**  
Yes, PyTorch is an open-source library and is free to use. Additionally, many public APIs for Bitcoin prices offer free tiers, but ensure you review their usage policies.

**Python Libraries / Bindings**:

- `torch`: Core PyTorch library for building and training models.  
- `torch.nn`: PyTorch module for crafting neural network architectures.  
- `requests`: For accessing APIs and fetching Bitcoin price data.  
- `pandas`: Used for data manipulation and handling time series.  
- `matplotlib`: For visualizing the time series and model predictions.

This project gives students a foundation in PyTorch and practical experience with a simple machine learning task involving real-time data ingestion and processing.

### **PyTorch Forecasting**

**Title**: Time Series Analysis using PyTorch Forecasting on Bitcoin Prices

**Difficulty**: 1 (easy)

**Description**:  
PyTorch Forecasting is a library built on top of PyTorch designed to make time series forecasting with neural networks simple and effective. It provides a high-level API to build powerful and complex models for time series prediction while abstracting much of the coding overhead associated with neural networks. This project will introduce you to the basic functionalities of PyTorch Forecasting using real-time Bitcoin price data to predict future trends.

**Describe technology**:

- **PyTorch Forecasting**: An open-source library that simplifies the process of designing and training neural networks for time series forecasting. It supports a range of models, evolves with the latest research, and is designed for ease of use, drawing on PyTorch's rich ecosystem.  
- Key concepts include:  
  - **TimeseriesDataset**: Prepares time series data for model training by handling encoding and normalizing.  
  - **Temporal Fusion Transformer (TFT)**: A model architecture for multi-horizon time series forecasting.  
  - **Rich Output**: Provides various diagnostic metrics and visualization options to evaluate model performance.

**Describe the project**:

- The project involves fetching real-time Bitcoin price data from a public API like CoinGecko.  
- **Step 1**: Use Python packages (requests, pandas) to pull and clean the Bitcoin price data, focusing on features like timestamp and price.  
- **Step 2**: Prepare the dataset using PyTorch Forecasting’s TimeseriesDataset for supervised learning, ensuring it's ready for neural network input.  
- **Step 3**: Implement a simple model using PyTorch Forecasting's Temporal Fusion Transformer or another model of your choice.  
- **Step 4**: Train the model to predict future Bitcoin prices based on historical data.  
- **Step 5**: Visualize the results and compare predicted prices against actual values using Matplotlib.

**Useful resources**:

- [PyTorch Forecasting Documentation](https://pytorch-forecasting.readthedocs.io/)  
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)  
- [CoinGecko API Documentation](https://docs.coingecko.com/v3.0.1/reference/introduction)  
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)

**Is it free?**  
Yes, both PyTorch Forecasting and the CoinGecko API are free to use. You will need Python installed on your machine.

**Python libraries / bindings**:

- **PyTorch Forecasting**: Install via `pip install pytorch-forecasting`.  
- **PyTorch**: Required for PyTorch Forecasting; install via `pip install torch`.  
- **Requests**: For API calls, install via `pip install requests`.  
- **Pandas**: For data manipulation, install via `pip install pandas`.  
- **Matplotlib**: For visualization, install via `pip install matplotlib`.

### **pytracking**

**Title:** Real-Time Bitcoin Price Alert System with PyTracking​

**Difficulty:** 2 (Medium)​

**Description:** In this project, students will develop a real-time Bitcoin price alert system using PyTracking, a Python library designed for tracking and analyzing data. The system will monitor Bitcoin prices and send notifications when specific thresholds are crossed. This project introduces students to data ingestion from APIs, data processing, and implementing alert mechanisms using Python.​

**Describe Technology:** PyTracking is a Python library primarily used for tracking and analyzing data. It provides functionalities to process and monitor data, making it suitable for applications that require real-time tracking and alerts. In this project, PyTracking will be utilized to monitor Bitcoin price movements and trigger alerts based on predefined conditions.​

**Describe the Project:**

**Objective:** To monitor real-time Bitcoin prices and send alerts when specific price thresholds are reached.​

**Steps:**

1. **Data Ingestion:** Utilize a public Bitcoin API (such as CoinGecko) to fetch real-time price data in JSON format.​  
2. **Data Processing:** Parse the JSON response to extract the current Bitcoin price.​  
3. **Conditional Logic:** Implement logic to evaluate if the price crosses predetermined thresholds (e.g., a 5% increase or decrease from the previous hour).​  
4. **Alerts:** Use PyTracking to monitor the price data and trigger alerts when the set conditions are met.​  
5. **Automation:** Set up a Python script to automatically execute these steps at regular intervals (e.g., every 10 minutes).​

**Useful Resources:**

* [PyTracking Documentation](https://pypi.org/project/pytracking/)​  
* [CoinGecko API Documentation](https://docs.coingecko.com/v3.0.1/reference/introduction)

**Is it free?** Yes, PyTracking is an open-source library and free to use. CoinGecko also offers free access to their API for fetching cryptocurrency data.​

**Python Libraries / Bindings:**

* **Requests:** To make HTTP requests for fetching Bitcoin price data from the API (`pip install requests`).​  
* **PyTracking:** For monitoring and analyzing data (`pip install pytracking`).​  
* **Schedule:** To assist with running the script at regular intervals (`pip install schedule`).​

By completing this project, students will gain experience in working with APIs, processing real-time data, and implementing alert systems using Python.​

### **pywhy**

**Title:** Real-Time Bitcoin Price and News Sentiment Analysis with PyWhy and NewsAPI​

**Difficulty:** 3 (Difficult)

**Description:** This project aims to analyze real-time Bitcoin price data alongside news sentiment to uncover potential causal relationships between media coverage and price fluctuations. Students will utilize PyWhy for causal inference and NewsAPI to fetch relevant news articles, enabling a comprehensive analysis of how external factors influence Bitcoin's market behavior.​

**Describe Technology:**

* **PyWhy:**  
  * A Python library focused on causal inference.​  
  * Provides tools to model, estimate, and validate causal relationships in data.​  
  * Supports Directed Acyclic Graph (DAG) causal modeling and other statistical methods.​  
  * Offers an intuitive API for visualizing, simulating, and analyzing causality within datasets.​  
* **NewsAPI:**  
  * A service that provides access to news articles from various sources worldwide.​  
  * Allows fetching articles based on keywords, sources, and dates.​  
  * Offers a free tier suitable for development and testing purposes.​

**Describe the Project:**

**Objective:** To analyze the impact of news sentiment on real-time Bitcoin price movements using causal inference techniques.​

**Tasks:**

1. **Data Ingestion:**  
   * Fetch real-time Bitcoin price data at regular intervals using a public API (e.g., CoinGecko).​  
   * Retrieve news articles related to Bitcoin using NewsAPI, focusing on recent publications.​  
2. **Data Preprocessing:**  
   * Clean and structure the Bitcoin price data for analysis.​  
   * Perform sentiment analysis on the fetched news articles to quantify their sentiment scores.​  
3. **Causal Modeling:**  
   * Utilize PyWhy to construct a causal model examining the relationship between news sentiment and Bitcoin price movements.​  
   * Identify potential confounding variables and adjust the model accordingly.​  
4. **Analysis and Visualization:**  
   * Conduct time series analysis to observe correlations and causal effects.​  
   * Visualize the findings using libraries like Matplotlib or Seaborn to illustrate trends and causal links.​

**Useful Resources:**

* [PyWhy](https://www.pywhy.org)  
* [NewsAPI](https://newsapi.org/)  
* [CoinGecko API Documentation](https://docs.coingecko.com/v3.0.1/reference/introduction)​

**Is it Free?**

Yes, both PyWhy and NewsAPI offer free access suitable for development and testing. NewsAPI's free tier allows for up to 100 requests per day, which should suffice for small-scale projects.  CoinGecko also provides free access to cryptocurrency data.​

**Python Libraries / Bindings:**

* `pywhy`​  
* `newsapi-python`​  
* `requests`​  
* `pandas`​  
* `nltk` (for sentiment analysis)​  
* `matplotlib` or `seaborn`​

This project offers a practical introduction to causal inference and sentiment analysis, providing students with valuable skills in analyzing the interplay between media coverage and cryptocurrency markets.​

### **Qlik Sense**

**Title**: Real-time Bitcoin Price Analysis Using Qlik Sense

**Difficulty**: 3 (difficult)

**Description**: Qlik Sense is a powerful business intelligence and data visualization tool that allows users to create interactive reports and dashboards. It enables rapid insights through a robust associative data model and advanced analytics capabilities. In this project, students will learn to integrate Qlik Sense with real-time data streams, specifically focusing on ingesting and visualizing Bitcoin price data. The project will involve creating a real-time dashboard in Qlik Sense that showcases price trends, volatility, and predictive analytics by implementing a time series analysis using Qlik Sense's advanced analytical functions and Python.

**Describe technology**:

- Qlik Sense is designed for self-service analytics that simplifies data and derives meaningful insights, utilizing a drag-and-drop interface and powerful AI-driven features.  
- It uses an associative data model to link data from multiple sources, providing a unique advantage in data analysis.  
- Through its open APIs, Qlik Sense supports integrating analytics with custom applications, including Python scripts, to extend its capabilities.

**Describe the project**:

- **Step 1**: Set up a real-time data feed using Python to pull Bitcoin price data from a public API (e.g., CoinGecko) at regular intervals.  
- **Step 2**: Use Qlik Sense's data connector capabilities to ingest this data continuously, ensuring that the data is seamless and real-time through scheduled batch updates.  
- **Step 3**: Create a Qlik Sense dashboard that initially shows real-time Bitcoin price charts and other basic metrics like 24-hour high and low prices.  
- **Step 4**: Implement time series analysis on the data using Qlik Sense's analytical capabilities and integrate Python for advanced statistical modeling, like trend forecasting or volatility analysis.  
- **Step 5**: Extend the dashboard to allow for interactive exploration, letting users filter by time intervals, compare historical data, and visualize predictive analytics results.

**Useful resources**:

- [Qlik Sense Official Documentation](https://help.qlik.com/)  
- [CoinGecko API Documentation](https://docs.coingecko.com/v3.0.1/reference/introduction)

**Is it free?**: 

Qlik Sense offers a free trial version, which provides access to basic features necessary for completing this project. This free tier has limitations on data size and storage but is sufficient for academic and small-scale projects.

**Python libraries / bindings**:

- `requests`: For making HTTP requests to obtain the Bitcoin pricing data from the public API. Install with `pip install requests`.  
- `pandas`: For handling and manipulating the data before integrating with Qlik Sense. Install with `pip install pandas`.  
- `Qlik-Py-Tools`: A set of Python scripts that can be used to build advanced analytics capabilities directly within Qlik Sense, enabling deeper integrations. This can be configured to run as a service in the background.

This project aims to deepen students' understanding of Qlik Sense's functionalities and the integration of Python for dynamic data analysis and visualization in a complex real-world scenario involving real-time data.

### **Rasa**

Title: Real-time Bitcoin Price Analysis using Rasa

Difficulty: 3 (Difficult)

**Description**

- Rasa is an open-source machine learning framework for building AI assistants and chatbots. It enables developers to create conversational AI applications that can understand and process natural language input, and respond accordingly.  
- This project focuses on leveraging Rasa to ingest real-time Bitcoin price data via conversational interfaces. Participants will utilize basic Python packages for additional data processing and analysis.  
- The goal is to provide hands-on experience with Rasa and demonstrate its potential in developing applications for real-time data interaction and time series analysis.

**Describe technology**

- Rasa consists of two main components: Rasa NLU (Natural Language Understanding), which handles intent classification and entity extraction, and Rasa Core, which is responsible for dialogue management.  
- It uses machine learning models to interpret user input and make decisions about which actions to take next.  
- Rasa allows integration with various messaging platforms, APIs, and data sources to build versatile chatbots and assistants.

**Describe the project**

- Students will create a basic AI assistant using Rasa to interact with users and provide real-time updates on Bitcoin prices.  
- The assistant will use a public API, such as CoinGecko, to fetch Bitcoin prices at regular intervals.  
- Participants will implement a module in Rasa to process incoming data, analyze price trends, and provide insights by performing basic time series analysis, such as identifying price highs and lows within specified time frames.  
- The project includes setting up a simple conversation flow where users can ask about current Bitcoin prices or request historical data summaries.  
- The Rasa assistant will frame responses in natural language, allowing users to interact seamlessly and intuitively.

**Useful resources**

- Rasa documentation: [Rasa Official Documentation](https://rasa.com/docs/)  
- Rasa GitHub repository: [Rasa GitHub](https://github.com/RasaHQ/rasa)  
- CoinGecko API documentation: [CoinGecko API Documentation](https://docs.coingecko.com/v3.0.1/reference/introduction)  
- Tutorials for integrating Rasa with external APIs: [Rasa API Integration Guide](https://rasa.com/docs/rasa/connectors/your-own-website)

**Is it free?**

- Yes, Rasa is open-source and free to use. There are no associated costs for using the Rasa framework. Public data APIs like CoinGecko also offer free access to a wide range of cryptocurrency data.

**Python libraries/bindings**

- `rasa`: The main Python package for building and running Rasa assistants. Install using `pip install rasa`.  
- `requests`: A simple Python HTTP library to fetch data from public APIs such as CoinGecko. Install using `pip install requests`.  
- `pandas`: A popular data manipulation library useful for processing and analyzing time series data. Install using `pip install pandas`.

### **Ray**

**Title**: Real-time Bitcoin Data Processing with Apache Ray

**Difficulty**: 2 (Medium)

**Description**

Apache Ray is an open-source, distributed computing framework that allows you to build applications that can handle large-scale, real-time data processing tasks. With Ray, developers can more easily scale their Python workloads to utilize multiple cores and nodes, offering both simplicity and powerful abstractions for concurrent and parallel computing.

This project involves utilizing Ray to ingest real-time Bitcoin price data, process it, and perform time series analysis on the data. Students will gain hands-on experience with real-time data processing using Ray, learning to build scalable systems with a focus on distributed computing paradigms.

**Describe technology**

- **Ray Core Concepts**: Understand the key features of Apache Ray, such as its actor model for stateful computation and task parallelism. Learn about Ray's architecture, including its scheduler, distributed execution, and object store for shared-memory and communication between tasks.  
    
- **Real-time Data Processing**: Learn how Ray can handle streaming data and integrate with other data sources and systems to manage continuous data ingestion. Understand how the framework can one take advantage of the multi-core processors for improved performance.

**Describe the project**

- **Real-time Data Ingestion**: Implement a Python script using Ray to continuously ingest Bitcoin price data from an API such as CoinGecko or another public data source. Set up a mechanism for maintaining live connections and fetching fresh data in real-time.  
    
- **Data Processing and Transformation**: Use Ray's parallel processing capabilities to handle large volumes of incoming data. Apply transformations to the data, such as converting timestamps, calculating percentage changes, or filtering for specific criteria.  
    
- **Time Series Analysis**: Conduct a basic time series analysis on the ingested and processed data, such as moving averages or volatility indices. Utilize Ray's distributed nature to efficiently manage and calculate these metrics across large datasets.  
    
- **Visualization and Reporting**: Optionally, integrate visualization tools such as Matplotlib or Plotly to create real-time graphs and dashboards that display the Bitcoin prices and analysis results.

**Useful resources**

- [Ray Official Documentation](https://docs.ray.io/en/latest/)  
- [CoinGecko API Documentation](https://docs.coingecko.com/v3.0.1/reference/introduction)

**Is it free?**

Yes, Apache Ray is an open-source project licensed under the Apache License 2.0, and it can be used for free. However, students may incur costs from any required cloud infrastructure or data services used during the project.

**Python libraries / bindings**

- **Ray**: Core library for distributed computing and parallel processing. It can be installed using `pip install ray`.  
    
- **Requests or HTTPx**: Used for accessing public APIs for Bitcoin data. Install via `pip install requests` or `pip install httpx`.  
    
- **Pandas**: Essential for data manipulation and analysis, especially useful in handling time series data. Install with `pip install pandas`.  
    
- **Plotly or Matplotlib**: For visualizing time series data. Install with `pip install plotly` or `pip install matplotlib`.

This project provides a practical understanding of distributed real-time data processing systems and how Ray can be leveraged to enhance Python's capabilities for time-sensitive data tasks.

### **Redis**

**Title**: Real-time Bitcoin Price Analytics with Redis

**Difficulty**: 2 (medium)

**Description**  
Redis is an open-source, in-memory data structure store used as a database, cache, and message broker. It supports various data structures such as strings, lists, sets, and hashes and is known for its speed and efficiency in handling real-time data, making it a favored choice for developing scalable applications that require fast data access and processing. In this project, students will gain hands-on experience with Redis, focusing on its capabilities for ingesting and processing real-time Bitcoin price data.

**Describe technology**

- Redis is an in-memory data store that holds data in memory, allowing for rapid data access and manipulation.  
- Supports multiple data structures such as strings, lists, sets, sorted sets, and hashes.  
- Ideal for real-time analytics due to its low latency and high throughput.  
- Includes pub/sub messaging system for real-time event streaming.  
- Persistence options: Redis offers both snapshot storage (RDB) and append-only file (AOF) logs for data durability.  
- Supplementary features include data replication for high availability and clustering for scalability.

**Describe the project**  
This project involves setting up a Redis server to process real-time Bitcoin price data using a public API (e.g., CoinGecko). The project is divided into the following steps:

1. **Data Ingestion**:  
     
   - Use Python to develop a client that fetches real-time Bitcoin prices at regular intervals from the API.  
   - Utilize Redis commands to store data in appropriate data structures (e.g., lists or sorted sets) for efficient retrieval and analysis.

   

2. **Real-time Processing**:  
     
   - Implement Redis Pub/Sub to stream real-time updates to subscribed clients. This can be used for immediate data analysis and reporting.  
   - Perform basic processing like calculating moving averages or percent changes using Redis' in-memory computations.

   

3. **Time Series Analysis**:  
     
   - Use Redis for time-series data storage and processing, implementing basic time-series operations such as slicing and filtering to analyze price fluctuations.  
   - Extend the project with visual representations of the data using third-party Python libraries (like Matplotlib or Plotly) to visualize trends over time.

   

4. **Final Presentation**:  
     
   - Conclude the project with a presentation detailing your findings, the choices made during implementation, and potential real-world applications.

**Useful resources**

- [Redis Documentation](https://redis.io/documentation)  
- [Redis Python Client (redis-py) Documentation](https://pypi.org/project/redis/)  
- [CoinGecko API Documentation](https://docs.coingecko.com/v3.0.1/reference/introduction)

**Is it free?**  
Yes, Redis is open-source and free to use. However, commercial support and hosted services offered by Redis Labs may incur costs.

**Python libraries / bindings**

- `redis`: Python client for interacting with Redis. Install using `pip install redis`.  
- `requests`: For making HTTP requests to fetch data from APIs. Install using `pip install requests`.  
- `Matplotlib` or `Plotly`: For data visualization purposes. Install using `pip install matplotlib` or `pip install plotly`.

### **River**

**Title:** Ingest Bitcoin Prices using River for Real-Time Processing  
**Difficulty:** (3=difficult)

**Description:**  
River is an online streaming machine learning library designed for incremental learning from continuous data streams. It's particularly suited for handling real-time data as it learns and adapts incrementally with each new incoming data point. This project involves utilizing River to construct a real-time Bitcoin price analysis tool, focusing on time series analysis. Students will gain hands-on experience in ingesting and processing continuous Bitcoin price data streams using River and basic Python packages for data extraction.

**Describe Technology:**

- **River Library:** River emphasizes scalable, real-time data processing and streaming analytics. As a machine learning library for Python, it provides algorithms for both supervised and unsupervised learning that can update incrementally. Key features include:  
  - Online learning, where models are updated as data arrives.  
  - Streaming data processing capabilities, crucial for real-time data applications.  
  - A range of built-in algorithms for regression, classification, and clustering tasks.  
  - Integration with other Python libraries such as pandas for additional data manipulation.

**Describe the Project:**

- **Objective:** Implement a robust, real-time streaming solution in Python using River to analyze Bitcoin price data, with a focus on time series analysis.  
- **Data Acquisition:** Use a public API, like the CoinGecko API, to stream Bitcoin price data continuously. Implement mechanisms to handle API requests efficiently, ensuring real-time data ingestion.  
- **Data Processing:**  
  - Use River to set up a real-time data processing pipeline. This should include maintaining a rolling window of recent Bitcoin prices and implementing an online learning model to predict short-term trends.  
  - Focus on time series analysis techniques like autoregressive models to work with the real-time data using River's streaming capabilities.  
- **Analysis:**  
  - Perform real-time analytics to compute metrics like moving averages, volatility indexes, or other time series indicators.  
  - Develop a visualization module using matplotlib to visualize the trends and predictions dynamically.  
- **Complexity**: This project is demanding as it combines aspects of real-time data ingestion, machine learning algorithm implementation, and time-series analysis with a strong emphasis on River's capabilities.

**Useful Resources:**

- [River API Documentation](https://riverml.xyz/latest/api/overview/): Access the official documentation to understand the library's capabilities and get started with streaming data applications.  
- [CoinGecko API Documentation](https://docs.coingecko.com/v3.0.1/reference/introduction): Familiarize with APIs like CoinGecko or others to ensure a seamless data ingestion pipeline.

**Is it free?**

- Yes, River is an open-source library and is freely available for use. However, API services may have usage limitations or require a subscription for extensive use, so students should check the specific API terms.

**Python Libraries / Bindings:**

- **River:** The core library for incremental learning from streaming data. Install via `pip install river`.  
- **requests:** A simple library for making HTTP requests to APIs. Useful for data ingestion from public Bitcoin price APIs. Install via `pip install requests`.  
- **matplotlib:** A plotting library for Python and its numerical mathematics extension, NumPy, useful for visualization. Install via `pip install matplotlib`.

This project demands a sophisticated approach to managing real-time data and implementing machine learning models that adapt instantly to new data inputs, providing a comprehensive learning experience in big data systems using River.

### **s3fs**

**Title**: Time Series Analysis of Bitcoin Prices Using s3fs

**Difficulty**: 3 (difficult)

**Description**  
This project involves leveraging the s3fs Python library to build a sophisticated big data system that ingests, processes, and analyzes real-time Bitcoin price data using time series analysis techniques. The objective is to provide students with an in-depth understanding of working with cloud-based data storage and manipulation, focusing on handling large datasets efficiently.

**Describe technology**

- **s3fs**: s3fs is a Python library that provides a convenient interface for working with Amazon S3 storage. It integrates the S3 API into a standard Python filesystem interface, enabling easier manipulation of files stored in the cloud.  
- Key Features:  
  - Transparent access to S3 buckets using Python's built-in file handling capabilities.  
  - Support for streaming large datasets, facilitating efficient data processing.  
  - Compatible with many Python packages, enhancing its utility in data science workflows.

**Describe the project**

- **Objective**: Implement a robust system that ingests real-time Bitcoin price data from a public API (e.g., CoinGecko) and performs time series analysis using Python, all while storing and managing the data using Amazon S3.  
    
- **Core Steps**:  
    
  1. **Data Ingestion**: Use Python packages like `requests` to pull Bitcoin price data at regular intervals and interface with s3fs to store this data in an S3 bucket.  
  2. **Data Preprocessing**: Write scripts using s3fs to access and clean the accumulated raw data, preparing it for analysis.  
  3. **Time Series Analysis**:  
     - Implement time series analysis techniques using libraries such as Pandas and statsmodels to detect trends, seasonality, and anomalies in the Bitcoin price data.  
     - Focus on crafting models that can predict future price movements or highlight significant historical changes.  
  4. **Visualization**: Generate plots using `matplotlib` or `seaborn` to visualize the trends and analysis results stored in S3.  
  5. **Reporting**: Store the final analysis and visualization results in a structured format back in the S3 bucket for sharing and further review.


- **Expected Outcome**:  
    
  - A comprehensive understanding of integrating s3fs with Python for data storage and retrieval.  
  - Experience in modeling and analyzing real-time data using time series techniques, culminating in actionable insights into Bitcoin price behavior.

**Useful resources**

- [CoinGecko API Documentation](https://docs.coingecko.com/v3.0.1/reference/introduction)  
- [s3fs Documentation](https://s3fs.readthedocs.io/en/latest/)  
- [Pandas Time Series Documentation](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html)  
- [Statsmodels Time Series Analysis](https://www.statsmodels.org/stable/tsa.html)  
- [Amazon S3 Documentation](https://docs.aws.amazon.com/s3/index.html)

**Is it free?**  
s3fs itself is free and open-source. However, to use Amazon S3, you'll need an AWS account. AWS provides a free tier for S3, but data storage and transfer may incur costs once you exceed the free tier limits.

**Python libraries / bindings**

- **s3fs**: `pip install s3fs` \- Essential for interacting with S3 storage within a Python environment.  
- **requests**: `pip install requests` \- For harvesting real-time Bitcoin data from external APIs.  
- **pandas**: `pip install pandas` \- Crucial for data manipulation and analysis.  
- **statsmodels**: `pip install statsmodels` \- Useful for advanced statistical analysis and time series forecasting.  
- **matplotlib** / **seaborn**: (`pip install matplotlib seaborn`) \- Required for creating visual representations of the analysis results.

This project offers a challenging yet rewarding opportunity to master cloud data storage with s3fs while applying time series analysis techniques in a practical context.

### **SageMath**

**Title**: Analyzing Bitcoin Time Series with SageMath

**Difficulty**: 3 (Difficult \- it should take around 14 days to complete)

**Description**:  
SageMath is an open-source mathematics software system that integrates a wide range of mathematics-related packages and functionalities into a unified Python-based interface. It provides extensive tools for symbolic mathematics, numerical computations, data visualization, algebra, calculus, and much more. For this project, students will leverage SageMath’s capabilities to ingest, process, and analyze real-time Bitcoin price data, focusing on time series analysis.

**Describe technology**:

- **SageMath**: A powerful mathematical software built on Python, combining many existing open-source packages into a common interface. It covers many aspects of mathematics, such as algebra, calculus, and discrete mathematics.  
- **Capabilities**:  
  - Symbolic computation using SymPy, a symbolic mathematics library.  
  - Data visualization using matplotlib and other related plotting libraries.  
  - Advanced mathematical algorithms and functions accessible through an intuitive interface.  
- **Use Cases**:  
  - Complex mathematical modeling and simulations.  
  - Statistical analysis and probability calculations.  
  - Educational purposes for teaching advanced mathematics and computations.

**Describe the project**:

- **Objective**: Implement a real-time data ingestion pipeline for Bitcoin price data using SageMath, followed by an in-depth time series analysis with focus on pattern identification and prediction.  
    
- **Steps**:  
    
  1. **Data Ingestion**: Set up a Python script to fetch real-time Bitcoin price data from a public API (e.g., CoinGecko). Implement a mechanism to store the captured data in an SQLite database or a CSV file for local storage and retrieval.  
       
  2. **Preprocessing**: Utilize SageMath to clean and preprocess the ingested data, handling missing values and smoothing out noise in the series.  
       
  3. **Exploratory Data Analysis**: Use SageMath's visualization tools to plot the time series data, identifying trends, seasonality, and any anomalies present in the dataset.  
       
  4. **Time Series Modeling**: Apply advanced time series models, such as ARIMA or Exponential Smoothing, for forecasting future Bitcoin prices. Students will explore parameter tuning and model validation within SageMath, utilizing its strong support for mathematical operations and modeling.  
       
  5. **Visualization and Reporting**: Create detailed plots and reports that encompass findings from the data analysis and modeling phases. This includes predictions, model accuracy, and insights derived from the data.

**Useful resources**:

- [SageMath Documentation](https://doc.sagemath.org/)  
- [CoinGecko API Documentation](https://docs.coingecko.com/v3.0.1/reference/introduction)

**Is it free?**  
Yes, SageMath is open-source and free to use.

**Python libraries / bindings**:

- **SageMath**: Provides extensive functionality for mathematics; install using instructions from the [SageMath Install Guide](https://www.sagemath.org/download.html).  
- **SymPy**: For symbolic mathematics; install via `pip install sympy`.  
- **matplotlib**: For data visualization; install using `pip install matplotlib`.  
- **pandas**: To handle data structures like DataFrames; one can install it with `pip install pandas`.  
- **requests**: For HTTP requests to fetch data from APIs; can be installed via `pip install requests`.  
- **SQLite3 or csv**: For data storage, both of which are part of the Python standard library.

### **SAP BusinessObjects**

**Title**: Real-time Bitcoin Data Analysis using SAP BusinessObjects

**Difficulty**: 1 (easy)

**Description**

This project will introduce students to SAP BusinessObjects, a suite of front-end applications that allow business users to view, sort, and analyze business intelligence data. In this project, students will focus on using SAP BusinessObjects alongside basic Python packages to implement a real-time data ingestion system for Bitcoin price analysis. The goal is to showcase the capabilities of SAP BusinessObjects in integrating and visualizing Bitcoin time-series data, providing students with an understanding of its basic functionalities without overwhelming them with complex setups.

**Describe Technology**

- *SAP BusinessObjects*: A comprehensive business intelligence suite that offers tools for reporting, data visualization, and analytics. It is designed to take data from various sources and present it in a user-friendly graph, chart, or report format.  
- Core functionalities include:  
  - **Reporting**: Create pixel-perfect, rich reports that cater to diverse business needs.  
  - **Dashboards**: Build interactive dashboards for a real-time view of business performance.  
  - **Data Visualization**: Utilize intuitive visualizations to discover insights and make informed decisions.  
  - **Ad-hoc Queries**: Generate insights on demand with self-service querying tools.

**Describe the Project**

- **Objective**: Use SAP BusinessObjects to visualize real-time Bitcoin prices and conduct a basic time-series analysis.  
- **Steps Involved**:  
  1. **Data Ingestion with Python**: Utilize Python to fetch real-time Bitcoin data from a public API such as CoinGecko.  
  2. **Data Storage**: Store the fetched data in a simple CSV format which SAP BusinessObjects can access.  
  3. **Connect SAP BusinessObjects**: Import the stored Bitcoin data into SAP BusinessObjects for reporting and visualization.  
  4. **Create Reports and Dashboards**: Use the reporting and dashboard features of SAP BusinessObjects to visualize Bitcoin price trends.  
  5. **Basic Time-Series Analysis**: Implement a simple time-series analysis to identify and visualize patterns in the Bitcoin data over time, such as moving averages.  
- **Outcome**: Understand how SAP BusinessObjects can be leveraged for real-time data integration and visualization, providing insights into Bitcoin price movements efficiently.

**Useful Resources**

- [SAP BusinessObjects Overview](https://www.sap.com/products/technology-platform/bi-platform.html)  
- [Getting Started with SAP BusinessObjects](https://help.sap.com/doc/85513bb7cec348c8ad353cab52e87822/4.3.2/en-US/webi43sp2_getting_started_en.pdf)  
- [CoinGecko API Documentation](https://docs.coingecko.com/v3.0.1/reference/introduction)

**Is it Free?**

SAP BusinessObjects is a commercial product and requires a valid license for use. However, students may use a trial version or access the software through institutional licensing provided by their college or university.

**Python Libraries / Bindings**

- *requests*: For making HTTP requests to fetch real-time Bitcoin data.  
- *pandas*: For manipulating and storing data in a CSV format suitable for SAP BusinessObjects.  
- *matplotlib/plotly*: For preliminary data visualization (if needed) before importing to SAP BusinessObjects.

### **Scikit-learn**

**Title**: Time Series Analysis of Bitcoin Prices Using Scikit-learn

**Difficulty**: 1 (easy)

**Description**  
Scikit-learn is a widely used Python library for machine learning and data analysis. It provides simple and efficient tools for data mining and analysis, making it accessible for beginners and robust enough for advanced users. In this project, students will learn to utilize Scikit-learn to perform basic time series analysis on real-time Bitcoin price data, focusing on data preprocessing and model training strategies.

**Describe technology**

- Scikit-learn is an open-source library designed for data analysis and machine learning applications.  
- Provides a range of supervised and unsupervised learning algorithms via a consistent interface.  
- Contains modules for preprocessing, model selection, and evaluation, which are essential for building machine learning pipelines.  
- Offers simplicity and efficiency, paired with documentation and tutorials conducive to learning.

**Describe the project**

- Obtain real-time Bitcoin price data using a public API, e.g., CoinGecko or CoinMarketCap.  
- Preprocess the acquired Bitcoin price data with Scikit-learn: handle missing values, normalize or standardize the data, and split into training and test sets.  
- Implement a simple linear regression model to predict Bitcoin prices at future timestamps using Scikit-learn.  
- Evaluate the model's performance using appropriate metrics, such as mean squared error.  
- Visualize the results using matplotlib to interpret the model's accuracy and insights, focusing on price trends over time.

**Useful resources**

- [Scikit-learn Documentation](https://scikit-learn.org/stable/index.html)  
- [CoinGecko API Documentation](https://docs.coingecko.com/v3.0.1/reference/introduction)  
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)

**Is it free?**  
Yes, Scikit-learn is free to use as it is open source and licensed under the BSD license.

**Python libraries / bindings**

- `scikit-learn`: For data preprocessing and modeling. Install using `pip install scikit-learn`.  
- `pandas`: For data manipulation and handling. Install using `pip install pandas`.  
- `matplotlib`: For data visualization. Install using `pip install matplotlib`.  
- An HTTP library, such as `requests`, is also needed to fetch data from the chosen API. Install using `pip install requests`.

### **SendGrid**

**Title**: Real-Time Bitcoin Data Analysis with SendGrid

**Difficulty**: Medium (2)

**Description**  
This project focuses on utilizing SendGrid, a cloud-based email service provider, to send alerts based on real-time Bitcoin data ingested via public APIs. Students will learn how to integrate SendGrid with Python to automate the process of sending notifications when certain conditions are met within the Bitcoin time series data. This project helps students grasp real-time data ingestion, processing, and email-based notification systems.

**Describe technology**

- **SendGrid**: A popular cloud-based email service that provides reliable transaction and marketing email delivery. It offers API endpoints for sending emails programmatically and allows you to manage contacts, track email delivery, and view analytics. It's highly scalable and used by businesses for email marketing campaigns, transactional emails, and alerts.

**Describe the project**

- Goal: Develop a system using SendGrid to send email alerts whenever the real-time Bitcoin price crosses a predefined threshold.  
- Steps:  
  - Use Python to ingest Bitcoin price data in real-time from a public API (e.g., CoinGecko or CryptoCompare).  
  - Process the data to analyze Bitcoin price trends and identify significant price movements or thresholds.  
  - Implement a decision-making function to trigger SendGrid's email API when Bitcoin prices exceed certain limits.  
  - Configure SendGrid's API to send customized email alerts to specified recipients, including recent Bitcoin price data and analysis results.  
  - Ensure the system can dynamically check for these conditions at regular intervals, demonstrating principles of time-series analysis and asynchronous processing.  
- This project provides a practical approach to understanding real-time data handling and automated alert systems through workflows integrating email notifications.

**Useful resources**

- [SendGrid Official Documentation](https://sendgrid.com/docs/)  
- [CoinGecko API Documentation](https://docs.coingecko.com/v3.0.1/reference/introduction)  
- [Twilio SendGrid API Libraries for Python](https://github.com/sendgrid/sendgrid-python)

**Is it free?**

- SendGrid offers a free tier with a limited number of emails per day. CoinGecko’s API is free for basic usage but has usage limits.

**Python libraries / bindings**

- **sendgrid**: A Python client library for SendGrid, enabling easy interaction with SendGrid's email API. Install it using `pip install sendgrid`.  
- **requests**: A Python library for making HTTP requests to APIs. Useful for interfacing with both the Bitcoin price API and the SendGrid API. Install it using `pip install requests`.  
- **schedule**: A Python library for job scheduling, allowing the automation of periodic data fetching and alert checks. Install it using `pip install schedule`.

### **SHAP**

**Title**: Analyzing Bitcoin Price Data Streams Using SHAP

**Difficulty**: 3 (difficult)

**Description**  
SHAP (SHapley Additive exPlanations) is a game-theoretic approach to interpret the predictions of machine learning models. The main concept behind SHAP values is to fairly distribute the "payout" among the features, considering their contributions to the prediction. SHAP provides a unified framework for feature importance that can be applied to any machine learning model by explaining the output of the model in terms of the inclusion of each feature. By quantifying the impact of each feature on the model's predictions, SHAP values facilitate a deeper understanding of model behavior. This project will explore using SHAP to interpret predictions from a real-time Bitcoin price forecasting model, showcasing SHAP's capabilities in explaining time-series model outputs.

**Describe technology**

- **Overview**: SHAP is centered around distributing contribution values (attributions) fairly across all features concerning each instance, using principles from cooperative game theory. It provides both local and global explanations, offering insights into individual predictions and overall feature impact.  
- **How it Works**:  
  - Shapley values calculate the contribution of each feature by considering the difference in model output with and without the feature, aggregated over all feature combinations.  
  - SHAP values are computed for various machine learning models such as tree-based models, deep learning models, and any model providing prediction probabilities.  
  - SHAP provides visualization tools to assess model explanations, enabling model developers to interpret complex models robustly.

**Describe the project**

- **Objective**: Implement a real-time Bitcoin price prediction system using time series analysis and leverage SHAP to explain the model's predictions.  
- **Steps**:  
  1. **Data Ingestion**: Use a public API, like CoinGecko or CoinMarketCap, to stream real-time Bitcoin price data. Set up a data ingestion pipeline using Python to fetch and preprocess the data for time series analysis.  
  2. **Model Development**: Develop a predictive model using a Python library like statsmodels or scikit-learn to forecast Bitcoin prices. Select a suitable algorithm for time series forecasting, such as ARIMA, LSTM, or Prophet.  
  3. **SHAP Integration**: Integrate SHAP to compute explanation values for the predictions made by the model. This involves creating SHAP summary plots, dependence plots, and waterfall plots to interpret how different features (e.g., previous price, volume) affect the model's predictive performance.  
  4. **Evaluation & Presentation**: Evaluate the interpretability of the model predictions and present your findings in a report, highlighting how SHAP values provide insights into model decision-making processes.

**Useful resources**

- SHAP Documentation: [GitHub Repository](https://github.com/slundberg/shap)  
- Bitcoin Price APIs: [CoinGecko API Documentation](https://docs.coingecko.com/v3.0.1/reference/introduction)  
- Python Visualization with SHAP: [SHAP Visualizations Example](https://shap.readthedocs.io/en/latest/index.html#visualizations)

**Is it free?**  
Yes, SHAP is an open-source library, and the necessary Python libraries like statsmodels, scikit-learn, and visualization libraries (e.g., Matplotlib, Seaborn) are free to use. However, accessing certain real-time data through APIs may have rate limits or associated costs beyond basic usage tiers.

**Python libraries / bindings**

- **SHAP**: Install using pip with `pip install shap`. This library provides tools to explain the outputs of machine learning models.  
- **Statsmodels/Scikit-learn**: Depending on the choice of the model, these libraries are fundamental for statistical modeling and machine learning tasks.  
- **Pandas/Numpy**: General data processing and handling libraries that assist in managing time-series data efficiently.  
- **Matplotlib/Seaborn/Plotly**: Visualization libraries to illustrate SHAP values and time series data trends.

By undertaking this project, students will gain experience in real-time data ingestion, model building for time series forecasting, and the application of SHAP for model interpretability. This provides hands-on exposure to complex data science techniques applicable to financial data analysis and beyond.

### **Sisense**

**Title**: Real-Time Bitcoin Analytics with Sisense

**Difficulty**: 3 (difficult)

**Description**  
This project involves setting up a real-time data ingestion and analytics system for Bitcoin using Sisense, an advanced analytics platform. Students will focus on implementing an architecture that consumes live Bitcoin data from public APIs and transforms it into actionable insights using Sisense's robust data visualization and analytics features. The project requires integrating Sisense with Python for data manipulation and conducting a time series analysis to predict Bitcoin price trends.

**Describe technology**  
Sisense is a powerful business intelligence (BI) tool designed to handle and analyze large datasets with ease. It provides end-to-end solutions for data preparation, analysis, and visualization. Key features include:

- Seamless integration with various data sources, including APIs for real-time data streaming.  
- Robust data preparation tools to clean and transform raw data.  
- Advanced analytics capabilities to handle complex data relationships and calculations.  
- Customizable dashboards and visualizations for data storytelling and insight sharing.  
- Extensible through REST APIs and SDKs for further customization and integration with other tools.

**Describe the project**  
The project is divided into several critical phases:

1. **Data Ingestion**:  
     
   - Set up a connection to a real-time Bitcoin price API, such as CoinGecko, using a Python script.  
   - Capture this data constantly and prepare it to be fed into Sisense.

   

2. **Data Integration and Preparation**:  
     
   - Use Sisense's data preparation tools to clean and structure the raw Bitcoin data.  
   - Create relationships between datasets to enrich the data for more comprehensive analysis.

   

3. **Time Series Analysis**:  
     
   - Perform a time series analysis on the historical Bitcoin price data using Python’s `pandas` and `statsmodels` libraries.  
   - Build predictive models to forecast future Bitcoin prices.

   

4. **Visualization and Insights**:  
     
   - Design dashboards in Sisense to visualize current Bitcoin prices, historical trends, and future predictions.  
   - Implement features for users to interactively explore data, such as filtering by date ranges and comparing forecasted prices to historical data.

This project provides experience in integrating Sisense with real-time data systems, performing time series analysis, and creating insightful BI dashboards.

**Useful resources**

- [Sisense Documentation](https://docs.sisense.com/main/Home.htm)  
- [CoinGecko API Documentation](https://docs.coingecko.com/v3.0.1/reference/introduction)  
- [Pandas Documentation](https://pandas.pydata.org/docs/)  
- [Statsmodels Documentation](https://www.statsmodels.org/stable/index.html)

**Is it free?**  
Sisense offers a free trial, but typically requires a paid subscription for ongoing use. Check with Sisense for academic pricing or special educational access programs.

**Python libraries / bindings**

- **Requests**: To interact with and fetch data from Bitcoin APIs. Install it using `pip install requests`.  
- **Pandas**: For data manipulation and preparation tasks. Install it using `pip install pandas`.  
- **Statsmodels**: For performing advanced statistical analyses, including time series forecasting. Install it using `pip install statsmodels`.  
- **Sisense REST API**: To automate processes or extend functionality within Sisense using Python.

### **Snowflake**

**Title**: Real-time Bitcoin Data Analysis with Snowflake

**Difficulty**: 2 (medium)

**Description**:  
This project explores using Snowflake, a cloud-based data warehousing service, to ingest, process, and analyze real-time Bitcoin price data. Snowflake's scalable architecture and robust SQL capabilities make it an excellent choice for handling big data and performing complex queries. The project will introduce students to Snowflake's core functionalities, including its data loading mechanisms, seamless integration with cloud storage, and advanced querying features. This will be paired with Python packages for auxiliary data processing tasks.

**Describe technology**:

- **Snowflake**: A cloud data platform that offers data warehousing, SQL analytics, and data integration services. Its key features include an architecture that separates storage and compute for scalability, support for structured and semi-structured data, and extensive data sharing capabilities.  
- **Core Concepts**:  
  - Warehouses: Resources for computation that allow for flexibility in scaling.  
  - Databases and Schemas: Organizational units for storing data.  
  - Tables and Views: Structures for managing data access and queries.  
  - Snowpipe: A continuous data ingestion service for loading data in near real-time.

**Describe the project**:

- **Objective**: Implement a system using Snowflake to ingest, store, and perform time-series analysis on Bitcoin price data.  
- **Steps**:  
  1. **Data Ingestion**:  
     - Use the Snowpipe service to set up a continuous ingestion pipeline.  
     - Pull Bitcoin price data from a public API, such as CoinGecko.  
     - Store the JSON data in cloud storage (e.g., AWS S3, Google Cloud Storage).  
  2. **Data Processing**:  
     - Load the data into Snowflake tables for further analysis.  
     - Transform raw data into a structured table format using SQL operations within Snowflake.  
  3. **Time-Series Analysis**:  
     - Use Snowflake's SQL capabilities to perform basic time-series analyses, such as moving averages and trendline computations.  
  4. **Visualization**:  
     - Export processed data to a Python environment for visualization using libraries like Matplotlib or Seaborn.  
- **Outcome**: Students will gain experience in setting up a real-time data pipeline and applying time-series analysis using Snowflake and Python.

**Useful resources**:

- Snowflake Documentation: [docs.snowflake.com](https://docs.snowflake.com)  
- CoinGecko API Documentation: [CoinGecko API Documentation](https://docs.coingecko.com/v3.0.1/reference/introduction)  
- Python Matplotlib Documentation: [matplotlib.org](https://matplotlib.org)  
- Python Seaborn Documentation: [seaborn.pydata.org](https://seaborn.pydata.org)

**Is it free?**:  
Snowflake offers a free trial with credits that can be used to explore its services, but students will need to be careful to manage resources to avoid any additional charges. Access to cloud storage may also incur additional costs.

**Python libraries / bindings**:

- **snowflake-connector-python**: Python connector to perform standard database operations in Snowflake. Install with `pip install snowflake-connector-python`.  
- **requests**: Library for making HTTP requests to retrieve Bitcoin price data from APIs. Install with `pip install requests`.  
- **matplotlib**: Used for creating static, animated, and interactive visualizations in Python. Install with `pip install matplotlib`.  
- **pandas**: A library for data manipulation and analysis, ideal for preprocessing and cleaning data. Install with `pip install pandas`.

Through this project, students will have the opportunity to integrate cloud-based data warehousing with real-time data ingestion and gain hands-on experience with Snowflake's advanced capabilities for data analysis.

### **Socket.IO**

**Title**: Real-time Bitcoin Data Processing with Socket.IO

**Difficulty**: 3 (Difficult)

**Description**:  
In this project, you will leverage Socket.IO to design and implement a real-time data ingestion and processing system tailored for Bitcoin price data. Socket.IO is a JavaScript library for real-time web applications, in particular enabling bidirectional communication between web clients and servers via websockets, which is ideal for handling streaming data. Students will develop a Python-based application to establish a live connection to a public Bitcoin price API to ingest data continuously and apply time series analysis techniques.

**Describe technology**:

- **Socket.IO**:  
  - Socket.IO provides a seamless and uncomplicated interface for WebSockets, allowing real-time bi-directional communication.  
  - It is composed of two parts: a client-side library that runs in the browser, and a server-side library for Node.js.  
  - The library abstracts away the differences in websockets implementation, allowing you to focus on the application logic.  
  - Key features include automatic reconnection on lost connections and immediate cross-platform support including mobile devices.

**Describe the project**:

- The project involves setting up a Socket.IO server in Python to ingest real-time Bitcoin price data from a websocket-supported API like CoinGecko or similar.  
- Design a client component in JavaScript/Python that establishes a connection to your Python server.  
- Implement a mechanism for the server to process incoming raw data, focusing on improving data quality and structure.  
- Apply time series analysis techniques (such as moving average, exponential smoothing, or anomaly detection) directly on the streaming data.  
- Ensure processed data is stored in a time-series database like InfluxDB or TimescaleDB for further querying and analysis.  
- Visualize the processed real-time data on a dashboard with dynamic charts using a visualization library like D3.js or Plotly.

**Useful resources**:

- [Socket.IO Official Documentation](https://socket.io/)  
- [A Beginner’s Guide to WebSockets](https://www.html5rocks.com/en/tutorials/websockets/basics/)  
- [CoinGecko API Documentation](https://docs.coingecko.com/v3.0.1/reference/introduction)  
- [InfluxDB Documentation](https://docs.influxdata.com/influxdb/)  
- [Plotly Python Graphing Library](https://plotly.com/python/)

**Is it free?**  
Yes, Socket.IO is open source, and you can freely use it under the MIT license. APIs like CoinCap typically offer free tiers with limits on API calls.

**Python libraries / bindings**:

- **socketio**: Python library for Socket.IO server implementation (`pip install python-socketio`)  
- **requests**: For making HTTP requests (`pip install requests`)  
- **pandas**: Data manipulation and analysis (`pip install pandas`)  
- **numpy**: Numerics and mathematical functions (`pip install numpy`)  
- **matplotlib/plotly**: Visualization libraries for plotting data (`pip install matplotlib` or `pip install plotly`)  
- **influxdb-client**: For storing time-series data in InfluxDB (`pip install influxdb-client`)

By the end of this project, students will gain a comprehensive understanding of how to utilize Socket.IO for real-time data processing and apply time-series analytical techniques on streaming data, framing a robust background to extend into other applications or data types.

### **spaCy**

**Title:** Real-Time Bitcoin Sentiment Analysis with spaCy and Selenium​

**Difficulty:** 3 (Difficult)​

**Description:** This project involves utilizing **spaCy**, an advanced natural language processing (NLP) library in Python, to perform real-time sentiment analysis on Bitcoin-related tweets. By integrating **Selenium** for web scraping, students will collect live Twitter data without relying on the Twitter API, process the textual content using spaCy, and analyze the correlation between public sentiment and Bitcoin price fluctuations over time.​

**Describe Technology:**

* **spaCy:**  
  * A powerful, open-source NLP library designed for efficient and scalable text processing.​  
  * Provides functionalities such as tokenization, part-of-speech tagging, named entity recognition (NER), and more.​  
  * Supports integration with deep learning frameworks like TensorFlow and PyTorch.​  
* **Selenium:**  
  * A browser automation tool that enables programmatic control of web browsers.  
  * Useful for web scraping dynamic content that traditional scraping tools might not handle effectively.​

**Describe the Project:**

**Objective:** To develop a system that scrapes real-time Bitcoin-related tweets using Selenium, processes the textual data with spaCy for sentiment analysis, and examines the correlation between public sentiment and Bitcoin price movements.​

**Steps:**

1. **Data Ingestion:**  
   * Utilize Selenium to automate the scraping of real-time tweets containing Bitcoin-related keywords (e.g., "Bitcoin", "BTC").​  
   * Implement a scraping mechanism based on the [selenium-twitter-scraper](https://github.com/godkingjay/selenium-twitter-scraper) GitHub repository, which allows for scraping tweets from user profiles, hashtags, or search queries without requiring Twitter API access.​  
2. **Data Preprocessing:**  
   * Clean and preprocess the scraped tweet text using spaCy, including tokenization, stop-word removal, and lemmatization.​  
   * Perform Named Entity Recognition (NER) to identify mentions of cryptocurrencies and related entities.​  
3. **Sentiment Analysis:**  
   * Integrate a sentiment analysis tool, such as VADER or TextBlob, with spaCy to assign sentiment scores to each tweet.​  
   * Categorize tweets into positive, negative, or neutral sentiments based on the assigned scores.​  
4. **Correlation with Bitcoin Price:**  
   * Fetch real-time Bitcoin price data from a public API (e.g., CoinGecko).​  
   * Store both sentiment scores and Bitcoin pricing data in a structured format (e.g., a pandas DataFrame).​  
   * Conduct time series analysis to explore the relationship between public sentiment and Bitcoin price fluctuations.​  
5. **Visualization:**  
   * Create visual representations, such as line plots or scatter plots, to depict sentiment trends alongside Bitcoin price movements over time.​

**Useful Resources:**

* [spaCy Documentation](https://spacy.io/usage)  
* [Selenium with Python Documentation](https://selenium-python.readthedocs.io/)  
* [selenium-twitter-scraper GitHub Repository](https://github.com/godkingjay/selenium-twitter-scraper)  
* [CoinGecko API Documentation](https://docs.coingecko.com/v3.0.1/reference/introduction)

**Is it Free?**

* **spaCy** is open-source and free to use.​  
* **Selenium** is also open-source and free.​  
* Accessing real-time Bitcoin price data through public APIs like CoinGecko is free, though some services may have rate limits or usage restrictions.​

**Python Libraries / Bindings:**

* `spaCy`: Install via `pip install spacy`.​  
* `Selenium`: Install via `pip install selenium`. Requires a web driver compatible with your browser (e.g., ChromeDriver for Chrome).​  
* `pandas`: For data manipulation and analysis; install via `pip install pandas`.​  
* `matplotlib` or `seaborn`: For data visualization; install via `pip install matplotlib seaborn`.​  
* `requests`: To fetch data from APIs; install via `pip install requests`.​  
* `vaderSentiment` or `TextBlob`: For sentiment analysis; install via `pip install vaderSentiment` or `pip install textblob`.​

This project offers a comprehensive experience in web scraping, natural language processing, sentiment analysis, and time series analysis within the context of cryptocurrency markets.​

### **Spark SQL**

**Title**: Implementing Real-Time Bitcoin Price Analysis with Spark SQL  
**Difficulty**: 2 (Medium)

**Description**  
Spark SQL is a module of Apache Spark, designed for structured data processing. It allows users to run SQL queries on DataFrames and is highly advantageous for handling big data. One of its primary features is the ability to perform transformations and actions on large datasets using SQL-like syntax.

**Describe technology**

- Spark SQL enables seamless integration with Spark's core capabilities.  
- It provides API support in Python, Java, Scala, and R.  
- It leverages in-memory data processing for faster analytics.  
- Supports Hive metastore to manage metadata about tables and databases.  
- Provides a simple way to execute SQL queries using Spark’s scalable data processing engine.

**Describe the project**  
This easy-level project focuses on building a basic real-time data ingestion and processing pipeline using Spark SQL to analyze Bitcoin prices. The goal is to retrieve Bitcoin price data from a publicly available API and perform a time series analysis using Spark SQL.

- **Ingest**: Use Python to call a Bitcoin price API (e.g., CoinGecko) and ingest data in real-time.  
- **Store**: Store the incoming data in-memory using Spark DataFrames.  
- **Query**: Implement Spark SQL to perform time-series analysis.  
  - Example Queries:  
    - Calculate average price over the past week.  
    - Identify daily maximum and minimum prices.  
  - Use Spark SQL functions like `avg`, `max`, and `min` to perform these analyses easily.  
- **Visualize**: Optionally, use basic Python plotting libraries like Matplotlib or Plotly to visualize the results of your analysis.

**Useful resources**

- [Apache Spark SQL Documentation](https://spark.apache.org/sql/)  
- [CoinGecko API Documentation](https://docs.coingecko.com/v3.0.1/reference/introduction)  
- [Spark SQL Programming Guide](https://spark.apache.org/docs/latest/sql-programming-guide.html)

**Is it free?**  
Yes, Apache Spark is open-source software, and you can deploy it locally or on a cloud provider's free tier.

**Python libraries / bindings**

- **pyspark**: The official Python library for Spark. To work with Spark SQL in Python, you'll use the `pyspark.sql` module, which allows you to interact with Spark DataFrames and execute SQL queries.  
  - Installation: `pip install pyspark`  
- **requests**: A library for making HTTP requests to call the Bitcoin price API.  
  - Installation: `pip install requests`  
- **pandas**: Optional, for intermediate data manipulation if needed.  
  - Installation: `pip install pandas`  
- **matplotlib/plotly**: Optional, for plotting the time series analysis results.  
  - Installation: `pip install matplotlib` or `pip install plotly`

By completing this project, students will gain hands-on experience with Spark SQL, as well as foundational skills in processing and analyzing real-time data using Python.

### **SQLAlchemy**

**Title**: Real-time Bitcoin Analysis with SQLAlchemy

**Difficulty**: 2

**Description**  
Students will explore the use of SQLAlchemy for managing and accessing large datasets with the focus on real-time Bitcoin price analysis. SQLAlchemy is a SQL toolkit and Object-Relational Mapping (ORM) library for Python that provides a full suite of well-known enterprise-level persistence patterns, designed for efficient and high-performing database access. Know the basics of SQLAlchemy, its ORM capabilities, and how it facilitates database interactions within Python applications.

**Describe technology**

- **SQLAlchemy Core**: Offers a schema-centric SQL abstraction layer and a full suite of enterprise-level persistence patterns.  
- **Object Relational Mapper (ORM)**: Allows the definition of classes mapped to database tables; relations can be established between classes and SQL queries can be built using these objects.  
- **Engine Interface**: Manages connections to the database, supporting multiple databases (e.g., SQLite, PostgreSQL, MySQL).  
- **Session Management**: Provides a factory for creating new sessions to interact with the database, supporting transaction management and workflow control.

**Describe the project**  
This project involves creating a real-time data processing system to ingest Bitcoin prices and conduct time series analysis:

- **Data Ingestion**: Use a public API such as CoinGecko or CryptoCompare to fetch real-time Bitcoin prices. This data will be ingested at regular intervals and stored using SQLAlchemy, providing ORM capabilities for easy data manipulation.  
- **Database Modeling**: Design database schema using SQLAlchemy ORM to handle time series data efficiently. This includes defining tables for storing the Bitcoin price data and any additional metadata required for analysis.  
- **Data Processing**: Implement a data processing routine to calculate key metrics over time, such as moving averages or price volatility, utilizing Python standard libraries.  
- **Visualization**: Use a library like Matplotlib or Seaborn to visualize the Bitcoin prices and derived metrics over time, showcasing SQLAlchemy's efficacy in handling real-time data.  
- **Analysis Demonstration**: Write Python scripts using SQLAlchemy to retrieve and process the data, demonstrating real-time analysis and visualization.

**Useful resources**

- [SQLAlchemy Official Documentation](https://docs.sqlalchemy.org/)  
- [CoinGecko API Documentation](https://docs.coingecko.com/v3.0.1/reference/introduction)  
- [Matplotlib Official Documentation](https://matplotlib.org/stable/users/index.html)  
- [Seaborn Documentation](https://seaborn.pydata.org/)

**Is it free?**  
Yes, SQLAlchemy is open-source and free to use. You might need an API key for certain usage levels with data providers like CoinGecko or CryptoCompare, though these often offer free tiers.

**Python libraries / bindings**

- **SQLAlchemy**: Use it for database interactions with ORM features. Install via `pip install sqlalchemy`.  
- **Requests**: Use it to fetch data from the Bitcoin price API. Install with `pip install requests`.  
- **Pandas**: Use it for data manipulation and processing. Install with `pip install pandas`.  
- **Matplotlib/Seaborn**: Use it for visualizing data insights. Install with `pip install matplotlib seaborn`.

###   **SQLite**

**Title**: Time Series Analysis of Bitcoin Prices using SQLite

**Difficulty**: 2 (medium)

**Description**  
SQLite is a software library that provides a relational database management system. It is self-contained, serverless, and requires a minimal setup, making it ideal for embedding into other applications. It supports SQL queries, which makes it suitable for handling structured data efficiently. SQLite is used globally in both simple applications and complex systems due to its simplicity and flexibility.

**Describe technology**

- **Lightweight and Efficient**: SQLite is a compact database solution integrated into the application software, eliminating the need for a separate server process.  
- **SQL Support**: It fully supports SQL standards, enabling complex queries and data manipulation.  
- **Zero Configuration**: SQLite does not require any installation or configuration, facilitating a hassle-free setup.  
- **ACID Compliance**: Transactions in SQLite are compliant with the ACID properties, ensuring reliable and safe data manipulation even during system failures.  
- **Platform Independent**: Available across multiple operating systems without the need for extensive configuration, making it versatile for various development environments.

**Describe the project**  
In this project, students are required to build a real-time data ingestion and processing system using SQLite to analyze Bitcoin price fluctuations. The focus will be on implementing this system in Python:

1. **Data Ingestion**:  
     
   - Fetch real-time Bitcoin price data from a public API such as CoinGecko or CoinMarketCap.  
   - Store the incoming data into an SQLite database, emphasizing the use of SQL to create and manage the database schema.

   

2. **Data Processing and Analysis**:  
     
   - Query the SQLite database to extract relevant Bitcoin pricing information.  
   - Perform time series analysis, which may include calculating moving averages, rate of change, or volatility metrics.  
   - Utilize Python libraries such as Pandas to support data manipulation and Matplotlib for data visualization of price trends.

**Useful resources**

- [CoinGecko API Documentation](https://docs.coingecko.com/v3.0.1/reference/introduction)  
- [SQLite Official Documentation](https://sqlite.org/docs.html)  
- [SQLite Python Tutorial](https://www.tutorialspoint.com/sqlite/sqlite_python.htm)  
- [Pandas Time Series Tutorial](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html)  
- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/index.html)

**Is it free?**  
Yes, SQLite is open-source and freely available for both personal and commercial use.

**Python libraries / bindings**

- **sqlite3**: Built-in Python library that provides an interface for interacting with SQLite databases.  
- **pandas**: Library used for data manipulation and analysis; helps with loading data from SQLite and performing time series analysis.  
- **requests**: Library to handle API requests for fetching real-time Bitcoin pricing data.  
- **matplotlib**: Plotting library used to visualize the time series data and analysis results.

### **Stable-Baselines3**

**Title:** Real-Time Bitcoin Price Trend Analysis using Stable-Baselines3​

**Difficulty:** 2 (Medium)​

**Description:** Stable-Baselines3 is a set of reliable implementations of reinforcement learning (RL) algorithms in Python, designed for performance and ease of use. This project focuses on applying RL methods to time series forecasting, specifically predicting Bitcoin price trends. By leveraging real-time data ingestion techniques and utilizing Gymnasium—a modern replacement for OpenAI's deprecated Gym library—students will develop a system that analyzes and predicts Bitcoin price movements.​

**Technology Overview:**

* **Stable-Baselines3:**  
  * Offers modular and user-friendly implementations of various RL algorithms.​  
  * Facilitates quick testing and iteration with different RL techniques.​  
  * Easily integrates with other open-source libraries, enhancing its capacity for learning new environments.​  
* **Gymnasium:**  
  * A modern, open-source library for developing and comparing RL algorithms, succeeding the deprecated OpenAI Gym.​  
  * Provides a standardized API for creating custom RL environments.​  
  * Compatible with Stable-Baselines3, enabling seamless integration.​

**Project Outline:**

1. **Data Ingestion:**  
   * Utilize a public API, such as CoinGecko, to collect real-time Bitcoin price data.​  
   * Preprocess the data for analysis, including handling missing values and normalizing features.​  
2. **Environment Creation:**  
   * Define a custom environment using Gymnasium to represent the state-action-reward setup pertinent to Bitcoin price movements.​  
   * Ensure the environment adheres to Gymnasium's API standards for compatibility with Stable-Baselines3.​  
3. **RL Model Training:**  
   * Develop and train a reinforcement learning model using Stable-Baselines3, configuring it to learn from historical Bitcoin data.​  
   * Experiment with different RL algorithms (e.g., DQN, PPO) to identify the most effective approach.​  
4. **Prediction and Analysis:**  
   * Utilize the trained model to predict future Bitcoin price trends.​  
   * Analyze the model's performance against actual market data, employing metrics such as mean squared error.​  
5. **Evaluation:**  
   * Customize reward functions based on performance metrics to refine prediction accuracy.​  
   * Assess the robustness of the model under different market conditions.​

**Useful Resources:**

* [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)​  
* [Gymnasium Documentation](https://gymnasium.farama.org/)​  
* [CoinGecko API Documentation](https://docs.coingecko.com/v3.0.1/reference/introduction)

**Is it Free?**

Yes, both Stable-Baselines3 and Gymnasium are open-source and free to use. Public APIs like CoinGecko offer free access to fundamental endpoints, though they may have limitations on request rates.​

**Python Libraries / Dependencies:**

* `stable_baselines3`: Provides implementations of RL algorithms. Install using `pip install stable-baselines3`.​  
* `gymnasium`: Required for defining environments for RL models. Install using `pip install gymnasium`.​  
* `requests`: For accessing real-time Bitcoin price data from public APIs. Install using `pip install requests`.​  
* `pandas`: For data manipulation and preprocessing. Install using `pip install pandas`.​

This project offers a practical introduction to applying reinforcement learning techniques to financial time series data, providing valuable insights into the dynamics of cryptocurrency markets.​

### **statsmodels**

**Title**: Analyzing Bitcoin Trends Using Statsmodels

**Difficulty**: 2 (medium)

**Description**:  
In this project, students will utilize the `statsmodels` library to perform time series analysis on real-time Bitcoin data. `statsmodels` is a powerful Python library for statistical modeling and econometrics. This project will guide students through the process of ingesting Bitcoin price data from an API, processing it, and then using `statsmodels` to perform time series analysis. The aim is to forecast Bitcoin price movements and provide insights into its volatility over time.

**Describe Technology**:

- **statsmodels**: An open-source Python package that provides classes and functions for estimating many different statistical models, as well as for conducting statistical tests and data exploration.  
  - Key functionalities include linear regression, ANOVA, ARIMA, time series analysis, and hypothesis testing.  
  - It integrates well with `pandas` for handling and manipulating data and `matplotlib` for visualizing results.  
  - Example functionalities:  
    - Fitting a linear regression model: `OLS(y, X).fit()`  
    - Conducting an ARIMA time series analysis: `ARIMA(y, order=(p,d,q)).fit()`

**Describe the Project**:

- **Objective**: Design and implement a system using `statsmodels` to perform time series forecasting on real-time Bitcoin prices.  
- **Steps**:  
  1. **Data Ingestion**: Use a public API (e.g., CoinGecko API) to fetch real-time Bitcoin price data every few minutes.  
  2. **Data Storage and Preprocessing**: Store the collected data in a `pandas` DataFrame for easy manipulation. Handle missing data, normalize the prices, and resample the data to create a uniform time series.  
  3. **Time Series Analysis**:  
     - Explore the time series using `statsmodels` to identify trends, seasonality, and noise.  
     - Use an appropriate time series model such as ARIMA to fit the historical Bitcoin price data.  
     - Forecast future Bitcoin prices and evaluate the model's accuracy.  
  4. **Visualization**: Plot the original time series, along with the model's fitted values and forecasts, using `matplotlib`.  
- **Outcome**: By the end of this project, students will have a functional prototype that ingests real-time Bitcoin data, processes it, and uses statistical models to make forecasts and understand historical trends.

**Useful Resources**:

- [statsmodels Documentation](https://www.statsmodels.org/)  
- [CoinGecko API Documentation](https://docs.coingecko.com/v3.0.1/reference/introduction)

**Is it free?**:  
Yes, both `statsmodels` and the CoinGecko API offer free access for educational purposes. The required Python libraries are also open-source, making this project cost-effective for students.

**Python Libraries / Bindings**:

- `statsmodels`: Main library for statistical modeling. Install using `pip install statsmodels`.  
- `pandas`: Used for data manipulation and analysis. Install using `pip install pandas`.  
- `matplotlib`: Essential for plotting data. Install using `pip install matplotlib`.  
- `requests`: To make HTTP requests to fetch data from the API. Install using `pip install requests`.

### **Streamlit**

**Title**: Real-Time Bitcoin Price Monitoring with Streamlit

**Difficulty**: 3 (difficult)

**Description**: This project focuses on building an interactive dashboard to monitor real-time Bitcoin prices using Streamlit, a popular Python library for creating web applications with minimal effort. Streamlit is incredibly useful for data scientists looking to quickly develop and deploy web-based data applications without needing advanced web development skills. The project's main challenge is ingesting, processing, and visualizing real-time data with an emphasis on time series analysis. Students will be required to use Python to fetch live Bitcoin data from an API, process this data to identify trends and patterns, and present the insights using an interactive Streamlit interface.

**Describe technology**:

- **Streamlit**: A powerful open-source app framework in Python, Streamlit turns Python data scripts into shareable web apps. It's particularly suited for data science applications because it allows for the quick development of interactive dashboards and visualizations without requiring HTML, CSS, or JavaScript.  
  - Key features include:  
    - **Easy-to-use APIs**: Create complex UIs with simple Python scripts.  
    - **Instant updates**: Modify the Python code, and the Streamlit app updates in real-time.  
    - **Data-driven components**: Integrate tables, charts, and plots with minimal boilerplate code.  
    - **Seamless integration**: Leverage existing Python libraries for data visualization like Matplotlib, Plotly, and Altair.

**Describe the project**:

- **Objective**: Develop a real-time dashboard application to monitor Bitcoin prices and analyze time series data using Streamlit.  
- **Implementation steps**:  
  1. **Data Ingestion**: Use Python to fetch live data from a Bitcoin price API such as CoinGecko or Alpha Vantage.  
  2. **Data Processing**: Preprocess the raw data for use in visualizations; this includes time series transformation, calculating moving averages, and detecting anomalies.  
  3. **Time Series Analysis**: Implement techniques for forecasting and identifying patterns. Utilize libraries such as Prophet or statsmodels to perform these analyses.  
  4. **Building the App**:  
     - Design the Streamlit interface to display current prices, historical trends, and prediction models.  
     - Add interactive widgets like sliders, dropdowns, and buttons to filter data by time periods, view different metrics, and customize visualizations.  
  5. **Visualizations**: Develop dynamic charts (using libraries like Plotly or Matplotlib) integrated within Streamlit to visualize the processed data and analysis results.  
- **Outcome**: By completing the project, students will gain experience in creating interactive applications that monitor real-time data, enhancing their skills in data science, Python programming, and web app development.

**Useful resources**:

- [Streamlit Documentation](https://docs.streamlit.io/library/get-started)  
- [CoinGecko API Documentation](https://docs.coingecko.com/v3.0.1/reference/introduction)  
- [Prophet Documentation](https://facebook.github.io/prophet/docs/quick_start.html)  
- [Statsmodels Documentation](https://www.statsmodels.org/stable/index.html)

**Is it free?**

- **Yes**: Streamlit is open-source and free to use for developing applications. For deploying apps, Streamlit Cloud offers a free tier with some limitations.

**Python libraries / bindings**:

- **Streamlit**: (Install with `pip install streamlit`) \- Core framework for building interactive web apps.  
- **Requests**: (Install with `pip install requests`) \- Used to fetch data from APIs.  
- **Pandas**: (Install with `pip install pandas`) \- Essential for data manipulation and analysis.  
- **Plotly/Matplotlib/Altair**: (Install with `pip install plotly`, `matplotlib`, or `altair`) \- For creating interactive visualizations.  
- **Prophet/Statsmodels**: (Install with `pip install prophet` or `statsmodels`) \- To conduct time series analysis and forecasting.

### **Streamz**

**Title:** Real-Time Bitcoin Price Trend Analysis using Streamz​

**Difficulty:** 2 (Medium)​

**Description:** Streamz is a Python library that facilitates the creation of pipelines to manage continuous data streams. It allows for the construction of both simple and complex pipelines involving branching, joining, flow control, feedback, and back pressure. In this project, students will utilize Streamz to ingest real-time Bitcoin price data and perform time series analysis to detect trends and patterns.​

**Technology Overview:**

* **Streamz:**  
  * Enables the construction of pipelines for continuous data streams.​  
  * Supports integration with Pandas for streaming operations on continuous tabular data.​  
  * Allows for the development of complex pipelines with features like branching and flow control.​

**Project Outline:**

1. **Data Ingestion:**  
   * Utilize a public API, such as CoinGecko, to collect real-time Bitcoin price data.​  
   * Implement a Streamz source to ingest this data continuously.​  
2. **Data Processing:**  
   * Use Streamz to create a pipeline that processes the incoming data.​  
   * Integrate with Pandas to handle data in a tabular format, facilitating time series analysis.​  
3. **Time Series Analysis:**  
   * Apply moving averages and other statistical methods to identify trends and patterns in the Bitcoin price data.​  
   * Utilize Streamz's capabilities to handle real-time data processing and analysis.​  
4. **Visualization:**  
   * Implement real-time plotting of Bitcoin price trends using libraries such as Matplotlib or Plotly.​  
   * Ensure that the visualizations update dynamically as new data is ingested.​  
5. **Alert System:**  
   * Set up a system to send alerts when significant price movements are detected.​  
   * Utilize Streamz's flow control features to manage alert conditions and notifications.​

**Useful Resources:**

* [Streamz Documentation](https://streamz.readthedocs.io/en/latest/)​  
* [CoinGecko API Documentation](https://docs.coingecko.com/v3.0.1/reference/introduction)​  
* [Pandas Documentation](https://pandas.pydata.org/pandas-docs/stable/)​

**Is it Free?**

Yes, Streamz is an open-source library and free to use. Public APIs like CoinGecko offer free access to cryptocurrency data, though they may have rate limits.​

**Python Libraries / Dependencies:**

* `streamz`: For building and managing data pipelines. Install using `pip install streamz`.​  
* `pandas`: For data manipulation and analysis. Install using `pip install pandas`.​  
* `requests`: For accessing real-time Bitcoin price data from public APIs. Install using `pip install requests`.​  
* `matplotlib` or `plotly`: For data visualization. Install using `pip install matplotlib` or `pip install plotly`.​

This project offers a practical introduction to real-time data processing and time series analysis using Streamz, providing valuable insights into the dynamics of cryptocurrency markets.​

### **SWE-agent**

**Title:** Automated Bitcoin Price Alert System with SWE-Agent​

**Difficulty:** 2 (Medium)​

**Description:** In this project, students will utilize SWE-Agent, an AI-driven system designed to assist in software engineering tasks, to develop an automated Bitcoin price alert system. The project involves setting up SWE-Agent to monitor real-time Bitcoin prices and send notifications when specific price thresholds are reached. This provides a practical introduction to integrating AI agents with cryptocurrency data monitoring.​

**Technology Overview:**

* **SWE-Agent:**  
  * Transforms large language models (LLMs) into autonomous software engineering agents capable of interacting with codebases.​  
  * Utilizes Agent-Computer Interfaces (ACIs) to perform tasks such as browsing repositories, editing code, and executing commands.​  
  * Simplifies the automation of software tasks, making it accessible for users with basic programming knowledge.​

**Project Outline:**

1. **Setup and Configuration:**  
   * Install SWE-Agent by following the official [Getting Started Guide](https://swe-agent.com/).​  
   * Configure SWE-Agent to use a suitable LLM (e.g., GPT-4) by setting the appropriate API keys.​  
2. **Data Ingestion:**  
   * Use a public API, such as CoinGecko, to fetch real-time Bitcoin price data.​  
   * Implement a Python script that retrieves the current Bitcoin price at regular intervals.​  
3. **Price Monitoring and Alert System:**  
   * Develop a function that checks if the Bitcoin price crosses predefined thresholds (e.g., a 5% increase or decrease).​  
   * Configure SWE-Agent to send email notifications or log alerts when these thresholds are met.​  
4. **Automation:**  
   * Set up a scheduling mechanism to run the price monitoring script at regular intervals (e.g., every 10 minutes).​  
   * Ensure that SWE-Agent operates autonomously, requiring minimal manual intervention.​

**Useful Resources:**

* [SWE-Agent Documentation](https://swe-agent.com/)​  
* [CoinGecko API Documentation](https://docs.coingecko.com/v3.0.1/reference/introduction)  
* [Python `requests` Library Documentation](https://docs.python-requests.org/en/latest/)​

**Is it Free?**

SWE-Agent is an open-source project and free to use. Access to certain LLMs may require subscriptions or incur usage costs. The CoinGecko API provides free access to cryptocurrency data, though with rate limits.​

**Python Libraries / Dependencies:**

* `sweagent`: Core library for deploying and managing SWE-Agent.​  
* `requests`: For making HTTP requests to fetch Bitcoin price data. Install using `pip install requests`.​  
* `schedule`: To assist with running the script at regular intervals. Install using `pip install schedule`.​  
* `smtplib`: For sending email notifications (included in Python's standard library).​

This project offers a practical introduction to using AI agents for automating tasks, specifically in monitoring cryptocurrency prices and sending alerts based on real-time data.​

### **Tableau**

**Title**: Analyze Bitcoin Trends Using Tableau

**Difficulty**: 1 (easy)

**Description**  
Tableau is a powerful data visualization tool used to simplify raw data into easy-to-understand, interactive visual forms. It's popular for helping users see and understand data through natural language queries, trend lines, and other visual reporting functionalities. This project introduces students to the basics of working with Tableau to perform time series analysis on real-time Bitcoin price data.

**Describe technology**

- Tableau enables users to quickly visualize data in an intuitive manner.  
- Key features include interactive dashboards, dynamic sorting, trends and predictive analysis, and varied graph types.  
- Drag-and-drop capability makes developing complex visualizations simple without extensive programming knowledge.  
- It connects to various data sources such as spreadsheets, relational databases, and cloud platforms, allowing flexible data integration.

**Describe the project**  
This easy one-week project involves using Tableau to visualize and analyze the historical price trends of Bitcoin. Students will fetch Bitcoin price data from a public API like CoinGecko using Python, creating a CSV file for later use in Tableau. The tasks include the following steps:

1. Use Python to regularly gather price data from a public Bitcoin API. Parse and store this data in CSV format.  
2. Import the CSV data into Tableau and clean it if necessary (e.g., handle missing values or outliers).  
3. Develop a series of time series visualizations using Tableau:  
   - Create basic line charts demonstrating Bitcoin’s price changes over time.  
   - Use Tableau’s built-in functionalities to calculate moving averages and add trend lines.  
   - Make dashboards to display different visualizations and facilitate easy trend analysis.  
4. Analyze data features such as price dips, peaks, and any noticeable seasonal or periodic patterns.

**Useful resources**

- [Tableau Official Website](https://www.tableau.com/)  
- [Tableau Public](https://public.tableau.com/s/) \- A free platform to visualize data and share insights online.  
- [CoinGecko API Documentation](https://docs.coingecko.com/v3.0.1/reference/introduction)

**Is it free?**  
Tableau Public can be utilized to perform all required tasks in this project for free, although it has limitations compared to the paid versions like Tableau Desktop.

**Python libraries / bindings**

- `requests`: This library will be used to fetch data from the Bitcoin API. Install it via `pip install requests`.  
- `pandas`: Useful for managing data within Python and converting it into CSV format. Install using `pip install pandas`.

This project serves as a practical introduction to using Tableau for data visualization with real-world cryptocurrency data, incorporating essential concepts in data ingestion, cleansing, and time series analysis.

### **TensorFlow**

**Title**: Real-time Bitcoin Price Analysis with TensorFlow

**Difficulty**: 3 (difficult)

**Description**

This project involves leveraging TensorFlow, a powerful open-source machine learning framework, to manage and analyze Bitcoin price data in real-time. TensorFlow's capabilities in handling complex data operations and building machine learning models make it an ideal choice for developing an advanced time series analysis system designed to predict Bitcoin price trends.

**Describe technology**

- **TensorFlow**:  
  - Developed by the Google Brain team, TensorFlow is a comprehensive open-source platform designed for machine learning. It facilitates building and training neural networks to recognize patterns and make predictions, utilizing a flexible architecture that allows deployment across various platforms. TensorFlow supports both CPU and GPU computing and is suitable for research and production environments.  
  - **Core Features**:  
    - **Tensor operations**: TensorFlow efficiently handles multi-dimensional arrays, which are foundational for neural network operations.  
    - **Keras API**: A high-level neural networks API, integrated into TensorFlow, that simplifies building and training machine learning models.  
    - **TensorFlow Serving**: Allows deployment and serving of machine learning models in production environments.  
    - **TF Data Pipelines**: Assists in building scalable data input pipelines to preprocess data efficiently.

**Describe the project**

- **Project Objective**: Develop a sophisticated data pipeline using TensorFlow that ingests real-time Bitcoin price data from public APIs such as CoinGecko, processes it for anomalies, and uses a recurrent neural network (RNN) for time series analysis to predict future prices.  
    
- **Steps**:  
    
  1. **Data Ingestion**:  
     - Use Python's requests library to fetch live Bitcoin price data from an open API at regular intervals.  
  2. **Data Preprocessing**:  
     - Set up TensorFlow Data pipelines to process and clean the incoming data, handle missing values, and normalize prices for modeling.  
  3. **Model Development**:  
     - Design and train an RNN or LSTM (Long Short-Term Memory) model using TensorFlow's Keras API to predict future trends.  
     - Evaluate model performance with test datasets and adjust hyperparameters to enhance prediction accuracy.  
  4. **Real-time Prediction**:  
     - Implement a prediction service using TensorFlow Serving to continuously predict future Bitcoin prices based on incoming data.  
  5. **Visualization and Dashboard**:  
     - Develop a simple dashboard using Python libraries like Matplotlib or Plotly to visualize live price trends and model predictions.

**Useful resources**

- [TensorFlow Official Documentation](https://www.tensorflow.org/guide)  
- [Keras Documentation](https://keras.io/)  
- [CoinGecko API Documentation](https://docs.coingecko.com/v3.0.1/reference/introduction)  
- [Python Requests Library Documentation](https://docs.python-requests.org/en/latest/)

**Is it free?**

Yes, TensorFlow is an open-source framework available for free. You need access to a Python environment and internet connectivity to interact with external APIs.

**Python libraries / bindings**

- **TensorFlow**: For building and training machine learning models.  
- **Keras (included with TensorFlow)**: For simpler neural network model building.  
- **Requests**: To fetch real-time data from APIs (`pip install requests`).  
- **Pandas**: For data manipulation and cleaning (`pip install pandas`).  
- **Matplotlib/Plotly**: For visualizing data and predictions (`pip install matplotlib` or `pip install plotly`).

### **TensorFlow Agents**

**Title:** Reinforcement Learning for Bitcoin Price Prediction Using TensorFlow Agents​

**Difficulty:** 3 (Difficult)​

**Description:** In this project, students will utilize TensorFlow Agents (TF-Agents), a robust library for reinforcement learning (RL) in TensorFlow, to develop an agent capable of predicting Bitcoin price movements. By creating a custom RL environment that reflects the dynamics of Bitcoin trading, students will train an agent to make informed decisions based on historical price data. This project offers hands-on experience in applying RL techniques to financial time series data, encompassing environment design, agent training, and performance evaluation.​

**Describe Technology:**

* **TensorFlow Agents (TF-Agents):**  
  * A comprehensive library for building RL algorithms in TensorFlow.​  
  * Provides modular components such as policies, environments, and networks, facilitating the development and testing of RL models.​  
  * Supports a variety of RL algorithms, including Deep Q-Networks (DQN), Policy Gradient, and Actor-Critic methods.​  
  * Seamlessly integrates with TensorFlow's computational capabilities, enabling efficient model training and deployment.​

**Describe the Project:**

**Objective:** Develop a reinforcement learning agent using TF-Agents to predict and act upon Bitcoin price movements based on historical data.​

**Steps:**

1. **Data Collection:**  
   * Utilize a public API (e.g., CoinGecko) to gather historical Bitcoin price data.​  
   * Process and structure the data to reflect market states, including features such as price changes, trading volume, and time intervals.​  
2. **Environment Creation:**  
   * Design a custom RL environment using TF-Agents that simulates Bitcoin trading scenarios.​  
   * Define the state space (e.g., current price, recent trends), action space (e.g., buy, sell, hold), and reward structure (e.g., profit/loss based on actions).​  
3. **Agent Development:**  
   * Implement a Deep Q-Network (DQN) agent using TF-Agents, tailored to the custom trading environment.​  
   * Configure the neural network architecture to process time-series data effectively.​  
4. **Training and Evaluation:**  
   * Train the DQN agent on historical data, allowing it to learn optimal trading strategies through trial and error.​  
   * Evaluate the agent's performance using separate validation datasets to assess its predictive accuracy and profitability.​  
5. **Visualization and Analysis:**  
   * Visualize the agent's trading decisions and corresponding Bitcoin price movements using libraries like Matplotlib or Plotly.​  
   * Analyze the results to identify patterns, strengths, and areas for improvement in the trading strategy.​

**Useful Resources:**

* [TF-Agents Documentation](https://www.tensorflow.org/agents)  
* [CoinGecko API Documentation](https://docs.coingecko.com/v3.0.1/reference/introduction)  
* [TensorFlow Time Series Tutorial](https://www.tensorflow.org/tutorials/structured_data/time_series)  
* [TF-Agents DQN Tutorial](https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial)

**Is it Free?**

Yes, both TensorFlow and TF-Agents are open-source libraries and free to use. Accessing historical Bitcoin price data from APIs like CoinGecko is also free, though there may be rate limits or usage terms to consider.​

**Python Libraries / Bindings:**

* **TensorFlow:** Core library for machine learning tasks. Install via `pip install tensorflow`.​  
* **TF-Agents:** Library for reinforcement learning in TensorFlow. Install via `pip install tf-agents`.​  
* **Pandas:** For data manipulation and preprocessing. Install via `pip install pandas`.​  
* **NumPy:** For numerical computations. Install via `pip install numpy`.​  
* **Matplotlib / Plotly:** For data visualization. Install via `pip install matplotlib` or `pip install plotly`.​  
* **Requests:** For making HTTP requests to fetch data from APIs. Install via `pip install requests`.​

This project provides a comprehensive introduction to applying reinforcement learning techniques to financial time series data using TensorFlow Agents, offering practical experience in environment design, agent training, and performance evaluation.​

### **TensorFlow Extended (TFX)**

**Title**: Real-Time Bitcoin Data Processing using TensorFlow Extended (TFX)

**Difficulty**: 2 (medium difficulty, should take around 10 days to complete)

**Description**  
TensorFlow Extended (TFX) is an end-to-end platform for deploying production machine learning (ML) pipelines. It provides a cohesive set of tools that enable the integration and management of various ML components, facilitating seamless data ingestion, validation, transformation, and model training/serving. TFX is uniquely tailored to handle large-scale data processing in a robust and efficient manner. This project focuses on using TFX to ingest and process real-time Bitcoin price data for time series analysis, with the aim of creating a streamlined ML pipeline.

**Describe Technology**

- **Data Ingestion**: TFX Pipeline uses components like `ExampleGen` to ingest data; it can handle streaming data and batch inputs.  
- **Data Validation**: The `SchemaGen` and `ExampleValidator` components help in validating the incoming data schema and identifying anomalies.  
- **Data Transformation**: Using `Transform` component, TFX applies feature transformations (e.g., scaling, encoding) necessary for ML model training.  
- **Training and Serving**: `Trainer` and `Pusher` components allow model training using TensorFlow and deployment in a serving environment.  
- **Scalability and Robustness**: Built on TensorFlow, TFX is optimized for high-performance processing and scalability across large datasets.

**Describe the Project**  
The project will involve setting up a TFX pipeline to handle real-time Bitcoin price data to perform time series analysis. Here are the key steps:

- **Set Up Data Ingestion**: Use public APIs (such as CoinGecko or CryptoCompare) to fetch real-time Bitcoin price data and ingest it using TFX's `ExampleGen`.  
- **Data Validation**: Utilize `SchemaGen` to establish a data schema and `ExampleValidator` to ensure data quality by detecting anomalies or inconsistencies.  
- **Data Transformation**: Implement transformations required for time series data, like normalizing prices and creating lag features, using the `Transform` component.  
- **Model Training**: Use a simple TensorFlow LSTM model to forecast Bitcoin price trends and train it using the `Trainer` component.  
- **Model Deployment**: Deploy the trained model using the `Pusher` component for real-time predictions.

Through this project, students will gain practical experience in building and managing ML pipelines using TFX, focusing on the challenges of real-time data processing and analysis.

**Useful Resources**

- TensorFlow TFX Documentation: [TFX Documentation](https://www.tensorflow.org/tfx)  
- Official TensorFlow Guide: [TensorFlow Documentation](https://www.tensorflow.org/guide)  
- Data API References (CoinGecko): [CoinGecko API Documentation](https://docs.coingecko.com/v3.0.1/reference/introduction)

**Is it Free?**  
Yes, TFX is an open-source framework available under the Apache 2.0 license. While Python libraries used in the project are free, note that data APIs may impose rate limits or require subscriptions for extensive usage.

**Python Libraries / Bindings**

- **TensorFlow Extended (TFX)**: The core library for creating and deploying ML pipelines.  
- **TensorFlow**: Required for model training and data processing.  
- **Pandas**: For preliminary data manipulation before ingestion into TFX.  
- **Requests/HTTP Libraries**: To interact with Bitcoin price data APIs.  
- **NumPy**: For handling numerical computations in data transformation.

This project will not only enable students to familiarize themselves with TFX but also provide practical insights into real-time data processing and ML model deployment workflows.

### **TensorFlow Probability**

**Title**: Implementing Time Series Forecasting with TensorFlow Probability for Bitcoin Prices

**Difficulty**: 3 (difficult)

**Description**

TensorFlow Probability (TFP) is a library built on TensorFlow that enables probabilistic reasoning and statistical analysis at scale. It combines tools for deep learning with probabilistic modeling, offering functionalities such as probability distributions, probabilistic layers, and Bayesian methods to create statistical models and perform inference. TensorFlow Probability is ideal for handling tasks where quantifying uncertainty is important, such as financial forecasting and anomaly detection.

In this challenging project, students will utilize TensorFlow Probability to ingest and analyze real-time Bitcoin price data using probabilistic modeling and time series analysis. By the end of the project, students should be able to implement probabilistic models to forecast Bitcoin prices, capture uncertainties, and gain insights into future price trends.

**Describe technology**

- **TensorFlow Probability (TFP)**: A library that provides a suite of tools for probabilistic modeling, allowing for statistical reasoning in AI. TFP integrates with TensorFlow to offer a powerful platform for building complex probabilistic models.  
  - **Probability Distributions**: Built-in support for a wide array of distributions allows for modeling complex, real-world scenarios.  
  - **Probabilistic Layers and Functions**: Framework to build neural networks where components capture uncertainty, essential for predictive modeling.  
  - **Markov Chain Monte Carlo (MCMC)**: Tools to perform sampling-based Bayesian inference.  
  - **Variational Inference**: Methods for approximating probability distributions through optimization, facilitating complex model evaluations.

**Describe the project**

1. **Data Ingestion**:  
     
   - Use Python's `requests` library to fetch real-time Bitcoin price data from a public API such as CoinGecko at regular intervals.  
   - Store the data on a local system or cloud storage for further processing and analysis.

   

2. **Preprocessing**:  
     
   - Clean and prepare the data for time series analysis, involving handling missing data and normalizing the prices.  
   - Segment the data into training and testing datasets.

   

3. **Probabilistic Modeling**:  
     
   - Build a time series model using TensorFlow Probability’s `sts` (Structural Time Series) library.  
   - Utilize probabilistic layers in TensorFlow to design a model that can predict future Bitcoin prices and quantify uncertainties in these predictions.  
   - Apply MCMC or variational inference techniques for accurate model training and inference.

   

4. **Forecasting**:  
     
   - Implement forecasting functionalities to predict future price movements and assess the uncertainty around these predictions.  
   - Visualize the forecasted prices alongside actual historical data to evaluate the model's performance.

   

5. **Evaluation**:  
     
   - Compare the model's predictions to actual price movements using metrics such as Mean Absolute Error (MAE) and evaluate the model's confidence intervals in capturing uncertainties.

**Useful resources**

- [TensorFlow Probability Official Documentation](https://www.tensorflow.org/probability)  
- [Structural Time Series Guide](https://www.tensorflow.org/probability/api_docs/python/tfp/sts) for time series modeling within TensorFlow Probability.  
- [CoinGecko API Documentation](https://docs.coingecko.com/v3.0.1/reference/introduction) for accessing real-time Bitcoin price data.

**Is it free?**

Yes, TensorFlow Probability is open-source software and free to use. Access to Bitcoin price data from public APIs like CoinGecko is also generally free, though usage limits may apply.

**Python libraries / bindings**

- **TensorFlow and TensorFlow Probability**: Core libraries required for building probabilistic models and integrating them with neural network models.  
- **Pandas**: For data manipulation and analysis, especially for handling time series data.  
- **NumPy**: To perform efficient numerical computations, especially useful for data manipulation and processing.  
- **Matplotlib or Seaborn**: Libraries for visualizing the results, particularly the time series plots and model predictions.  
- **Requests**: To fetch real-time Bitcoin data from the API.

These resources combined provide the necessary tools to design, develop, and deploy a sophisticated probabilistic model for real-time Bitcoin price analysis using TensorFlow Probability.

### **Terraform**

**Title:** Local Bitcoin Price Data Processing with Terraform​

**Difficulty:** 1 (Easy)

**Description:** In this project, students will utilize Terraform, an open-source infrastructure as code (IaC) tool by HashiCorp, to automate the setup of a local environment for ingesting and processing Bitcoin price data. By leveraging Terraform's capabilities, students can define and manage infrastructure components locally without the need for a cloud provider. This project offers a hands-on introduction to Terraform's functionalities in a local setting, focusing on data ingestion, processing, and storage using Python.​

**Describe Technology:**

* **Terraform:**  
  * A declarative IaC tool that allows users to define infrastructure components in configuration files using HashiCorp Configuration Language (HCL).​  
  * Supports various providers, including local resources, enabling infrastructure management without external cloud services.​  
  * Features such as execution plans and a dependency graph facilitate safe and efficient infrastructure changes.​

**Describe the Project:**

**Objective:** Use Terraform to provision a local environment for ingesting and processing Bitcoin price data using Python.​

**Steps:**

1. **Set Up Terraform:**  
   * Install Terraform on your local machine by downloading the appropriate binary and adding it to your system's PATH.​  
2. **Provision Local Resources:**  
   * Write Terraform configurations to create local resources, such as directories for data storage and local files for logging.​  
   * Utilize Terraform's `local_file` resource to manage local files.​  
3. **Ingest Bitcoin Data:**  
   * Develop a Python script that fetches real-time Bitcoin price data from a public API (e.g., CoinGecko) at regular intervals.​  
   * Store the retrieved data in the provisioned local directories.​  
4. **Process Data:**  
   * Implement basic time series analysis using Python libraries to preprocess the ingested data, such as calculating moving averages or identifying trends.​  
5. **Automate with Terraform:**  
   * Use Terraform's `null_resource` with `local-exec` provisioner to automate the execution of the Python scripts, ensuring data ingestion and processing occur at defined intervals.

**Useful Resources:**

* [Terraform Official Documentation](https://developer.hashicorp.com/terraform/docs)  
* [CoinGecko API Documentation](https://www.coingecko.com/en/api)  
* [Python `pandas` Library Documentation](https://pandas.pydata.org/docs/)

**Is it Free?**

Yes, Terraform is free and open-source. Accessing real-time data from CoinGecko's API is also free, though there may be rate limits or usage terms to consider.​

**Python Libraries / Bindings:**

* **`requests`:** For making HTTP requests to the Bitcoin pricing API (`pip install requests`).​  
* **`pandas`:** For data manipulation and time series analysis (`pip install pandas`).​  
* **`schedule`:** For scheduling the data fetching at regular intervals (`pip install schedule`).​

This project guides students through the basics of Terraform for local infrastructure management while utilizing Python for data processing, offering a practical approach to Infrastructure as Code in the context of real-time data handling.​

### **TextBlob\#1**

**Title**: Bitcoin News Keyword Trend Analysis Using TextBlob  
**Difficulty**: 3 (Difficult)

**Description**  
**Describe technology**:

- **TextBlob**:  
  - A lightweight Python library for text processing.  
  - Provides tools for **noun phrase extraction**, part-of-speech tagging, and frequency analysis.  
  - Uses NLTK under the hood but simplifies complex NLP tasks.  
- **NewsAPI**:  
  - Aggregates news articles from global sources.  
  - Free tier allows keyword-based filtering (e.g., "Bitcoin," "blockchain").

**Describe the project**:  
**Objective**: Analyze Bitcoin-related news articles to identify trending keywords (e.g., "halving," "regulation," "ETF approval") and correlate their frequency with Bitcoin price movements over time.

**Tasks**:

1. **Ingest News Data**:  
   - Fetch Bitcoin-related articles using NewsAPI.  
   - Store metadata (title, description, publication date) in a DataFrame.  
2. **Keyword Extraction with TextBlob**:  
   - Use TextBlob’s `noun_phrases` method to extract key terms (e.g., "market crash," "institutional adoption").  
   - Create a frequency dictionary of terms per time window (hourly/daily).  
3. **Bitcoin Price Integration**:  
   - Fetch historical price data (e.g., from CoinGecko API).  
   - Align price changes with keyword frequency using timestamps.  
4. **Trend Correlation Analysis**:  
   - Use `pandas` to calculate **term-frequency volatility** (e.g., spikes in "regulation" mentions).  
   - Apply Granger causality tests (via `statsmodels`) to determine if keyword trends *precede* price changes.  
5. **Visualization**:  
   - Plot keyword frequency trends against price charts (e.g., "ETF approval" mentions vs. BTC price spikes).  
   - Use heatmaps to highlight correlations between specific terms and price movements.

**Useful resources**:

- [TextBlob Noun Phrase Extraction Guide](https://textblob.readthedocs.io/en/dev/quickstart.html#noun-phrase-extraction)  
- [Granger Causality in Time Series](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.grangercausalitytests.html)  
- [NewsAPI Python Client](https://newsapi.org/docs/client-libraries/python)

**Is it free?**

- TextBlob: Free and open-source.  
- NewsAPI: Free tier available (500 requests/day).  
- CoinGecko API: Free for non-commercial use.

**Python libraries / bindings**:

- `textblob` (core NLP tasks)  
- `newsapi-python` (news ingestion)  
- `pandas` (time-series alignment)  
- `statsmodels` (Granger causality tests)  
- `matplotlib`/`plotly` (interactive visualizations)

### **TextBlob**

**Title:** Real-Time Bitcoin Sentiment Analysis Using TextBlob

**Difficulty:** 3 (Difficult)​

**Description:** In this project, students will leverage TextBlob, a Python library for processing textual data, to perform real-time sentiment analysis on news articles related to Bitcoin. By integrating NewsAPI, students can access a wide range of news sources to gather relevant articles. The objective is to understand market sentiments and trends associated with Bitcoin prices and explore how this sentiment data can be utilized in time-series analysis for predictive modeling.​

**Describe Technology:**

* **TextBlob:**  
  * Simplifies text processing tasks by providing intuitive functions and methods.​  
  * Utilizes the Natural Language Toolkit (NLTK) and Pattern libraries for comprehensive NLP capabilities.​  
  * Performs sentiment analysis using a pre-trained classifier that returns polarity (ranging from \-1.0 to 1.0) and subjectivity (ranging from 0.0 to 1.0).​  
  * Supports multiple languages for translation and detection, aiding in processing global data trends.​  
* **NewsAPI:**  
  * Provides access to news articles from over 30,000 sources worldwide through a simple HTTP REST API.  
  * Allows filtering of news based on keywords, sources, language, and publication dates.​  
  * Offers a free tier for non-commercial projects with certain usage limitations.​

**Describe the Project:**

**Objective:** Create a pipeline to ingest news articles about Bitcoin using NewsAPI, process the data with TextBlob to perform sentiment analysis, and integrate these sentiment scores into a time-series analysis to predict Bitcoin price movements.​

**Tasks:**

1. **Set Up NewsAPI Client:**  
   * Register for an API key on NewsAPI.org  
   * Use the `newsapi-python` client library to fetch real-time news articles related to Bitcoin.  
2. **Ingest News Data:**  
   * Fetch articles mentioning Bitcoin using the NewsAPI client, focusing on key terms and relevant sources.​  
   * Store the fetched articles, including metadata such as publication date and source, in a Pandas DataFrame for analysis.​  
3. **Perform Sentiment Analysis:**  
   * Utilize TextBlob to analyze the sentiment of each article, calculating polarity and subjectivity scores.​  
   * Aggregate sentiment scores over defined time windows (e.g., daily or hourly) to observe trends.​  
4. **Integrate with Bitcoin Price Data:**  
   * Collect historical and real-time Bitcoin price data from a public API (e.g., CoinGecko).​  
   * Merge sentiment data with corresponding Bitcoin price data based on timestamps.​  
5. **Time-Series Analysis:**  
   * Apply time-series forecasting methods, such as ARIMA or LSTM models, to predict future Bitcoin price trends using sentiment scores as additional features.​  
6. **Visualization:**  
   * Use Matplotlib or Seaborn to visualize the correlation between sentiment trends and Bitcoin price movements.​

**Useful Resources:**

* [TextBlob Documentation](https://textblob.readthedocs.io/en/dev/)  
* [NewsAPI Python Client Library](https://newsapi.org/docs/client-libraries/python)

**Is it Free?**

Yes, TextBlob is an open-source library. NewsAPI offers a free tier for non-commercial projects, which should suffice for educational purposes. However, there may be usage limits, so it's advisable to review their terms of	 service.​

**Python Libraries / Bindings:**

* **TextBlob:** For natural language processing tasks, including sentiment analysis. Install with `pip install textblob`.​  
* **newsapi-python:** For accessing the NewsAPI. Install with `pip install newsapi-python`.   
* **Pandas:** For data manipulation and analysis. Install with `pip install pandas`.​  
* **statsmodels:** For implementing ARIMA models in time-series analysis. Install with `pip install statsmodels`.​  
* **Matplotlib/Seaborn:** For data visualization. Install with `pip install matplotlib seaborn`.​

This project provides a comprehensive approach to understanding the impact of news sentiment on Bitcoin price movements, combining real-time data ingestion, natural language processing, and time-series forecasting.​

### 

### **TinyDB**

**Title**: Real-Time Bitcoin Data Analysis using TinyDB

**Difficulty**: 1 (easy)

**Description**: This project involves using TinyDB, a lightweight and simple JSON-based database for Python, to store and analyze real-time Bitcoin price data. TinyDB is an ideal choice for projects that require minimal setup, allowing students to focus on understanding data ingestion and basic analysis in a time-series context. In this project, students will fetch Bitcoin price data from a public API, store it using TinyDB, and perform basic time-series analysis to understand trends and patterns in price fluctuations.

**Describe Technology**:

- **TinyDB** is a simple, document-oriented database that runs within Python applications. It stores data as JSON documents, making it easy to set up and use for small-scale projects or during initial development stages.  
- Core functionalities include:  
  - Easy installation and setup within any Python environment as it does not require a server.  
  - Supports custom queries using traditional Python expressions.  
  - Provides modules for handling data persistence and caching.

**Describe the Project**:

- **Step 1**: Install TinyDB and required Python libraries. Use `pip install tinydb` to get started.  
- **Step 2**: Use a public API (e.g., CoinGecko) to fetch Bitcoin price data at regular intervals.  
- **Step 3**: Store the ingested data into a TinyDB database. This involves defining a schema for the data (e.g., timestamp, price) and writing functions to add new entries.  
- **Step 4**: Implement basic time-series analysis:  
  - Calculate moving averages to observe price trends over time.  
  - Visualize price changes and trends using simple plots with libraries like Matplotlib.  
- **Step 5**: Document findings and insights from the data analysis, focusing on observed trends and price fluctuations of Bitcoin over the collected dataset.

**Useful Resources**:

- TinyDB Documentation: [TinyDB Docs](https://tinydb.readthedocs.io)  
- CoinGecko API Documentation for Bitcoin data: [CoinGecko API Documentation](https://docs.coingecko.com/v3.0.1/reference/introduction)  
- Python Plotting with Matplotlib: [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)

**Is it free?**: Yes, this project is entirely free to develop and execute. TinyDB is open-source and does not require any subscription or payment. Access to public APIs for Bitcoin data is typically free but may have usage limits.

**Python Libraries / Bindings**:

- **TinyDB**: Lightweight, document-oriented database for storing Bitcoin price data.  
- **Requests**: To make HTTP requests to fetch data from the Bitcoin API. Install it using `pip install requests`.  
- **Matplotlib**: For plotting and visualizing data trends. Can be installed using `pip install matplotlib`.

This project serves as an introductory exercise in handling real-time data, building foundational skills in data ingestion, minor data persistence, and basic time-series analysis using Python.

### **Transformers**

**Title**: Real-Time Bitcoin Analysis Using Transformers

**Difficulty**: 3=difficult

**Description**  
Transformers are a type of deep learning model architecture that has revolutionized natural language processing (NLP) and is increasingly being applied to various domains, including time series forecasting. With their self-attention mechanisms and ability to model long-range dependencies, transformers provide a powerful tool for analyzing sequential data like financial time series. This project involves using transformers to ingest, process, and analyze real-time bitcoin price data to perform predictive analytics. By employing transformer models, students will delve into the complexity of time series analysis and gain experience with cutting-edge deep learning technology applied to financial data.

**Describe technology**

- **Transformers Architecture**: Originally developed for NLP tasks, transformers use self-attention and feed-forward neural networks to capture complex patterns in data. Transformers can handle sequential input efficiently, making them suitable for time series data.  
- **Self-Attention Mechanism**: This allows the model to weigh the significance of different points in the input sequence, which is crucial for capturing temporal dependencies in time series tasks.  
- **Scalability**: Transformers can process sequences in parallel, thus enabling the handling of large datasets and real-time processing.  
- **Model Variants**: Explore various transformer model variants, such as the Transformer Encoder for feature extraction and the Time-series Transformer specifically designed for temporal data.

**Describe the project**

- **Objective**: Implement a real-time bitcoin price analysis system using transformer models to predict future price movements based on live data ingestion.  
- **Data Ingestion**: Use Python to fetch real-time bitcoin prices from an API (e.g., CoinGecko or Binance API). Develop a real-time stream processing pipeline to collect and store incoming data continuously.  
- **Data Preprocessing**: Transform the raw bitcoin price data into a suitable format for input to the transformer model (e.g., normalization, generation of input sequences).  
- **Model Implementation**: Utilize Python libraries (such as TensorFlow or PyTorch) to implement and train a transformer model for time series forecasting of bitcoin prices. Implement strategies to tune and optimize the model for better prediction accuracy.  
- **Performance Evaluation**: Evaluate the model's performance using appropriate metrics (e.g., MAE, RMSE) and validate its predictions through backtesting on historical data. Develop visualizations to display the model's forecast and analyze the outcomes.  
- **Future Enhancements**: Propose potential improvements, such as integrating additional financial indicators or exploring hybrid models for enhanced predictive performance.

**Useful resources**

- PyTorch Documentation: [PyTorch](https://pytorch.org/docs/stable/index.html)  
- TensorFlow Documentation: [TensorFlow](https://www.tensorflow.org/api_docs)  
- [CoinGecko API Documentation](https://docs.coingecko.com/v3.0.1/reference/introduction)

**Is it free?**  
Yes, this project is free to implement as it relies on open-source libraries, and data from public APIs such as CoinGecko can be accessed at no cost. However, ensure you adhere to any API usage terms and limitations.

**Python libraries / bindings**

- **PyTorch or TensorFlow**: For building and training transformer models.  
- **Pandas**: For data manipulation and processing.  
- **NumPy**: For numerical operations and array handling.  
- **Matplotlib or Seaborn**: For visualizing the model predictions and analytics.  
- **Requests**: To fetch data from APIs.  
- **Streamlit or Dash**: For creating an interactive web-based dashboard to display real-time predictions.

### **Tryton**

**Title:** Developing a Simple Library Management Module in Tryton​

**Difficulty:** 1 (Easy)​

**Description:** This project introduces students to the Tryton ERP system by guiding them through the creation of a basic module for managing a library's book inventory. Participants will learn how to set up a Tryton development environment, define data models, and implement functionalities to add, view, and manage book records.​

**Describe technology:** Tryton is an open-source, three-tier, high-level general-purpose application platform. It is written in Python and serves as a robust platform for developing business applications, offering a comprehensive suite of modules for various business processes.​

**Describe the project:**

* **Objective:** To create a Tryton module named `library` that allows users to manage a collection of books, including functionalities to add new books, view existing ones, and categorize them by subject.​  
* **Steps:**  
  1. **Set Up the Development Environment:**  
     * Install Tryton and its dependencies.  
     * Configure a PostgreSQL database for Tryton.  
  2. **Create the Module Structure:**  
     * Initialize a new directory named `library` within the Tryton modules directory.  
     * Create essential files: `tryton.cfg`, `__init__.py`, and `library.py`.  
  3. **Define the Data Model:**  
     * In `library.py`, define a `Book` class with fields such as `title`, `isbn`, `subject`, and `abstract`.  
     * Specify field types and constraints to ensure data integrity.  
  4. **Register the Model:**  
     * In `__init__.py`, register the `Book` model with Tryton's Pool to make it available within the system.  
  5. **Create Views:**  
     * Design XML files to define the form and tree views for the `Book` model, enabling users to interact with the book records through the Tryton client.  
  6. **Implement Menus and Actions:**  
     * Configure menu items and actions to access the `Book` model's views within the Tryton interface.  
  7. **Install and Test the Module:**  
     * Install the `library` module in Tryton.  
     * Test the functionalities by adding, viewing, and managing book records.

**Useful resources:**

* [Tryton Module Tutorial](https://docs.tryton.org/7.0/server/tutorial/module/index.html)​  
* [Writing Your First Tryton Module](https://medium.com/@prkshpp/writing-your-first-tryton-module-992c77d2f021)​  
* [Tryton Discussion: Step-by-Step Tutorial for Beginners](https://discuss.tryton.org/t/step-by-step-tutorial-for-beginners/2217)​

**Is it free?** Yes, Tryton is an open-source platform and free to use.​

**Python libraries / bindings:**

* `trytond`: The core Tryton server package.  
* `tryton`: The Tryton client application.​

This project provides students with foundational experience in developing modules for the Tryton ERP system, encompassing data modeling, view creation, and module integration.​

### **tslearn**

**Title**: Analyze Bitcoin Prices with tslearn

**Difficulty**: 1 (easy)

**Description**  
tslearn is a Python package specifically designed for time series analysis. The library supports an array of time series data processing functionalities, such as machine learning on time series, time series clustering, classification, and regression. Key features of tslearn include tools for normalization, metrics dedicated to time series (such as Dynamic Time Warping), and an interface that synchronizes well with popular libraries such as NumPy and Scikit-learn.

In this project, students will utilize tslearn to analyze real-time Bitcoin price data. They will work on fetching the data from a public API, process it, and perform a basic time series analysis to understand trends and patterns over a specified time frame. The focus will be on demonstrating the core capabilities of tslearn using straightforward time series analysis tasks.

**Describe technology**

- **tslearn**: Provides extensive tools and functions tailored specifically for time series data handling and analysis. It offers:  
  - Various metrics and methods for time series data (e.g., DTW, Soft-DTW).  
  - Tools for time series transformations such as scaling and interpolation.  
  - Compatibility with popular machine learning libraries for enhanced analytic capabilities.  
  - Predefined datasets for quick testing and benchmarking of algorithms.

**Describe the project**

- Fetch real-time Bitcoin price data using a public API such as CoinGecko.  
- Preprocess the incoming data to handle missing values and normalize the data.  
- Apply tslearn’s functionalities to:  
  - Perform clustering based on daily or hourly price patterns, using methods like k-means or hierarchical clustering.  
  - Analyze and visualize time series patterns, identifying significant trends or anomalies.  
- Extend the project by comparing different methods of time series classification offered by tslearn and evaluating their effectiveness on Bitcoin's price volatility.  
- Students will present a simple report with visualization of their findings using libraries like Matplotlib or Seaborn, demonstrating the application of tslearn's tools in analyzing Bitcoin prices effectively.

**Useful resources**

- [tslearn Documentation](https://tslearn.readthedocs.io/en/stable/)  
- [CoinGecko API Documentation](https://docs.coingecko.com/v3.0.1/reference/introduction)

**Is it free?**  
Yes, tslearn is an open-source library and can be freely used for academic and personal projects. Access to Bitcoin price data via public APIs like CoinGecko is typically free, but may have usage limitations or rate restrictions.

**Python libraries / bindings**

- `tslearn`: The primary library for time series analysis. Install it using `pip install tslearn`.  
- `requests`: For making HTTP requests to the Bitcoin price API.  
- `numpy`: For numerical operations and data manipulation.  
- `pandas`: For handling and processing time series data in a structured format.  
- `matplotlib` and `seaborn`: For creating data visualizations to present time series data insights.

### **txtai**

**Title:** Real-Time Bitcoin Sentiment Analysis Using txtai​

**Difficulty:** 2 (Medium)​

**Description:** In this project, students will utilize txtai, an open-source embeddings database, to perform real-time sentiment analysis on news articles related to Bitcoin. By integrating NewsAPI to fetch current news and employing txtai's natural language processing capabilities, the goal is to analyze market sentiment and explore its correlation with Bitcoin price movements. This project offers hands-on experience in semantic search, data ingestion, and time-series analysis within the cryptocurrency domain.​

**Describe Technology:**

* **txtai:**  
  * An all-in-one embeddings database designed for semantic search, language model orchestration, and workflow automation.  
  * Leverages transformer models to create embeddings, enabling efficient and accurate semantic search capabilities.​  
  * Supports building applications that require understanding and processing of natural language queries.​  
* **NewsAPI:**  
  * Provides access to news articles from over 30,000 sources worldwide through a simple HTTP REST API.​  
  * Allows filtering of news based on keywords, sources, language, and publication dates.​  
  * Offers a free tier for non-commercial projects with certain usage limitations.​

**Describe the Project:**

**Objective:** Develop a real-time data ingestion and processing pipeline to analyze Bitcoin-related news articles using txtai, aiming to assess market sentiment and its potential impact on Bitcoin price trends.​

**Tasks:**

1. **Set Up NewsAPI Client:**  
   * Register for an API key on NewsAPI.org.​  
   * Use the `newsapi-python` client library to fetch real-time news articles related to Bitcoin.​  
2. **Ingest News Data:**  
   * Fetch articles mentioning Bitcoin using the NewsAPI client, focusing on key terms and relevant sources.​  
   * Store the fetched articles, including metadata such as publication date and source, in a Pandas DataFrame for analysis.​  
3. **Perform Sentiment Analysis:**  
   * Utilize txtai to analyze the sentiment of each article, calculating sentiment scores.​  
   * Aggregate sentiment scores over defined time windows (e.g., daily or hourly) to observe trends.​  
4. **Integrate with Bitcoin Price Data:**  
   * Collect historical and real-time Bitcoin price data from a public API (e.g., CoinGecko).​  
   * Merge sentiment data with corresponding Bitcoin price data based on timestamps.​  
5. **Time-Series Analysis:**  
   * Apply time-series forecasting methods, such as ARIMA or LSTM models, to predict future Bitcoin price trends using sentiment scores as additional features.​  
6. **Visualization:**  
   * Use Matplotlib or Seaborn to visualize the correlation between sentiment trends and Bitcoin price movements.​

**Useful Resources:**

* [txtai Documentation](https://neuml.github.io/txtai/)  
* [NewsAPI Python Client Library](https://newsapi.org/docs/client-libraries/python)  
* [Pandas Documentation](https://pandas.pydata.org/docs/)

**Is it Free?**

Yes, txtai is an open-source library. NewsAPI offers a free tier for non-commercial projects, which should suffice for educational purposes. However, there may be usage limits, so it's advisable to review their terms of service.​

**Python Libraries / Bindings:**

* **txtai:** For semantic search and natural language processing tasks. Install with `pip install txtai`.​  
* **newsapi-python:** For accessing the NewsAPI. Install with `pip install newsapi-python`.​  
* **Pandas:** For data manipulation and analysis. Install with `pip install pandas`.​  
* **statsmodels:** For implementing ARIMA models in time-series analysis. Install with `pip install statsmodels`.​  
* **Matplotlib/Seaborn:** For data visualization. Install with `pip install matplotlib seaborn`.​

This project provides a comprehensive approach to understanding the impact of news sentiment on Bitcoin price movements, combining real-time data ingestion, natural language processing with txtai, and time-series forecasting.​

### **Vaex**

**Title**: Real-Time Bitcoin Price Analysis using Vaex

**Difficulty**: 2 (Medium)

**Description**:  
Vaex is an open-source Python library optimized for large dataset manipulation and exploration at high speed. It allows for out-of-core operations on large datasets without the need for large amounts of RAM. Vaex enables efficient real-time data analysis by leveraging memory-mapped files and virtual columns. This project will involve using Vaex to ingest, process, and analyze real-time Bitcoin price data. By the end of this project, students will have hands-on experience using Vaex's powerful data manipulation functionalities and have built a simple system for real-time time series analysis of Bitcoin prices.

**Describe technology**:

- **Vaex Core Concepts**:  
    
  - Understand Vaex's approach to data handling through lazy operations, memory mapping, and virtual columns.  
  - Learn about Vaex's functionality for handling large datasets, including out-of-core DataFrames that can handle data that doesn't fit into RAM.


- **Efficient Data Operations**:  
    
  - Explore Vaex's capabilities for filtering, grouping, and aggregating data.  
  - Use Vaex to perform high-performance joint and split operations on large datasets.


- **Visualization and Plotting**:  
    
  - Vaex integrates with various plotting libraries to visualize large datasets efficiently.

**Describe the project**:

- **Objective**: Implement a system to ingest real-time Bitcoin price data from a public API, analyze and visualize time series trends using Vaex.  
- **Steps**:  
  1. **Data Ingestion**:  
     - Use Python to connect to a Bitcoin price data API (such as CoinGecko) to fetch real-time prices at regular intervals.  
  2. **Data Storage**:  
     - Store data in a format compatible with Vaex, such as CSV or HDF5, and load it efficiently for analysis.  
  3. **Data Processing**:  
     - Utilize Vaex to process and filter the data. Examples may include calculating moving averages, detecting price anomalies, or other time series transformations.  
  4. **Time Series Analysis**:  
     - Implement a basic time series analysis, examining trends, volatility, and potentially forecasting future prices based on historical data.  
  5. **Visualization**:  
     - Create visual representations of the analysis using Vaex's integration with other visualization libraries.

**Useful resources**:

- [Vaex Documentation](https://vaex.io/docs/)  
- [Vaex GitHub Repository](https://github.com/vaexio/vaex)  
- [CoinGecko API Documentation](https://docs.coingecko.com/v3.0.1/reference/introduction)

**Is it free?**  
Yes, Vaex is an open-source library and can be used freely. CoinGecko also provides a free tier of their API for accessing Bitcoin and other cryptocurrency data, with certain request limitations.

**Python libraries / bindings**:

- **Vaex**: Primary library for data manipulation and analysis, installable via `pip install vaex`  
- **Requests**: To interact with external APIs, such as fetching live data from CoinGecko; installable via `pip install requests`  
- **Matplotlib/Plotly**: Recommended for visualization alongside Vaex for enhanced plotting capabilities; install using `pip install matplotlib` or `pip install plotly`  
- **Pandas**: For any auxiliary data manipulation not directly supported by Vaex; install via `pip install pandas`

### **Watchtower**

**Title:** Implementing Cloud-Based Logging for a Cryptocurrency Price Tracker Using Watchtower​

**Difficulty:** 1 (Easy)​

**Description:** This project guides students through the development of a cryptocurrency price tracker application with integrated cloud-based logging using the Watchtower library. Participants will build a Python application that fetches real-time cryptocurrency prices and logs relevant information to AWS CloudWatch Logs, facilitating centralized monitoring and analysis.​

**Describe technology:** Watchtower is a lightweight adapter between the Python logging system and AWS CloudWatch Logs. It allows applications to send log messages directly to CloudWatch without the need for additional system-wide log collectors, enabling centralized log management and analysis.​

**Describe the project:**

* **Objective:** To develop a Python application that tracks cryptocurrency prices and utilizes Watchtower to log data to AWS CloudWatch Logs for centralized monitoring.​  
* **Steps:**  
  1. **Set Up AWS Environment:**  
     * Create an AWS account if you don't have one.  
     * Configure AWS credentials using the AWS CLI (`aws configure`).  
  2. **Install Required Libraries:**  
- Install Watchtower and Boto3 using pip:  
  `pip install watchtower boto3`  
  *   
  3. **Develop the Cryptocurrency Price Tracker:**  
     * Use a public API (e.g., CoinGecko) to fetch real-time cryptocurrency prices.  
     * Implement functionality to retrieve and display current prices for selected cryptocurrencies.  
  4. **Integrate Watchtower for Logging:**  
     * Set up Python's logging module to use Watchtower's `CloudWatchLogHandler`.  
     * Configure the logger to send log messages to a specified CloudWatch log group.  
     * Log relevant information, such as fetched prices and any errors encountered during API requests.  
  5. **Run and Monitor the Application:**  
     * Execute the application to ensure it fetches prices and logs data correctly.  
     * Verify that log messages appear in the AWS CloudWatch console under the designated log group.

**Useful resources:**

* [Watchtower Documentation](https://kislyuk.github.io/watchtower/)​  
* [Boto3 Documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)​  
* [CoinGecko API Documentation](https://www.coingecko.com/en/api)​

**Is it free?** Watchtower and Boto3 are open-source and free to use. AWS CloudWatch offers a free tier with limited usage; exceeding the free tier may incur costs.​

**Python libraries / bindings:**

* `watchtower`: For sending log messages to AWS CloudWatch Logs.​  
* `boto3`: AWS SDK for Python, used by Watchtower for AWS interactions.​  
* `requests`: For making HTTP requests to the cryptocurrency API.​

This project provides students with practical experience in integrating cloud-based logging into a Python application, enhancing skills in application monitoring and cloud services.​

### **XGBoost**

**Title**: Predict Bitcoin Prices Using XGBoost

**Difficulty**: 3 (difficult)

**Description**

This project will guide you through the advanced implementation of an XGBoost model to predict Bitcoin prices based on real-time data. XGBoost (Extreme Gradient Boosting) is a scalable and distributed gradient-boosted decision tree (GBDT) machine learning library that is renowned for its performance and efficiency in predictive modeling. In this challenging project, students will work on ingesting real-time Bitcoin price data, transforming it to the desired format, and then applying XGBoost to perform time series analysis and make future price predictions.

**Describe technology**

- **XGBoost**:  
  - XGBoost is an open-source software library providing a gradient boosting framework.  
  - It is designed to be highly efficient, flexible, and portable, making it suitable for rapid deployment in machine learning projects.  
  - The core functionalities include handling missing data gracefully, regularization to prevent overfitting, and parallelized implementations to improve performance.  
  - XGBoost supports various interfaces, but in this project, we'll focus on its Python API.

**Describe the project**

- **Objective**: Use XGBoost to perform real-time predictive modeling of Bitcoin prices.  
- **Steps**:  
  1. **Data Ingestion**: Begin by ingesting real-time Bitcoin price data from a public API such as CoinGecko. Use basic Python libraries like `requests` to pull the data at regular intervals.  
  2. **Data Processing**: Transform the raw JSON data into structured data frames using `pandas`. Handle missing values and perform feature engineering to create relevant predictors for your model.  
  3. **Model Implementation**:  
     - Set up the XGBoost model using the 'xgboost' Python library.  
     - Define the hyperparameters for the XGBoost algorithm, considering time series data nature.  
     - Train the model on historical data adjusted for patterns, trends, and seasonality.  
  4. **Prediction and Evaluation**:  
     - Deploy the model for real-time prediction of future Bitcoin prices.  
     - Evaluate the model's performance using metrics like MAE (Mean Absolute Error) and RMSE (Root Mean Square Error).  
  5. **Result Visualization**:  
     - Visualize the actual vs. predicted prices using `matplotlib` to identify how well the model forecasts.

**Useful resources**

- [XGBoost Documentation](https://xgboost.readthedocs.io/)  
- [CoinGecko API Documentation](https://docs.coingecko.com/v3.0.1/reference/introduction)  
- [Python Requests Documentation](https://docs.python-requests.org/en/master/)


**Is it free?**

Yes, XGBoost is open-source and free to use. However, accessing a continuous stream of Bitcoin price data might incur a fee depending on the data provider.

**Python libraries / bindings**

- `xgboost`: For model implementation and training.  
- `pandas`: For data manipulation and transformation.  
- `requests`: For API data ingestion.  
- `matplotlib`: For visualization of results.  
- `numpy`: For numerical computations supporting the data processing tasks.

### **Zapier Central**

**Title:** Automated Bitcoin Price Alerts Using Zapier​

**Difficulty:** 1 (Easy)​

**Description:** In this project, students will utilize Zapier, a no-code automation platform, to set up automated alerts based on real-time Bitcoin price fluctuations. By integrating various web applications without writing code, students will learn to create workflows that monitor Bitcoin prices and send notifications when certain thresholds are met. This project offers hands-on experience in workflow automation and real-time data monitoring.​

**Describe Technology:**

* **Zapier:**  
  * A cloud-based platform that allows users to integrate and automate tasks between different web applications.​  
  * Enables the creation of "Zaps," which are automated workflows triggered by specific events in connected apps.  
  * Supports integration with over 7,000 applications, including email services, messaging platforms, and data storage solutions.​

**Describe the Project:**

**Objective:** Set up an automated system that monitors Bitcoin prices and sends notifications when predefined price thresholds are crossed.​

**Steps:**

1. **Data Source Integration:**  
   * Utilize a cryptocurrency data provider (e.g., CoinGecko) that offers real-time Bitcoin price data through APIs.​  
2. **Zapier Account Setup:**  
   * Create a Zapier account and familiarize yourself with the dashboard and basic functionalities.​  
3. **Create a Zap:**  
   * **Trigger:** Set up a schedule trigger in Zapier to fetch Bitcoin price data at regular intervals (e.g., every 15 minutes).​  
   * **Action:** Use Zapier's Webhooks feature to make an HTTP request to the cryptocurrency data provider's API to retrieve the current Bitcoin price.​  
4. **Set Up Conditional Logic:**  
   * Implement a filter in Zapier to check if the retrieved Bitcoin price meets certain conditions (e.g., exceeds or falls below specified thresholds).​  
5. **Notification Actions:**  
   * Configure actions to send notifications through preferred channels:​  
     * **Email:** Use Zapier's Email by Zapier app to send an email alert.  
     * **SMS:** Integrate with an SMS service (e.g., Twilio) to send text message alerts.  
     * **Messaging Apps:** Connect to platforms like Slack or Microsoft Teams to post alert messages in designated channels.  
6. **Testing and Activation:**  
   * Test the Zap to ensure it functions as intended and activate it to start monitoring.​

**Useful Resources:**

* [Zapier Guides](https://zapier.com/resources/guides)  
* [CoinGecko API Documentation](https://www.coingecko.com/en/api)  
* [Zapier Webhooks Guide](https://help.zapier.com/hc/en-us/articles/8496083355661-How-to-get-started-with-Webhooks-by-Zapier)

**Is it Free?**

Zapier offers a free plan that includes:​

* Up to 100 tasks per month.​  
* 5 Zaps (automated workflows).  
* 15-minute update time.

For more advanced features or higher task allowances, paid plans are available starting at $19.99 per month.

By completing this project, students will gain practical experience in using Zapier to automate tasks and monitor real-time data, enhancing their understanding of workflow automation without the need for extensive programming knowledge.​

### **Zarr**

**Title**: Time Series Analysis of Bitcoin Prices using Zarr

**Difficulty**: 1 (easy)

**Description**:  
This project will introduce you to Zarr, a Python library designed to handle large, high-dimensional datasets efficiently. You'll learn how to use Zarr to store and manage Bitcoin price data for real-time time series analysis. Starting with real-time data ingestion from a public Bitcoin API, you'll explore how Zarr can be utilized to organize and handle large datasets without consuming too much memory, making it ideal for scenarios where large datasets are involved.

**Describe technology**:

- Zarr is a Python library specifically designed for the storage of large arrays.  
- It supports chunked storage, which allows for efficient handling of data that doesn't fit into memory, enabling operations on subsets of data.  
- Allows for the storage of multidimensional arrays using the NumPy API.  
- Offers compatibility with different storage backends such as local disk, cloud storage, or custom backends.  
- Provides support for seamless parallel computing and multiprocessing tasks.  
- Offers flexibility in storage by enabling various compression and encoding options.

**Describe the project**:

- **Objective**: Implement a simple system to store Bitcoin price data using Zarr for time series analysis.  
- **Step 1**: Set up data ingestion from a Bitcoin price API, like CoinGecko, to obtain real-time price updates.  
- **Step 2**: Use Zarr to store the ingested Bitcoin data. Demonstrate how to chunk the data to enable efficient access and manipulation.  
- **Step 3**: Perform a basic time series analysis on the stored data. This can involve computing moving averages, identifying trends, or visualizing the price changes over time.  
- **Step 4**: Demonstrate how to manage and access the stored data using Zarr's array indexing capabilities for efficient analysis without loading the entire dataset into memory.

**Useful resources**:

- [Zarr Documentation](https://zarr.readthedocs.io/en/stable/)  
- [Numpy User Guide](https://numpy.org/doc/stable/user/)  
- [CoinGecko API Documentation](https://docs.coingecko.com/v3.0.1/reference/introduction)  
- [Real-Time Data Visualization in Python](https://matplotlib.org/)

**Is it free?**  
Yes, Zarr is open-source and free to use. You can freely install and use it in your Python projects.

**Python libraries / bindings**:

- **Zarr**: Provides storage and manipulation of large arrays. Install with `pip install zarr`.  
- **NumPy**: Used for handling and processing numerical data. Install with `pip install numpy`.  
- **Requests**: For making HTTP requests to fetch data from the Bitcoin API. Install with `pip install requests`.  
- **Matplotlib**: For plotting and visualizing data. Install with `pip install matplotlib`.

