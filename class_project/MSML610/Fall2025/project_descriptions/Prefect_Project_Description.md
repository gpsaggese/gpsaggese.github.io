**Description**

In this project, students will utilize Prefect, a workflow orchestration tool, to automate and manage data pipelines for various data science tasks. Prefect helps streamline data workflows, allowing for efficient scheduling, monitoring, and error handling. Its features include:

- **Dynamic Workflows**: Build workflows that can adapt based on real-time data.
- **Task Scheduling**: Schedule tasks to run at specific intervals or based on triggers.
- **Error Handling**: Implement retry logic and notifications for task failures.
- **Visualization**: Monitor workflow execution and performance through an intuitive UI.

---

### Project 1: Predicting House Prices (Difficulty: 1)

**Project Objective**: Build a data pipeline that ingests housing data, preprocesses it, and trains a machine learning model to predict house prices based on various features.

**Dataset Suggestions**:
- **Dataset**: Ames Housing Dataset (available on Kaggle)
- **Link**: [Ames Housing Dataset](https://www.kaggle.com/datasets/prestonvong/AmesHousing)

**Tasks**:
- **Set Up Prefect Environment**: Create a Prefect project and configure the necessary infrastructure.
- **Data Ingestion**: Use Prefect to automate the download of the Ames Housing dataset from Kaggle.
- **Data Preprocessing**: Implement data cleaning and preprocessing tasks (handling missing values, encoding categorical features).
- **Model Training**: Train a regression model (e.g., Linear Regression) to predict house prices.
- **Evaluation**: Evaluate model performance using metrics like RMSE and visualize results.

---

**Project 2: Twitter Sentiment Analysis (Batch Mode)**

**Difficulty: 2 (Medium)**

**Project Objective**
Analyze public sentiment on a specific topic by collecting historical tweets and applying sentiment analysis, optimizing for accuracy and interpretability.

**Dataset Suggestions**
Kaggle Dataset: Twitter US Airline Sentiment

**Tasks**

- **Data Ingestion**: Load the historical tweets dataset into your workflow.

- **Text Preprocessing**: Clean tweets (remove hashtags, mentions, links, emojis).

- **Sentiment Analysis**: Use a pre-trained model (e.g., VADER, Hugging Face DistilBERT) to classify sentiment as positive/negative/neutral.

- **Data Storage**: Save results into CSV or a SQLite database for further exploration.

- **Visualization**: Plot sentiment distributions, word clouds, or topic-specific sentiment trends.

**Bonus Ideas (Optional)**

- Compare traditional lexicon-based methods (VADER/TextBlob) with transformer-based sentiment classifiers.
- Perform topic modeling (e.g., LDA or BERTopic) to see which themes drive positive vs negative sentiment.
- Extend analysis to multi-class classification (e.g., joy, anger, sadness) using emotion-labeled datasets.

---

### Project 3: Anomaly Detection in Network Traffic (Difficulty: 3)

**Project Objective**: Implement a complex data pipeline to analyze network traffic data and detect anomalies that could indicate security threats.

**Dataset Suggestions**:
- **Dataset**: UNSW-NB15 dataset (available on Kaggle)
- **Link**: [UNSW-NB15 Dataset](https://www.kaggle.com/datasets/mohammadami/unsw-nb15-dataset)

**Tasks**:
- **Prefect Workflow Design**: Design a Prefect workflow that orchestrates the entire data processing and analysis pipeline.
- **Data Ingestion**: Automate the loading of the UNSW-NB15 dataset from Kaggle.
- **Feature Engineering**: Extract and engineer relevant features from the raw network traffic data for better anomaly detection performance.
- **Anomaly Detection Model**: Implement a machine learning model (e.g., Isolation Forest or Autoencoder) to identify anomalous patterns in the data.
- **Monitoring and Alerts**: Set up monitoring for the workflow and create alerts for detected anomalies.

**Bonus Ideas**:
- For Project 1: Compare different regression models and analyze performance differences.
- For Project 2: Extend the pipeline to include geolocation data and analyze sentiment by region.
- For Project 3: Implement a dashboard using tools like Dash or Streamlit to visualize detected anomalies in real time.

