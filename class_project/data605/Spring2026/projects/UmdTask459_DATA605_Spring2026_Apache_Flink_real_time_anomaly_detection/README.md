# Apache Flink Real-Time Weather Anomaly Detection

**Course:** DATA605 Big Data Systems, Spring 2026  
**Author:** Rajesh Easwaramoorthy  
**UMD ID:** 122242479  
**GitHub Issue:** [#459](https://github.com/gpsaggese/gpsaggese.github.io/issues/459)

---

## Overview

This project builds a real-time streaming analytics application using **Apache Flink** (via PyFlink) that ingests live weather data from the OpenWeatherMap API and detects anomalies in temperature, humidity, and pressure readings.

Anomaly detection uses two methods together:

1. **City-specific rolling baseline** - flags sudden deviations relative to each city's recent history.
2. **Isolation Forest** (scikit-learn) - a global unsupervised model that identifies statistically rare readings across all cities.

When an anomaly is detected the system logs it, writes a time-series point to InfluxDB, and sends an email alert.

---

## Technology Stack

| Component | Role |
|---|---|
| **Apache Flink / PyFlink** | Stream processing engine, DataStream pipeline |
| **OpenWeatherMap API** | Live weather data source (4 cities) |
| **Isolation Forest** | Unsupervised anomaly detection model |
| **InfluxDB** | Time-series data storage |
| **Grafana** | Real-time visualisation dashboard |
| **Docker / Docker Compose** | Containerised deployment |

---

## Project Structure

```
UmdTask459_DATA605_Spring2026_Apache_Flink_real_time_anomaly_detection/
--- Dockerfile                          # Python 3.11 + Java JDK + PyFlink
--- docker_build.sh                     # Build Docker image
--- docker_jupyter.sh                   # Launch Jupyter Lab
--- docker_name.sh                      # Docker image naming variables
--- docker_bash.sh                      # Interactive shell in container
--- docker_clean.sh                     # Remove Docker images
--- requirements.txt                    # Python dependencies
--- flink_weather_anomaly_utils.py      # Shared utility functions
--- flink_weather_anomaly.API.ipynb     # PyFlink API tutorial notebook
--- flink_weather_anomaly.example.ipynb # End-to-end example notebook
--- data/
  --- weather_data.csv                # Training dataset (1200 rows)
  --- weather_logs_sample.csv         # Sample collected event log
--- models/
  --- isolation_forest.pkl            # Pre-trained model
--- pipeline/                           # Live streaming pipeline
    --- openweather_stream_job.py        # PyFlink job: API polling, detection, InfluxDB, email
    --- train_model.py                   # Model training script
    --- test_model.py                    # Model evaluation script
    --- docker-compose.yml               # Starts app + InfluxDB + Grafana
    --- .env.example                     # Template for required credentials
```

---

## Notebooks

### `flink_weather_anomaly.API.ipynb`
This notebook covers the PyFlink DataStream API. It walks through creating a `StreamExecutionEnvironment`, building streams with `from_collection()`, applying `map`, `filter`, and custom `MapFunction` classes, understanding the PyFlink type system (`Types.STRING`, `Types.INT`, etc.), and executing Flink jobs.

### `flink_weather_anomaly.example.ipynb`
This notebook runs through the full weather anomaly detection system, from loading and exploring the training data to training the Isolation Forest model, running records through a PyFlink DataStream pipeline, evaluating detection performance (precision, recall, F1, accuracy), and visualising anomaly events, deviation histograms, and the confusion matrix.

---

## Quick Start

### Prerequisites
- Docker installed and running
- Git repository cloned (`gpsaggese/umd_classes`)

### 1. Build the Docker image

```bash
cd class_project/DATA605/Spring2026/projects/UmdTask459_DATA605_Spring2026_Apache_Flink_real_time_anomaly_detection
bash docker_build.sh
```

### 2. Launch Jupyter Lab

```bash
bash docker_jupyter.sh -p 8888
```

Open your browser at **http://localhost:8888** and run the notebooks in order:
1. `flink_weather_anomaly.API.ipynb`
2. `flink_weather_anomaly.example.ipynb`

---

## Live Pipeline

The `pipeline/` folder contains the full production streaming system. It runs
continuously, polls OpenWeatherMap every 10 minutes, and feeds each reading
through the PyFlink anomaly detection pipeline.

### Services started by Docker Compose

| Service | Port | Role |
|---|---|---|
| `weather-anomaly-app` | - | PyFlink job: fetches API data, detects anomalies, writes to InfluxDB, sends email alerts |
| `influxdb` | 8086 | Time-series storage for all weather readings |
| `grafana` | 3000 | Real-time dashboard showing temperature, alerts, and deviations per city |

### Credentials (.env file)

The live pipeline requires a `.env` file in the `pipeline/` directory. Copy
`.env.example` and fill in your values:

```
OPENWEATHER_API_KEY=your_openweathermap_api_key
GMAIL_USER=your_gmail_address@gmail.com
GMAIL_APP_PASSWORD=your_16_character_app_password
ALERT_RECIPIENT=recipient_email@gmail.com
```

**OPENWEATHER_API_KEY** - free API key from https://openweathermap.org/api (sign up, free tier is enough).

**GMAIL_APP_PASSWORD** - not your regular Gmail password. Go to Google Account, Security, 2-Step Verification, then App Passwords and generate a 16-character password for this app. This is required because Gmail blocks direct password login for third-party apps.

**ALERT_RECIPIENT** - the email address that receives anomaly alerts. Can be the same as GMAIL_USER.

The `.env` file is never committed to the repository because it contains secrets.

### Running the live pipeline

```bash
cd pipeline
cp .env.example .env
# fill in your credentials in .env
docker compose up --build
```

Once running, open **http://localhost:3000** in your browser to view the Grafana dashboard.

---

## Key Design Decisions

**Why PyFlink instead of raw Python?**  
PyFlink provides the DataStream abstraction that separates *what to compute* from *how to execute it*. The same `map()` function can run locally (parallelism=1) for development or on a distributed Flink cluster for production without code changes.

**Why Isolation Forest?**  
Weather anomaly detection is an unsupervised problem, since there are no labelled anomaly examples at runtime. Isolation Forest efficiently identifies outliers by randomly partitioning the feature space; anomalies are isolated in fewer splits than normal points.

**Why dual detection?**  
The Isolation Forest is trained on a global dataset and may miss city-specific anomalies (e.g., an unusual cold snap in a normally warm city). The rolling city baseline catches localised deviations that the global model overlooks.

---

## References

- [Apache Flink Documentation](https://flink.apache.org/docs/stable/)
- [PyFlink DataStream API Guide](https://nightlies.apache.org/flink/flink-docs-stable/docs/dev/python/datastream_tutorial/)
- [OpenWeatherMap API](https://openweathermap.org/api)
- [Isolation Forest, Liu et al. (2008)](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf)
- [scikit-learn IsolationForest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
