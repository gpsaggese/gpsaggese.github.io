# Apache Flink

## Description
- Apache Flink is an open-source stream processing framework for real-time data
  processing and analytics.
- It provides high-throughput, low-latency processing capabilities for both
  batch and streaming data.
- Flink supports complex event processing (CEP), allowing users to detect
  patterns and anomalies in data streams.
- The framework is designed for scalability, enabling it to handle large volumes
  of data across distributed systems seamlessly.
- Flink integrates well with various data sources and sinks, including Hadoop,
  Apache Kafka, and various databases, making it versatile for different data
  engineering tasks.

## Project Objective
The goal of the project is to build a real-time analytics application that
processes streaming data from a public API to detect anomalies in temperature
readings. Students will optimize the anomaly detection model to minimize false
positives while ensuring timely alerts.

## Dataset Suggestions
1. **OpenWeatherMap API**
   - **URL**: [OpenWeatherMap API](https://openweathermap.org/api)
   - **Data Contains**: Current weather data, including temperature, humidity,
     and atmospheric pressure.
   - **Access Requirements**: Free tier available, requires sign-up for an API
     key.

2. **NOAA Climate Data Online**
   - **URL**: [NOAA API](https://www.ncdc.noaa.gov/cdo-web/webservices/v2)
   - **Data Contains**: Historical climate data including temperature,
     precipitation, and wind speed.
   - **Access Requirements**: Free access, requires registration for a token.

3. **USGS Earthquake Hazards Program**
   - **URL**: [USGS Earthquake API](https://earthquake.usgs.gov/fdsnws/event/1/)
   - **Data Contains**: Real-time earthquake data, including location and
     magnitude.
   - **Access Requirements**: Public API, no authentication required.

4. **Kaggle Weather Data**
   - **URL**:
     [Kaggle Weather Dataset](https://www.kaggle.com/datasets/uciml/air-quality)
   - **Data Contains**: Air quality data with temperature readings from various
     locations.
   - **Access Requirements**: Free to use after creating a Kaggle account.

## Tasks
- **Set Up Apache Flink Environment**: Install and configure Apache Flink on
  your local machine or use a cloud service like Google Cloud.
- **Data Ingestion**: Implement Flink jobs to ingest streaming data from the
  selected API(s) using connectors.
- **Data Processing**: Develop Flink transformations to clean and preprocess the
  incoming data streams, ensuring the data is ready for analysis.
- **Anomaly Detection Model**: Use a pre-trained machine learning model (e.g.,
  Isolation Forest) to detect anomalies in the temperature readings.
- **Real-time Alerting**: Create a mechanism to send alerts (e.g., email or
  logging) when anomalies are detected in real-time.
- **Performance Evaluation**: Analyze the performance of the anomaly detection
  system, measuring metrics such as precision, recall, and response time.

## Bonus Ideas
- Implement a visualization dashboard using a tool like Grafana to display
  real-time analytics and alerts.
- Compare the performance of different anomaly detection algorithms (e.g., LOF
  vs. Isolation Forest) on the same dataset.
- Explore the integration of additional data sources (e.g., humidity or
  pressure) to enhance the anomaly detection model.
- Create a batch processing job to analyze historical data and compare results
  with real-time analysis.

## Useful Resources
- [Apache Flink Official Documentation](https://flink.apache.org/documentation.html)
- [OpenWeatherMap API Documentation](https://openweathermap.org/api)
- [NOAA Climate Data Online API Documentation](https://www.ncdc.noaa.gov/cdo-web/webservices/v2)
- [USGS Earthquake API Documentation](https://earthquake.usgs.gov/fdsnws/event/1/)
- [Kaggle Weather Dataset](https://www.kaggle.com/datasets/uciml/air-quality)
