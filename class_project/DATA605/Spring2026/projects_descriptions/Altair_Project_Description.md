# Altair

## Description
- Altair is a declarative statistical visualization library for Python, designed
  to create interactive and informative visualizations with minimal code.
- It leverages the power of Vega and Vega-Lite, allowing users to generate a
  wide range of visual representations of data using a concise syntax.
- The library is built on a grammar of graphics, which means users can easily
  combine visual elements and encode data attributes to create complex
  visualizations.
- Altair supports a variety of data types and formats, including pandas
  DataFrames, which makes it easy to integrate into existing data science
  workflows.
- It offers capabilities for interactive visualizations, enabling users to
  create charts that respond to user inputs, such as selections and filters.
- The library is particularly suited for exploratory data analysis (EDA) and
  communicating data insights effectively through visual storytelling.

## Project Objective
The goal of this project is to analyze and visualize trends in global
temperature changes over time, optimizing for insights into climate patterns and
anomalies. Students will employ Altair to create interactive visualizations that
help communicate their findings effectively.

## Dataset Suggestions
1. **Global Historical Climatology Network (GHCN)**
   - **Source Name**: NOAA
   - **URL**: [NOAA GHCN](https://www.ncdc.noaa.gov/ghcn-daily-description)
   - **Data Contains**: Daily temperature records from weather stations
     worldwide.
   - **Access Requirements**: No authentication required; data is publicly
     available.

2. **Kaggle Global Temperature Data**
   - **Source Name**: Kaggle
   - **URL**:
     [Kaggle Global Temperature](https://www.kaggle.com/datasets/berkeleyearth/climate-change-earth-surface-temperature-data)
   - **Data Contains**: Monthly average temperatures from 1750 to present,
     including anomalies.
   - **Access Requirements**: Free to use after signing up for a Kaggle account.

3. **NASA GISS Surface Temperature Analysis (GISTEMP)**
   - **Source Name**: NASA
   - **URL**: [NASA GISTEMP](https://datahub.io/core/global-temp)
   - **Data Contains**: Global surface temperature data from 1880 to present,
     including anomalies and trends.
   - **Access Requirements**: Publicly available, no authentication needed.

4. **OpenWeatherMap Historical Weather Data**
   - **Source Name**: OpenWeatherMap
   - **URL**: [OpenWeatherMap API](https://openweathermap.org/history)
   - **Data Contains**: Historical weather data including temperature, humidity,
     and precipitation.
   - **Access Requirements**: Free tier available; requires API key (easy
     signup).

## Tasks
- **Data Acquisition**: Load the selected dataset(s) into a pandas DataFrame for
  analysis, ensuring proper formatting and cleaning.
- **Exploratory Data Analysis**: Use Altair to create initial visualizations
  (e.g., line charts, histograms) to explore the dataset and identify trends or
  anomalies.
- **Interactive Visualization**: Develop interactive visualizations using Altair
  features, such as tooltips and selection filters, to allow users to explore
  data insights dynamically.
- **Insights and Reporting**: Summarize findings from the visualizations,
  discussing key trends in global temperatures and potential implications for
  climate change.
- **Presentation**: Prepare a presentation that showcases the visualizations and
  insights, emphasizing the storytelling aspect of the data analysis.

## Bonus Ideas
- Implement a comparison between different regions or time periods in the
  temperature data to identify localized trends.
- Explore additional variables such as CO2 levels or extreme weather events and
  visualize their relationships with temperature changes.
- Create a dashboard using Altair and Streamlit to allow users to interact with
  multiple visualizations simultaneously.

## Useful Resources
- [Altair Official Documentation](https://altair-viz.github.io/)
- [Vega-Lite Documentation](https://vega.github.io/vega-lite/)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Streamlit Documentation](https://docs.streamlit.io/)
