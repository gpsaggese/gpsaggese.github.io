# AutoKeras

## Description
- AutoKeras is an open-source AutoML library for deep learning that simplifies
  the process of building machine learning models.
- It automates the model selection and hyperparameter tuning process, allowing
  users to focus on data rather than the complexities of model architecture.
- The tool supports various types of tasks, including classification,
  regression, and image classification, making it versatile for different
  projects.
- AutoKeras provides a user-friendly API that integrates seamlessly with
  TensorFlow and Keras, making it accessible for both beginners and advanced
  users.
- The library includes functionalities for data preprocessing, model evaluation,
  and visualization of results, enhancing the overall workflow.

## Project Objective
The goal of this project is to develop a machine learning model that predicts
housing prices based on various features such as location, size, and amenities.
The project will focus on optimizing the model to achieve the highest accuracy
in price prediction.

## Dataset Suggestions
1. **Kaggle Housing Prices Dataset**
   - **Source**: Kaggle
   - **URL**:
     [Housing Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
   - **Data Contains**: Features of houses including area, number of rooms,
     location, and sale prices.
   - **Access Requirements**: A free Kaggle account for downloading the dataset.

2. **Zillow Home Value Index (ZHVI)**
   - **Source**: Zillow
   - **URL**:
     [Zillow Data](https://www.zillow.com/howto/api/Zillow-Data-API.htm)
   - **Data Contains**: Monthly median home values for various regions in the
     U.S.
   - **Access Requirements**: API key required for access, but free for
     non-commercial use.

3. **OpenStreetMap (OSM) Housing Data**
   - **Source**: OpenStreetMap
   - **URL**: [OSM Data](https://download.geofabrik.de/)
   - **Data Contains**: Geographic data including housing attributes (e.g.,
     number of floors, building type).
   - **Access Requirements**: No authentication needed, but requires data
     parsing.

4. **UCI Machine Learning Repository - Boston Housing Dataset**
   - **Source**: UCI Machine Learning Repository
   - **URL**: [Boston Housing](https://archive.ics.uci.edu/ml/datasets/Housing)
   - **Data Contains**: Various features of housing in Boston, including crime
     rate, number of rooms, and property tax rates.
   - **Access Requirements**: No restrictions for downloading.

## Tasks
- **Data Collection**: Gather data from the selected dataset(s) and prepare it
  for analysis.
- **Data Preprocessing**: Clean the data by handling missing values, encoding
  categorical variables, and normalizing numerical features.
- **Model Training**: Use AutoKeras to automatically search for the best model
  architecture and hyperparameters for predicting housing prices.
- **Model Evaluation**: Assess the model's performance using metrics such as
  Mean Absolute Error (MAE) and R-squared values.
- **Results Visualization**: Create visualizations to present the model's
  predictions against actual prices and highlight important features.

## Bonus Ideas
- Extend the project by incorporating additional features such as local
  amenities, school ratings, or economic indicators to improve prediction
  accuracy.
- Compare the performance of AutoKeras with traditional machine learning
  algorithms (like linear regression or decision trees) to highlight the
  advantages of using AutoML.
- Implement a web app using Flask or Streamlit to allow users to input features
  and get real-time price predictions.

## Useful Resources
- [AutoKeras Documentation](https://autokeras.com/)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [Zillow API Documentation](https://www.zillow.com/howto/api/APIOverview.htm)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)
- [OpenStreetMap API](https://wiki.openstreetmap.org/wiki/API)
