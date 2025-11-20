# ONNX Time Series Forecasting of Stock Prices - TODO List

## Project Overview
Build a time series forecasting model for stock prices using historical financial data, and use ONNX for cross-framework deployment.

---

## Phase 1: Environment Setup and Configuration

### 1.1 Docker Environment
- [ ] Verify Docker setup is complete (simple workflow)
- [ ] Test `docker_build.sh` script execution
- [ ] Test `docker_bash.sh` script execution
- [ ] Test `docker_jupyter.sh` script execution
- [ ] Verify Jupyter Lab is accessible

### 1.2 Project Dependencies
- [ ] Install TensorFlow/Keras in Docker container
- [ ] Install ONNX and ONNX Runtime
- [ ] Install tf2onnx converter
- [ ] Install pandas, numpy, scikit-learn
- [ ] Install matplotlib, seaborn for visualization
- [ ] Install Streamlit for dashboard
- [ ] Install technical analysis libraries (ta-lib or pandas-ta)
- [ ] Install transformers library (for Transformer model comparison)
- [ ] Update requirements.txt or Dockerfile with all dependencies

### 1.3 Project Structure
- [ ] Create folder structure following template guidelines
- [ ] Create placeholder files: ONNX.API.md, ONNX.API.ipynb
- [ ] Create placeholder files: ONNX.example.md, ONNX.example.ipynb
- [ ] Create utils modules: utils_data_io.py, utils_preprocessing.py, utils_model.py, utils_evaluation.py
- [ ] Create README.md with project overview

---

## Phase 2: Data Acquisition and Preprocessing

### 2.1 Data Collection
Already done, see /data

### 2.3 Data Preprocessing (utils_preprocessing.py)
- [ ] Implement function to load stock data from CSV
- [ ] Implement date parsing and sorting
- [ ] Implement missing value handling (forward fill, interpolation)
- [ ] Implement outlier detection and handling
- [ ] Implement train/validation/test split (chronological)
- [ ] Implement data normalization/scaling (MinMaxScaler)
- [ ] Document preprocessing pipeline in docstrings

### 2.4 Feature Engineering (utils_preprocessing.py)
- [ ] Implement moving averages (SMA, EMA) - 5, 10, 20, 50 days
- [ ] Implement volatility indicators (Bollinger Bands, ATR)
- [ ] Implement momentum indicators (RSI, MACD)
- [ ] Implement volume-based indicators
- [ ] Implement lagged features (previous N days)
- [ ] Implement rolling window creation for time series sequences
- [ ] Test feature engineering functions with sample data
- [ ] Document feature engineering approach in markdown

---

## Phase 3: Model Development

### 3.1 LSTM Model Architecture (utils_model.py)
- [ ] Design LSTM architecture (input shape, hidden layers, output)
- [ ] Implement LSTM model builder function
- [ ] Define hyperparameters (sequence length, hidden units, dropout)
- [ ] Implement model compilation (loss function, optimizer, metrics)
- [ ] Document model architecture with diagrams (use mermaid)

### 3.2 LSTM Model Training (utils_model.py)
- [ ] Implement training function with callbacks (EarlyStopping, ModelCheckpoint)
- [ ] Implement validation during training
- [ ] Train LSTM model on preprocessed data
- [ ] Save trained model in TensorFlow format
- [ ] Log training history (loss, metrics over epochs)
- [ ] Visualize training/validation curves
- [ ] Document training process and hyperparameters

### 3.3 Transformer Model (Optional Comparison)
- [ ] Research Transformer architecture for time series
- [ ] Implement Transformer model builder function (or use existing library)
- [ ] Train Transformer model with same data
- [ ] Save trained Transformer model
- [ ] Document Transformer architecture and training

---

## Phase 4: ONNX Conversion and Deployment

### 4.1 Model to ONNX Conversion (utils_model.py)
- [ ] Implement function to convert TensorFlow LSTM model to ONNX
- [ ] Convert trained LSTM model to ONNX format (.onnx file)
- [ ] Verify ONNX model structure using onnx.checker
- [ ] Implement function to convert Transformer model to ONNX (if applicable)
- [ ] Save ONNX models in appropriate directory (e.g., `models/onnx/`)
- [ ] Document conversion process and any issues encountered

### 4.2 ONNX Runtime Inference (utils_model.py)
- [ ] Implement ONNX Runtime session creation function
- [ ] Implement inference function using ONNX Runtime
- [ ] Test ONNX Runtime inference with sample input
- [ ] Verify output shape and data types match expectations
- [ ] Document ONNX Runtime setup and usage

### 4.3 Cross-Framework Comparison (utils_evaluation.py)
- [ ] Implement function to run inference with native TensorFlow
- [ ] Implement function to run inference with ONNX Runtime
- [ ] Compare predictions between frameworks (numerical accuracy)
- [ ] Measure inference time for both approaches
- [ ] Measure memory usage for both approaches
- [ ] Document comparison methodology and results

---

## Phase 5: Model Evaluation

### 5.1 Forecasting Metrics (utils_evaluation.py)
- [ ] Implement Mean Absolute Error (MAE) calculation
- [ ] Implement Root Mean Squared Error (RMSE) calculation
- [ ] Implement Mean Absolute Percentage Error (MAPE) calculation
- [ ] Implement additional metrics (R², directional accuracy)
- [ ] Create evaluation function that computes all metrics

### 5.2 Model Performance Evaluation
- [ ] Evaluate LSTM model on test set using ONNX Runtime
- [ ] Evaluate Transformer model on test set using ONNX Runtime
- [ ] Compare LSTM vs. Transformer performance
- [ ] Create performance comparison table
- [ ] Visualize predictions vs. actual values
- [ ] Document evaluation results in markdown

### 5.3 Error Analysis
- [ ] Analyze prediction errors over time
- [ ] Identify patterns in forecasting errors
- [ ] Visualize residuals
- [ ] Document findings and potential improvements

---

## Phase 6: Streamlit Dashboard

### 6.1 Dashboard Design
- [ ] Design dashboard layout (sidebar for inputs, main area for visualizations)
- [ ] Plan interactive components (stock selector, date range, model selector)
- [ ] Sketch dashboard wireframe

### 6.2 Dashboard Implementation (streamlit_app.py)
- [ ] Create Streamlit app file
- [ ] Implement stock selection dropdown
- [ ] Implement date range selector
- [ ] Implement model selector (LSTM vs. Transformer)
- [ ] Load ONNX models in Streamlit app
- [ ] Implement real-time forecast generation using ONNX Runtime
- [ ] Create time series visualization (actual vs. predicted)
- [ ] Display evaluation metrics on dashboard
- [ ] Add feature importance or technical indicators visualization
- [ ] Implement caching for performance optimization
- [ ] Test dashboard locally

### 6.3 Dashboard Documentation
- [ ] Document how to run Streamlit app
- [ ] Add screenshots of dashboard to markdown files
- [ ] Document interactive features and usage

---

## Phase 7: API Documentation (ONNX.API.*)

### 7.1 ONNX Native API Documentation (ONNX.API.md)
- [ ] Research and document ONNX library overview
- [ ] Explain what ONNX is and what problem it solves
- [ ] Document ONNX alternatives (TensorFlow Lite, TorchScript, etc.)
- [ ] Describe ONNX model format and structure
- [ ] Document key ONNX classes and functions (ModelProto, save, load, checker)
- [ ] Document ONNX Runtime API (InferenceSession, run)
- [ ] Create mermaid diagrams for ONNX architecture
- [ ] Add references to official ONNX documentation and tutorials

### 7.2 Custom Wrapper API Documentation (ONNX.API.md)
- [ ] Document custom wrapper functions for model conversion
- [ ] Document custom wrapper functions for ONNX Runtime inference
- [ ] Provide API usage examples with code snippets
- [ ] Document design decisions for wrapper layer
- [ ] Create class diagrams or interface definitions (dataclasses, Protocol)

### 7.3 ONNX API Notebook (ONNX.API.ipynb)
- [ ] Create notebook demonstrating ONNX model creation from scratch
- [ ] Demonstrate ONNX model loading and inspection
- [ ] Demonstrate model conversion (TensorFlow to ONNX)
- [ ] Demonstrate ONNX Runtime inference with simple examples
- [ ] Demonstrate custom wrapper API usage
- [ ] Add clear explanations in markdown cells
- [ ] Ensure notebook runs end-to-end (Restart & Run All)
- [ ] Keep execution time under 5 minutes

---

## Phase 8: Example Documentation (ONNX.example.*)

### 8.1 Stock Forecasting Example Documentation (ONNX.example.md)
- [ ] Write introduction to stock price forecasting problem
- [ ] Explain choice of LSTM and Transformer models
- [ ] Document complete data preprocessing pipeline
- [ ] Document feature engineering approach
- [ ] Document model training process
- [ ] Document ONNX conversion process
- [ ] Document evaluation methodology
- [ ] Create data flow diagrams using mermaid
- [ ] Create model architecture diagrams using mermaid
- [ ] Include performance comparison tables
- [ ] Add references to research papers and resources

### 8.2 Stock Forecasting Example Notebook (ONNX.example.ipynb)
- [ ] Create end-to-end notebook for stock forecasting
- [ ] Section 1: Import libraries and load utilities
- [ ] Section 2: Load and explore stock data
- [ ] Section 3: Preprocess data and engineer features
- [ ] Section 4: Build and train LSTM model
- [ ] Section 5: Convert LSTM model to ONNX
- [ ] Section 6: Perform inference with ONNX Runtime
- [ ] Section 7: Evaluate forecasting performance
- [ ] Section 8: Compare LSTM vs. Transformer (optional)
- [ ] Section 9: Visualize results and forecasts
- [ ] Add clear markdown explanations between code cells
- [ ] Ensure notebook runs end-to-end (Restart & Run All)
- [ ] Keep execution time reasonable (< 10 minutes)

### 8.3 Visual Documentation
- [ ] Create data preprocessing flowchart
- [ ] Create model training pipeline diagram
- [ ] Create ONNX conversion workflow diagram
- [ ] Create deployment architecture diagram
- [ ] Include performance comparison charts
- [ ] Add example forecast visualizations

---

## Phase 9: Utilities Module (utils files)

### 9.1 utils_data_io.py
- [ ] Implement functions to load stock CSV files
- [ ] Implement functions to save/load preprocessed data
- [ ] Implement functions to save/load models
- [ ] Add comprehensive docstrings
- [ ] Add type hints
- [ ] Write unit tests

### 9.2 utils_preprocessing.py
- [ ] Consolidate all preprocessing functions
- [ ] Ensure functions are modular and reusable
- [ ] Add comprehensive docstrings
- [ ] Add type hints
- [ ] Write unit tests

### 9.3 utils_model.py
- [ ] Consolidate model building functions
- [ ] Consolidate training functions
- [ ] Consolidate ONNX conversion functions
- [ ] Consolidate inference functions
- [ ] Add comprehensive docstrings
- [ ] Add type hints
- [ ] Write unit tests

### 9.4 utils_evaluation.py
- [ ] Consolidate evaluation metric functions
- [ ] Consolidate visualization functions
- [ ] Add comprehensive docstrings
- [ ] Add type hints
- [ ] Write unit tests

---

## Phase 10: Testing and Validation

### 10.1 Notebook Testing
- [ ] Test ONNX.API.ipynb - Restart & Run All
- [ ] Test ONNX.example.ipynb - Restart & Run All
- [ ] Verify all outputs are present and correct
- [ ] Verify execution time is reasonable
- [ ] Fix any errors or issues

### 10.2 Utility Function Testing
- [ ] Write unit tests for utils_data_io.py
- [ ] Write unit tests for utils_preprocessing.py
- [ ] Write unit tests for utils_model.py
- [ ] Write unit tests for utils_evaluation.py
- [ ] Run all unit tests with pytest
- [ ] Achieve reasonable test coverage

### 10.3 Integration Testing
- [ ] Test complete pipeline: data loading → preprocessing → training → conversion → inference → evaluation
- [ ] Test with multiple stocks
- [ ] Test error handling (missing data, invalid inputs)
- [ ] Document test results

### 10.4 Streamlit Testing
- [ ] Test Streamlit app with different stocks
- [ ] Test all interactive components
- [ ] Test on different browsers (if applicable)
- [ ] Verify performance and loading times

---

## Phase 11: Documentation Finalization

### 11.1 README.md
- [ ] Write project overview and objectives
- [ ] Document dataset and data source
- [ ] Document project structure (folder tree)
- [ ] Document Docker setup instructions
- [ ] Document how to run notebooks
- [ ] Document how to run Streamlit dashboard
- [ ] Add links to API and example documentation
- [ ] Add project status and completion checklist
- [ ] Include example outputs and visualizations

### 11.2 Markdown Files Review
- [ ] Review ONNX.API.md for completeness and clarity
- [ ] Review ONNX.example.md for completeness and clarity
- [ ] Ensure all diagrams render correctly
- [ ] Ensure all code snippets are properly formatted
- [ ] Check for typos and grammatical errors
- [ ] Verify all references and citations

### 11.3 Code Quality
- [ ] Review all code for readability and style (PEP 8)
- [ ] Add missing comments and docstrings
- [ ] Remove debug code and unused imports
- [ ] Ensure consistent naming conventions
- [ ] Run code linter (pylint, flake8)

---

## Phase 12: Final Review and Submission

### 12.1 Deliverables Checklist
- [ ] ONNX.API.md - Complete and reviewed
- [ ] ONNX.API.ipynb - Runs end-to-end, documented
- [ ] ONNX.example.md - Complete and reviewed
- [ ] ONNX.example.ipynb - Runs end-to-end, documented
- [ ] utils_data_io.py - Complete with docstrings
- [ ] utils_preprocessing.py - Complete with docstrings
- [ ] utils_model.py - Complete with docstrings
- [ ] utils_evaluation.py - Complete with docstrings
- [ ] Dockerfile - Tested and working
- [ ] docker_build.sh, docker_bash.sh, docker_jupyter.sh - Tested
- [ ] README.md - Complete with all instructions
- [ ] Streamlit app - Functional and documented

### 12.2 Final Testing
- [ ] Clean Docker environment and rebuild from scratch
- [ ] Test complete workflow in fresh Docker container
- [ ] Run all notebooks in sequence
- [ ] Verify all outputs are reproducible
- [ ] Test Streamlit app in Docker container

### 12.3 Submission Preparation
- [ ] Create final Git commit with all changes
- [ ] Tag release version (e.g., v1.0)
- [ ] Verify all files are included in repository
- [ ] Check that .gitignore excludes unnecessary files
- [ ] Create final project archive if required

### 12.4 Presentation Preparation (if required)
- [ ] Prepare slides summarizing the project
- [ ] Create demo script for live demonstration
- [ ] Prepare answers to potential questions
- [ ] Practice presentation timing (60 minutes)

---

## Additional Notes

### Key Success Criteria
- All notebooks must run end-to-end without errors
- Docker setup must work out of the box
- Documentation must be clear and beginner-friendly
- Code must be well-organized and reusable
- The tutorial should enable someone to learn ONNX basics in 60 minutes

### Common Pitfalls to Avoid
- Don't copy-paste code without understanding it
- Don't skip documentation for "obvious" steps
- Don't hardcode paths or parameters
- Don't ignore error handling
- Don't leave TODO comments in final submission
- Don't forget to test notebooks end-to-end before submission

### Resources
- ONNX Official Documentation: https://onnx.ai/
- ONNX Runtime Documentation: https://onnxruntime.ai/
- TensorFlow to ONNX: https://github.com/onnx/tensorflow-onnx
- Stock Price Forecasting Papers: (add relevant papers)
- Tutorial Template: instructions/tutorial_template/tutorial_github_data605_style/

### Time Estimates
- Phase 1-2 (Setup & Data): 1-2 days
- Phase 3 (Model Development): 2-3 days
- Phase 4-5 (ONNX & Evaluation): 2-3 days
- Phase 6 (Dashboard): 1-2 days
- Phase 7-8 (Documentation): 2-3 days
- Phase 9-10 (Testing): 1-2 days
- Phase 11-12 (Finalization): 1-2 days
- **Total Estimated Time: 10-18 days**

---

## Progress Tracking

**Project Start Date:** [To be filled]
**Expected Completion Date:** [To be filled]
**Actual Completion Date:** [To be filled]

**Current Phase:** Phase 1 - Environment Setup and Configuration
**Completion Status:** 0% (0/XXX tasks completed)

---

**Last Updated:** 2025-11-06
