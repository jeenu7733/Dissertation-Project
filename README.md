# Dissertation-Project

# Project Title: Forecasting Electricity Consumption using Machine Learning and Statistical Techniques

## Project Overview
This repository contains the dissertation project focused on forecasting electricity consumption using a combination of statistical, machine learning, and deep learning techniques. The project aims to develop accurate and robust models for predicting electricity demand, contributing to more efficient energy management and planning strategies.

## Objectives
- Analyze historical electricity consumption patterns
- Implement and compare four distinct forecasting models
- Evaluate the performance of different techniques
- Provide insights for better energy demand management

## Contents
- `Forecasting Electricity Consumption Using Machine Learning and Statistical Techniques.ipynb`: Main Jupyter notebook containing the entire project workflow
- `README.md`: This file, providing an overview of the project
- `Requirements.txt`: List of Python libraries required for this project
- `Total Household Consumption`: Captures the total electricity consumption of households with sample intervals ranging from one to fifteen minutes.
- `Electric Water Heater Consumption`: Focuses on the electricity consumption of electric water heaters within households.
- `Disaggregated Energy Consumption by Appliance`: Records the electricity consumption of individual appliances in households.


## Methodology
The project employs four main techniques:
1. ARIMA (AutoRegressive Integrated Moving Average): A statistical method for time series forecasting
2. K-Means Clustering: Used for pattern recognition in electricity consumption data
3. Random Forest: An ensemble machine learning method for regression tasks
4. GRU (Gated Recurrent Unit): A type of recurrent neural network for sequence prediction

## Key Features
- Comprehensive data preprocessing pipeline
- Feature engineering tailored for electricity consumption data
- Implementation of statistical, machine learning, and deep learning models
- Comparative analysis of model performance
- Visualization of results and forecasts

## Technologies Used
- Python 3.8+
- Pandas, NumPy for data manipulation
- Scikit-learn for K-Means and Random Forest
- Statsmodels for ARIMA implementation
- TensorFlow and Keras for GRU model
- Matplotlib and Seaborn for data visualization

## Setup and Installation
1. Clone the repository:
2. Install required packages:
      pip install -r requirements.txt

## Usage
1. Navigate to the `notebooks/` directory
2. Open and run the Jupyter notebooks in sequence for a step-by-step walkthrough of the project

## Results
The project demonstrates the effectiveness of various forecasting techniques, comparing the performance of ARIMA, K-Means clustering, Random Forest, and GRU models. Detailed findings are available in the `docs/research_paper.pdf`.

## Future Work
- Integration of external factors (weather, economic indicators) into the models
- Exploration of ensemble methods combining the strengths of multiple models
- Development of a user-friendly interface for model deployment
