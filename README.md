# Pediatric Appendicitis Diagnosis

A machine learning project for **pediatric appendicitis diagnosis** using clinical data and **explainable AI**.

## Overview
This project aims to support medical decision-making by predicting appendicitis risk in children from patient data.  
It combines **data preprocessing**, **model training**, **web-based prediction**, and **SHAP explainability** to make the results more understandable.

## Features
- Data cleaning and preprocessing
- Machine learning model training and evaluation
- Best model selection for prediction
- Simple application for patient input and diagnosis support
- SHAP visualization for model interpretability

## Tech Stack
- Python
- Jupyter Notebook
- Machine Learning
- LightGBM / Scikit-learn
- SHAP
- Flask / Streamlit

## Project Structure
- `clean-data/` – data cleaning notebook
- `Machine_learning/` – training scripts and saved model
- `application/` – diagnosis app and SHAP interpretation
- `data/` – datasets and model files

## Results
The selected model achieved strong performance for appendicitis prediction, and SHAP helped highlight the most important clinical features influencing the model’s decisions.

## Future Improvements
- Improve the app interface
- Add deployment for online use
- Expand the dataset
- Enhance model explainability
