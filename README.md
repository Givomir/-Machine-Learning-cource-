# ECG Predictor API with PySpark and FastAPI

## Description
This project is an API for detecting anomalies in ECG signals, built using FastAPI and PySpark. It leverages a pre-trained Support Vector Machine (SVM) model to classify ECG data as "Normal" or "Anomaly."

---

## Features
- **ECG Anomaly Detection:** Takes a list of features as input and returns a diagnosis (Normal/Anomaly).
- **FastAPI Integration:** Provides a user-friendly HTTP interface for requests and responses.
- **Large-Scale Data Processing with PySpark:** Handles scalable datasets efficiently.

---

## Technologies Used
- **FastAPI:** Provides a RESTful API interface.
- **PySpark:** Processes large-scale data.
- **Joblib:** Loads pre-trained models.
- **Scikit-learn:** Used for model training (SVM).
- **Pandas/Numpy:** Handles data manipulation.
- **Scipy:** Processes ECG signals.

---

## Project Structure
- **`app.py`:** Main file for the FastAPI application.
- **`predict.py`:** Core logic for model loading and predictions.
- **`config.py`:** Configuration file for application parameters.
- **`utility.py`:** Helper functions for data preprocessing.
- **`Train_Ecg_Classifier_V2.py`:** Code for training a Random Forest model as an alternative to SVM.
- **`CreateCSV.py`:** Generates sample CSV data for testing.

---

## Installation and Usage

### Installation
1. **Clone the repository:**  
   ```bash
   git clone https://github.com/your-username/ecg-predictor-api.git
   cd ecg-predictor-api


uvicorn app:app --reload
