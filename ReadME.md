# ECG Predictor API with PySpark and FastAPI

A FastAPI-based ECG prediction API using PySpark and a pre-trained Support Vector Machine (SVM) model. This API takes ECG features as input and predicts if the signal is normal or anomalous.

## Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Features
- **ECG Anomaly Detection**: Classifies ECG data as either "Normal" or "Anomaly."
- **FastAPI Endpoints**: Provides an HTTP API endpoint for easy integration.
- **PySpark for Data Processing**: Leverages PySpark for large-scale data handling.

---

## Requirements
- **Python** 3.6+
- **PySpark** 3.5.1
- **FastAPI** and **Uvicorn** for API management
- **Joblib** for loading the SVM model
- **Numpy** and **Pandas** for data handling

Refer to `requirements.txt` for specific library versions.

---

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/ecg-predictor-api.git
   cd ecg-predictor-api

## Usage

1. uvicorn app:app --reload
