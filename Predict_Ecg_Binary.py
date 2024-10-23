# Corrected predict.py - Load Model and Make Predictions
import joblib
import numpy as np
from utility import preprocess
from config import get_config

# Load the pre-trained SVM model
config = get_config()
model = joblib.load(config.model_path)

def make_prediction(input_data, config):
    try:
        # Preprocess the input data
        processed_data, _ = preprocess(input_data, config)

        # Ensure the data is in the format the SVM expects
        processed_data = processed_data.reshape(1, -1)

        # Make the prediction
        prediction = model.predict(processed_data)

        # Mapping the prediction to either "Normal" or "Anomaly"
        label_map = {
            0: 'Normal',    # Class 0 represents normal data
            1: 'Anomaly'    # Class 1 represents anomaly
        }

        # Get the diagnosis based on the model's prediction
        diagnosis = label_map.get(int(prediction[0]), "Unknown")

        return {"prediction": int(prediction[0]), "diagnosis": diagnosis}

    except Exception as e:
        return {"error": str(e)}
