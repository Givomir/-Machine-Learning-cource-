import joblib
import numpy as np
from config import get_config

# Load the pre-trained SVM model pipeline
config = get_config()
model_pipeline = joblib.load(config.model_path)

def make_prediction(input_data, config):
    try:
        # Convert input_data to the appropriate format
        input_data = np.array(input_data).reshape(1, -1)

        # Make the prediction using the entire pipeline
        prediction = model_pipeline.predict(input_data)

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