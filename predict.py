import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the pre-trained SVM model
MODEL_PATH = r"C:\stasi\SoftUni_Machine_learning\optimized_svm_model.pkl"  # Update this with the correct path
model = joblib.load(MODEL_PATH)

# Function to preprocess the input data (you may need to customize this)
def preprocess_data(input_data):
    scaler = StandardScaler()
    processed_data = scaler.fit_transform(np.array(input_data).reshape(1, -1))
    return processed_data

# Function to make predictions using the model
def make_prediction(input_data):
    try:
        # Preprocess the data
        processed_data = preprocess_data(input_data)
        
        # Make a prediction
        prediction = model.predict(processed_data)
        
        return {"prediction": int(prediction[0])}
    
    except Exception as e:
        return {"error": str(e)}
