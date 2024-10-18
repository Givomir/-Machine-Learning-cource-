from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from predict import make_prediction  # Import the prediction function

app = FastAPI()

# Define the schema for the incoming prediction requests
class PredictionInput(BaseModel):
    features: list  # A list of numerical features

# Define the prediction endpoint
@app.post("/predict")
async def predict(input_data: PredictionInput):
    try:
        # Call the prediction function from predict.py
        result = make_prediction(input_data.features)
        
        # Check if an error occurred during prediction
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# If needed, you can add additional routes here (e.g., for model retraining, data visualization, etc.)
