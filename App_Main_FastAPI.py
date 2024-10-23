# Corrected app.py - FastAPI Application for ECG Predictions
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from predict import make_prediction
from config import get_config

# Initialize FastAPI app
app = FastAPI()

# Define the schema for the incoming prediction requests
class PredictionInput(BaseModel):
    features: list

@app.post("/predict")
async def predict(input_data: PredictionInput):
    # Load the configuration
    config = get_config()

    # Call the prediction function
    result = make_prediction(input_data.features, config)

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return result

# Run using uvicorn:
# uvicorn app:app --reload
