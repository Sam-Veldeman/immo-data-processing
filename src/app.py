import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Load the XGBoost model from the .pkl file
with open("../models/df_house.pkl", "rb") as model_file:
    xgb_model = pickle.load(model_file)
# Check if the model is loaded successfully
print("XGBoost Model Loaded Successfully!")
# Initialize the FastAPI app
app = FastAPI()

# Define the input and output data models
class InputDataModel(BaseModel):
    Construction_year: float
    Total_surface: float
    Habitable_surface: float
    Bedroom_count: float
    Furnished: bool
    Fireplace: bool
    Terrace: bool
    Terrace_surface: float
    Garden_surface: float
    Facades: float
    SwimmingPool: bool
    Kitchen_equipped: int
    Condition_encoded: int

class OutputDataModel(BaseModel):
    Prediction: float

# Define the prediction endpoint
@app.post("/predict/", response_model=OutputDataModel)
async def predict(data: InputDataModel):
    try:
        # Convert the input data to a numpy array for prediction
        input_features = np.array([
            data.Construction_year, data.Total_surface, data.Habitable_surface,
            data.Bedroom_count, data.Furnished, data.Fireplace, data.Terrace,
            data.Terrace_surface, data.Garden_surface, data.Facades,
            data.SwimmingPool, data.Kitchen_equipped, data.Condition_encoded
        ]).reshape(1, -1)

        # Make the prediction using the XGBoost model
        prediction = xgb_model.predict(input_features)[0]

        # Return the prediction as JSON
        return {"Prediction": float(prediction)}

    except Exception as e:
        raise HTTPException(status_code=500, detail="Prediction Error")
