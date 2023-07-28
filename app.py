import pickle
import pandas as pd
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, constr, conint, confloat, validator
from src.preprocessing import preprocess_new_data
from src.prediction import predict_price

# Load the XGBoost model from the .pkl file
with open("./models/df_cleaned.pkl", "rb") as model_file:
    xgb_model = pickle.load(model_file)
with open("./models/scaler.joblib", "rb") as model_file:
    scaler = pickle.load(model_file)
with open("./models/encoder.joblib", "rb") as model_file:
    encoder = pickle.load(model_file)
# Initialize the FastAPI app
app = FastAPI()

# Define the input and output data models
class InputDataModel(BaseModel):
    Postalcode: conint(ge=1000, le=9992)  # Integer with max length of 4 (between 1000 and 9992)
    Type: constr(max_length=10)  # Only allows "HOUSE" or "APPARTMENT"
    Region: constr(max_length=10)  # Only allows specific values
    Province: constr(max_length=20)  # Only allows specific values
    Construction_year: confloat(ge=0.0)  # Float, greater than or equal to 0
    Total_surface: confloat(ge=0.0)  # Float, greater than or equal to 0
    Habitable_surface: confloat(ge=0.0)  # Float, greater than or equal to 0
    Bedroom_count: confloat(ge=0.0)  # Float, greater than or equal to 0
    Terrace: bool
    Garden_surface: confloat(ge=0.0)  # Float, greater than or equal to 0
    Facades: confloat(ge=0.0)  # Float, greater than or equal to 0
    Kitchen_equipped: conint(ge=0, le=1)  # Integer, either 0 or 1
    Condition_encoded: conint(ge=1, le=6)  # Integer, between 1 and 6

    @validator('Type')
    def type_must_be_valid(cls, value):
        if value not in ('HOUSE', 'APPARTMENT'):
            raise ValueError("Type must be 'HOUSE' or 'APPARTMENT'")
        return value

    @validator('Region')
    def region_must_be_valid(cls, value):
        if value not in ('FLANDERS', 'BRUSSELS', 'WALLONIE'):
            raise ValueError("Region must be 'FLANDERS', 'BRUSSELS', or 'WALLONIE'")
        return value

    @validator('Province')
    def province_must_be_valid(cls, value):
        allowed_provinces = [
            'East Flanders', 'Brussels', 'Liège', 'Flemish Brabant', 'Antwerp',
            'West Flanders', 'Hainaut', 'Limburg', 'Luxembourg', 'Namur', 'Walloon Brabant'
        ]
        if value not in allowed_provinces:
            raise ValueError("Invalid Province. Allowed values are: " + ", ".join(allowed_provinces))
        return value

class OutputDataModel(BaseModel):
    Prediction: float

#http://127.0.0.1:8000
# handling get request
@app.get('/')
async def main():
    return {"message": "Welcome to the Real Estate Price Prediction API!",
        "instructions": "To predict the property price, make a POST request to /predict/ with the following input data format:",
        "input_format": {
            "Postalcode": "str(!) with max length of 4 (between 1000 and 9993)",
            "Type": "str, Only allows: HOUSE|APPARTMENT",
            "Region": "str, Only allows specific values: FLANDERS|BRUSSELS|WALLONIE",
            "Province": "str, Only allows specific values: East Flanders|Brussels|Liège|Flemish Brabant|Antwerp|West Flanders|Hainaut|Limburg|Luxembourg|Namur|Walloon Brabant",
            "Construction_year": "Float, greater than or equal to 0",
            "Total_surface": "Float, greater than or equal to 0",
            "Habitable_surface": "Float, greater than or equal to 0",
            "Bedroom_count": "Float, greater than or equal to 0",
            "Terrace": "bool",
            "Garden_surface": "Float, greater than or equal to 0",
            "Facades": "Float, greater than or equal to 0",
            "Kitchen_equipped": "Integer 1 (equiped) or 0 (no kitchen)",
            "Condition_encoded": "Integer, between 1 and 6 where TO_RESTORE': 1,'TO_RENOVATE': 2,'TO_BE_DONE_UP': 3,'JUST_RENOVATED': 4,'GOOD': 5, 'AS_NEW': 6"
        }
            }

# Define the prediction endpoint
@app.post("/predict/")
async def predict(data: InputDataModel):
    try:
        input_data = json.loads(data.model_dump_json())
        df = pd.DataFrame(input_data, index=[0])
        df = preprocess_new_data(df)
        prediction = predict_price(df)

        # Return the prediction as JSON
        return {"Prediction": float(prediction[0])}

    except Exception as e:
        # Handle any exceptions and return an appropriate HTTP error response
        raise HTTPException(status_code=400, detail=str(e))