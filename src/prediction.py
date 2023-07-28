import json
import pickle
import pandas as pd

# new_data 
def predict_price(df):
    with open("./models/df_cleaned.pkl", 'rb') as f:
        pickle_model = pickle.load(f)
    predictions = pickle_model.predict(df)
    
    return predictions