import pandas as pd
import joblib

# get new data (in json format) and preprocess data - return Dataframe
def preprocess_new_data(df):
   df.drop_duplicates()
   df = df.dropna()
   # Convert the Postalcode column to string
   df["Postalcode"] = df["Postalcode"].astype(str)
   # Define the categorical columns + numerical columns
   cat_cols= ['Type', 'Postalcode', 'Region', 'Province']
   num_cols= [
      'Construction_year', 'Total_surface', 'Habitable_surface',
      'Bedroom_count', 'Terrace', 'Garden_surface', 'Facades',
      'Kitchen_equipped', 'Condition_encoded'
      ]
    

    # load encoder and scaler from original training
   encoder = joblib.load('./models/encoder.joblib')
   scaler = joblib.load('./models/scaler.joblib')

   encoded_columns = encoder.get_feature_names_out(input_features=cat_cols)
    
   X_test_enc = encoder.transform(df[cat_cols])
   X_test_enc_df = pd.DataFrame(X_test_enc.toarray(), columns=encoded_columns)

   # scale them
   X_test_scale = scaler.transform(df[num_cols])
    
   X_test = pd.concat([pd.DataFrame(X_test_scale, columns=num_cols), X_test_enc_df], axis=1)

   return X_test

