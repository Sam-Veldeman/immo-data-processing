from pathlib import Path
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import joblib
from datetime import datetime
import os

def get_latest_cleaned_data_csv(data_dir):
    data_dir = Path(data_dir)
    csv_files = list(data_dir.glob('cleaned_data_*.csv'))

    if csv_files:
        latest_csv = max(csv_files, key=os.path.getctime)
        return latest_csv
    else:
        return None

def train_model():
    # Define the directory containing cleaned data CSV files
    data_directory = Path('/opt/airflow/dags/Data')

    # Get the path to the latest cleaned_data CSV file
    latest_cleaned_data_csv = get_latest_cleaned_data_csv(data_directory)

    if latest_cleaned_data_csv:
        # Load the latest cleaned data CSV
        df = pd.read_csv(latest_cleaned_data_csv, index_col='id', skip_blank_lines=True)
    else:
        print("No cleaned_data CSV file found. Please run the cleaning script first.")
        return

    # Data preprocessing
    cat_cols = ['type', 'postalcode', 'region', 'province']
    num_cols = ['construction_year', 'total_surface', 'habitable_surface', 'bedroom_count', 'terrace', 'garden_surface', 'facades', 'kitchen_equipped', 'condition_encoded']
    X = df.drop(columns=['price'], axis=1)
    y = df[['price']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=22)

    encoder = OneHotEncoder(handle_unknown='ignore')
    X_train_enc = encoder.fit_transform(X_train[cat_cols])
    X_test_enc = encoder.transform(X_test[cat_cols])

    scaler = MinMaxScaler()
    X_train_scale = scaler.fit_transform(X_train[num_cols])
    X_test_scale = scaler.transform(X_test[num_cols])

    encoded_columns = encoder.get_feature_names_out(input_features=cat_cols)
    X_train_enc_df = pd.DataFrame(X_train_enc.toarray(), columns=encoded_columns)
    X_test_enc_df = pd.DataFrame(X_test_enc.toarray(), columns=encoded_columns)

    X_train_merged = pd.concat([pd.DataFrame(X_train_scale, columns=num_cols), X_train_enc_df], axis=1)
    X_test_merged = pd.concat([pd.DataFrame(X_test_scale, columns=num_cols), X_test_enc_df], axis=1)

    X_test = X_test_merged.dropna()
    X_train = X_train_merged.dropna()

    # Train the XGBoost model
    regressor = xgb.XGBRegressor(
        missing=0,
        booster='gbtree',
        objective="reg:squarederror",
        random_state=123,
        n_estimators=1000,
        learning_rate=0.1,
        max_depth=9,
        min_child_weight=3,
        gamma=0.0,
        colsample_bytree=0.3
    )

    regressor.fit(X_train, y_train)
    
    # Save the trained XGBoost model using joblib with a timestamp in the filename
    current_datetime = datetime.now()
    datestamp = current_datetime.strftime("%Y%m%d%H%M%S")
    model_filename = f'/opt/airflow/dags/models/xgb_model_{datestamp}.joblib'
    joblib.dump(regressor, model_filename)
    print(f"Trained XGBoost model saved as {model_filename}")
    return None
if __name__ == "__main__":
    train_model()
