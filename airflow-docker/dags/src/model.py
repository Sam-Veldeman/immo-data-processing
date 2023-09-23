from pathlib import Path
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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
        return None

    # Data preprocessing
    cat_cols = ['type', 'postalcode', 'region', 'province']
    num_cols = ['construction_year', 'total_surface', 'habitable_surface', 'bedroom_count', 'terrace', 'garden_surface', 'facades', 'kitchen_equipped', 'condition_encoded']
    X = df.drop(columns=['price'], axis=1)
    y = df[['price']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=22)
    
    transformer = ColumnTransformer(
        transformers=[
            ('onehotencoder', OneHotEncoder(handle_unknown='ignore'), cat_cols),
            ('minmaxscaler', MinMaxScaler(), num_cols)
        ])
    # XGBoost model parameters
    regressor_args = {
        'missing':0,
        'booster':'gbtree',
        'objective':"reg:squarederror",
        'random_state':123,
        'n_estimators':1000,
        'learning_rate':0.1,
        'max_depth':9,
        'min_child_weight':3,
        'gamma':0.0,
        'colsample_bytree':0.3
    }
    # create the pipeline
    pipeline = Pipeline([
        ('preprocessor', transformer),
        ('regressor', XGBRegressor(**regressor_args))
    ])
    regressor = pipeline.fit(X_train, y_train)
    # Perform cross-validation
    cross_val_scores = cross_val_score(pipeline, X, y, cv=KFold(n_splits=5, shuffle=True, random_state=42),
                                       scoring='neg_mean_squared_error')
    
    # Calculate evaluation metrics for the entire dataset
    y_pred = pipeline.predict(X)
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    # Save the cross-validation results and evaluation metrics
    results = {
        'Cross-Validation MSE Scores': cross_val_scores,
        'MAE': mae,
        'MSE': mse,
        'R2': r2,
    }
    
    results_df = pd.DataFrame(results)
    
    
    # Save the pipeline using joblib with a timestamp in the filename
    current_datetime = datetime.now()
    datestamp = current_datetime.strftime("%Y%m%d%H%M%S")
    model_filename = f'/opt/airflow/dags/models/xgb_model_{datestamp}.joblib'
    
    joblib.dump(regressor, model_filename)
    print(f"Trained XGBoost model saved as {model_filename}")
    # Save the cross-validation results and evaluation metrics to a CSV file
    results_filename = f'/opt/airflow/dags/models/model_results_{datestamp}.csv'
    results_df.to_csv(results_filename, index=False)
    print(f"Cross-validation results and evaluation metrics saved as {results_filename}")

    return None
if __name__ == "__main__":
    train_model()