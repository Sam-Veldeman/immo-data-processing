import sys
sys.path.insert(0, '../')
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import nbformat
import xgboost as xgb
from src.clean_data import run_cleanup
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
file_path = r'c:\Users\samve\OneDrive\0BeCode\repos\immo-data-processing\Data\Filtered_Data\house_details_v1.csv'
df = pd.read_csv(file_path, index_col='id', skip_blank_lines=True)
df_cleaned, df_house, df_apt= run_cleanup(df)
df_columns = [
            'Habitable surface', 'Bedroom count', 'Postalcode', 'Terrace surface', 'Garden surface', 'Kitchen equipped',
            'Construction year', 'Total surface', 'Garden surface', 'Facades'
            ]
house_columns = ['Habitable surface', 'Bedroom count', 'Terrace surface', 'Garden surface', 'Kitchen equipped', 'Facades', 'Postalcode']
apt_columns = ['Habitable surface', 'Bedroom count', 'Terrace surface', 'Kitchen equipped', 'Floor', 'Postalcode']

def set_Xy(df, columns):
    """"
    
    Sets the variables X and y on a given df and selection of columns

    Returns:
        X_train, X_split, y_train, y_split
    
    """
    X = df[columns].to_numpy()
    if X.shape[0] == 1:
        X = X.reshape(-1,1)
        X = np.hstack((X, ones))
    else:
        ones = np.ones((X.shape[0],1))
        X = np.hstack((X, ones))
    y = df[['Price']].to_numpy().reshape(-1,1)
    return X, y

def split_data(X, y, scaled=True):
    """"
    
    Splits the data by train_test_split()
    By default scaled = True, it will use the sklearn MinMaxScaler() to return scaled values.

    Returns:
        X_train, X_split, y_train, y_split
    
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=123)
    if scaled == True:
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, model=1):
    """"
    
    Given a model number, trains the model and returns the trained model.
    If no model number is given, standard model will be sklearn LinearRegression()
    Optional models:
        1: LinearRegression
        2: XGBoost

    Returns:
        regressor
    """
    if model == 1:
        regressor = xgb.XGBRegressor(booster='gbtree', objective="reg:squarederror", random_state=123, n_estimators=2000, learning_rate=0.02, max_depth=6)
    elif model == 2:
        regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    return regressor

def evaluate_model(regressor, X_test, y_test):
    y_pred = regressor.predict(X_test)
    score = regressor.score(X_test, y_test)  # Use X_test and y_test for scoring
    mse = mean_squared_error(y_true=y_test, y_pred=y_pred)
    return score, mse, y_pred

def cross_val(regressor, X_train, y_train):
    cv_scores = cross_val_score(regressor, X_train, y_train, cv=5, scoring='r2')
    mean_cv_score = np.mean(cv_scores)
    std_cv_score = np.std(cv_scores)
    return cv_scores, mean_cv_score, std_cv_score

def create_plot(y_test, y_pred):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_test.flatten(), y=y_pred, mode='markers', name='Predicted'))
    fig.add_trace(go.Scatter(x=y_test.flatten(), y=y_test.flatten(), mode='lines', name='Actual'))
    fig.update_layout(title='Regression Model Results', xaxis_title='Price', yaxis_title='Predicted Value')
    return fig

# Function to get user input for the model number
def get_model_input():
    while True:
        model_input = input("Choose the model:\n1 = XGBoost\n2 = Linear Regression\n[No input for default(1)]: ")
        if not model_input:
            return 1
        elif model_input.isdigit():
            model_number = int(model_input)
            if model_number in [1, 2]:
                return model_number
        print("Invalid input. Please enter 1 or 2.")

# Function to get user input for scaling choice
def get_scaled_input():
    while True:
        scale_input = input("Do you want to scale the data? (y/n) [Default is y]: ")
        if not scale_input:
            return True
        elif scale_input.lower() in ['y', 'n']:
            return True if scale_input.lower() == 'y' else False
        print("Invalid input. Please enter 'y' or 'n'.")

# Function to get user input for DataFrame choice
def get_df_input():
    while True:
        df_input = input("Choose the DataFrame (1 for entire DataFrame, 2 for houses, 3 for apartments) [Default is 1]: ")
        if not df_input:
            return 1
        elif df_input.isdigit():
            df_choice = int(df_input)
            if df_choice in [1, 2, 3]:
                return df_choice
        print("Invalid input. Please enter 1, 2, or 3.")

def model(df_input, model=1, scaled=True):
    if df_input == 1:
        df = df_cleaned
        columns = df_columns
    elif df_input == 2:
        df = df_house
        columns = house_columns
    else:
        df = df_apt
        columns = apt_columns
    X, y = set_Xy(df, columns)
    X_train, X_test, y_train, y_test = split_data(X, y, scaled=scaled)
    regressor = train_model(X_train, y_train, model=model)
    score, mse, y_pred = evaluate_model(regressor, X_test, y_test)
    cv_scores, mean_cv_score, std_cv_score = cross_val(regressor, X_train, y_train)
    fig = create_plot(y_test, y_pred)
    return regressor, score, mse, cv_scores, mean_cv_score, std_cv_score, fig