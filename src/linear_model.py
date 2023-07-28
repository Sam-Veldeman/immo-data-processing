import sys
sys.path.insert(0, '../')
import os
import joblib
import pickle
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import xgboost as xgb
from src.clean_data import run_cleanup
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
file_path = r'c:\Users\samve\OneDrive\0BeCode\repos\immo-data-processing\Data\house_details_v1.csv'
df = pd.read_csv(file_path, index_col='id', skip_blank_lines=True)
df, df_house, df_apt= run_cleanup(df)
df_cleaned= df
df_apt= df_apt.drop(columns= [
            'Type', 'Garden_surface', 'SwimmingPool', 'Condition', 'Postalcode', 'Street', 'Housenumber',
            'Box', 'City', 'Subtype', 'Location_area', 'Region', 'District', 'Province', 'Type_of_sale',
            'Garden', 'Kitchen_type', 'EPC_score', 'Latitude', 'Longitude', 'Property_url'
            ])
df_house= df_house.drop(columns= [
            'Type', 'Floor', 'Condition', 'Postalcode', 'Floor', 'Street', 'Housenumber', 'Box', 'City', 'Subtype',
            'Location_area', 'Region', 'District', 'Province', 'Type_of_sale', 'Garden', 'Kitchen_type',
            'EPC_score', 'Latitude','Longitude', 'Property_url'
            ])
df_cleaned= df_cleaned.drop(columns=[
            'Condition', 'Street', 'Housenumber', 'Box', 'City', 'Subtype', 'Location_area',
            'District', 'Type_of_sale', 'Garden', 'Kitchen_type', 'EPC_score', 'Latitude',
            'Longitude', 'Property_url', 'Floor', 'Furnished', 'Fireplace', 'Terrace_surface',
            'SwimmingPool'
            ])
def set_Xy(df):
    """
    Sets the variables X and y on a given DataFrame and selection of columns.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame containing the data.
        
    Returns:
        X (pd.DataFrame): All columns from df, without the target column ('Price')
        y (pd.DataFrame): Only the target column ('Price')
    """
    X = df.drop(columns=['Price'], axis=1)
    y = df[['Price']]
    return X, y

def split_data(X, y, df):
    cat_cols= ['Type', 'Postalcode', 'Region', 'Province']
    num_cols= ['Construction_year', 'Total_surface', 'Habitable_surface', 'Bedroom_count', 'Terrace', 'Garden_surface', 'Facades', 'Kitchen_equipped', 'Condition_encoded']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=22)

    encoder = OneHotEncoder(handle_unknown='ignore')
    X_train_enc = encoder.fit_transform(X_train[cat_cols])
    X_test_enc = encoder.transform(X_test[cat_cols])

    joblib.dump(encoder, '/users/samve/OneDrive/0BeCode/repos/immo-data-processing/models/encoder.joblib')

    scaler = MinMaxScaler()
    X_train_scale = scaler.fit_transform(X_train[num_cols])
    X_test_scale = scaler.transform(X_test[num_cols])

    joblib.dump(scaler, '/users/samve/OneDrive/0BeCode/repos/immo-data-processing/models/scaler.joblib')

    encoded_columns = encoder.get_feature_names_out(input_features=cat_cols)
    X_train_enc_df = pd.DataFrame(X_train_enc.toarray(), columns=encoded_columns)
    X_test_enc_df = pd.DataFrame(X_test_enc.toarray(), columns=encoded_columns)

    X_train_merged = pd.concat([pd.DataFrame(X_train_scale, columns=num_cols), X_train_enc_df], axis=1)
    X_test_merged = pd.concat([pd.DataFrame(X_test_scale, columns=num_cols), X_test_enc_df], axis=1)

    X_test= X_test_merged.dropna()
    X_train= X_train_merged.dropna()
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, model=1):
    """
    Trains a machine learning model on the provided data.

    Parameters:
        X_train (np.ndarray): Numpy array containing the feature variables for training.
        y_train (np.ndarray): Numpy array containing the target variable for training.
        model (int, optional): Model selection option. 
                               1 - XGBoost (default), 
                               2 - Linear Regression.

    Returns:
        regressor: Trained machine learning model.
    """
    if model == 1:
        regressor = xgb.XGBRegressor(missing= 0, booster='gbtree', objective="reg:squarederror", random_state=123, n_estimators=1000, learning_rate=0.1, max_depth=9, min_child_weight= 3, gamma= 0.0, colsample_bytree=0.3)       
    #'colsample_bytree': 0.3, 'gamma': 0.0, 'learning_rate': 0.1, 'max_depth': 9, 'min_child_weight': 3
    elif model == 2:
        regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    return regressor

def evaluate_model(regressor, X_test, y_test):
    """
    Evaluates the performance of the trained model on the test data.

    Parameters:
        regressor: Trained machine learning model.
        X_test (np.ndarray): Numpy array containing the feature variables for testing.
        y_test (np.ndarray): Numpy array containing the target variable for testing.

    Returns:
        score (float): R-squared score of the model.
        mse (float): Mean squared error of the model.
        y_pred (np.ndarray): Numpy array containing the predicted target variable.
    """
    y_pred = regressor.predict(X_test)
    score = regressor.score(X_test, y_test)  # Use X_test and y_test for scoring
    mse = mean_squared_error(y_true=y_test, y_pred=y_pred)
    return score, mse, y_pred

def cross_val(regressor, X_train, y_train):
    """
    Performs cross-validation on the model.

    Parameters:
        regressor: Trained machine learning model.
        X_train (np.ndarray): Numpy array containing the feature variables for training.
        y_train (np.ndarray): Numpy array containing the target variable for training.

    Returns:
        cv_scores (np.ndarray): Array of cross-validation scores.
        mean_cv_score (float): Mean of the cross-validation scores.
        std_cv_score (float): Standard deviation of the cross-validation scores.
    """
    cv_scores = cross_val_score(regressor, X_train, y_train, cv=5, scoring='r2')
    mean_cv_score = np.mean(cv_scores)
    std_cv_score = np.std(cv_scores)
    return cv_scores, mean_cv_score, std_cv_score

def create_plot(y_test, y_pred):
    """
    Creates an interactive scatter plot of actual prices vs. predicted prices.

    Parameters:
        y_test (np.ndarray): Numpy array containing the actual target variable (test data).
        y_pred (np.ndarray): Numpy array containing the predicted target variable.

    Returns:
        fig: Plotly Figure object containing the scatter plot.
    """
    # Calculate the absolute difference between the price and the predicted price
    absolute_difference = abs(y_test.flatten() - y_pred)
    # Create a list to store the colors for the dots based on the condition
    dot_colors = ['green' if diff < 0.1 * price else 'blue' for price, diff in zip(y_test.flatten(), absolute_difference)]
    # Create an interactive scatter plot of actual prices vs. predicted prices
    fig = go.Figure()
    # Add actual prices and predicted prices as scatter plot traces with custom colors
    fig.add_trace(go.Scattergl(x=y_test.flatten(),
                                y=y_pred,
                                mode='markers',
                                marker=dict(color=dot_colors, size=8),
                                name='Predicted vs. Actual Prices',
                                text=[f'Data Point Index: {i}' for i in np.arange(len(y_test))],
                                hoverinfo='text+x+y'))
    # Add a 1:1 line (y=x) to indicate perfect predictions
    diagonal_line = go.Scatter(x=[min(y_test.flatten()), max(y_test.flatten())],
                            y=[min(y_test.flatten()), max(y_test.flatten())],
                            mode='lines',
                            line=dict(color='red', dash='solid'),
                            name='Perfect Prediction')
    fig.add_trace(diagonal_line)
    # Update layout
    fig.update_layout(title='Actual vs. Predicted Prices (Test Data)',
                    xaxis_title='Actual Price',
                    yaxis_title='Predicted Price',
                    showlegend=True,
                    hovermode='closest'
                    )
    return fig
def save_plot(fig):
    """
    Saves the plot as an interactive HTML file.

    Parameters:
        fig: Plotly Figure object to be saved.

    Returns:
        None
    """
    output_folder = '../output/'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_file_path = os.path.join(output_folder, 'actual_vs_predicted_scatterplot.html')
    fig.write_html(output_file_path)
    print(f"Interactive scatter plot saved as an HTML at: {output_file_path}")

def get_model_input():
    """
    Gets user input for choosing the machine learning model.

    Returns:
        model_number (int): Selected model number. 
                            1 - XGBoost (default), 
                            2 - Linear Regression.
    """
    while True:
        model_input = input("Choose the model:\n1 = XGBoost\n2 = Linear Regression\n[No input for default(1)]: ")
        if not model_input:
            return 1
        elif model_input.isdigit():
            model_number = int(model_input)
            if model_number in [1, 2]:
                return model_number
        print("Invalid input. Please enter 1 or 2.")

def get_df_input():
    """
    Gets user input for choosing the DataFrame (houses, apartments, or entire DataFrame).

    Returns:
        df_choice (int): Selected DataFrame choice number. 
                         1 - Entire DataFrame (default),
                         2 - Houses,
                         3 - Apartments.
    """
    while True:
        df_input = input("Choose the DataFrame (1 for entire DataFrame, 2 for houses, 3 for apartments) [Default is 1]: ")
        if not df_input:
            return 1
        elif df_input.isdigit():
            df_choice = int(df_input)
            if df_choice in [1, 2, 3]:
                return df_choice
        print("Invalid input. Please enter 1, 2, or 3.")

def get_save_input(regressor, model_name):
    """
    Gets user input for choosing whether to save the plot as an HTML file.

    Parameters:
        fig: Plotly Figure object to be saved.

    Returns:
        None
    """
    
    while True:
        save_choice = input("Do you want to save the model as a pickle file? (y/n): ").lower()
        if save_choice == 'y':
            save_model(regressor, model_name)
            break
        elif save_choice == 'n':
            break
        else:
            print("Invalid input. Please enter 'y' or 'n'.")
def select_df(df_input):
    """
    Selects the appropriate DataFrame based on the user's choice.

    Parameters:
        df_input (int): DataFrame choice number. 
                        1 - Entire DataFrame,
                        2 - Houses,
                        3 - Apartments.

    Returns:
        df (pd.DataFrame): Selected DataFrame based on the user's choice.
        columns (list): List of column names based on the selected DataFrame.
    """
    if df_input == 1:
        df = df_cleaned
        model_name= 'df_cleaned'
    elif df_input == 2:
        df = df_house
        model_name= 'df_house'
    else:
        df = df_apt
        model_name= 'df_apt'
    return df, model_name

def save_model(regressor, model_name):
    """
    Save the trained model to a file using pickle.

    Parameters:
        regressor: Trained machine learning model.
        model_name (str): Name of the model to be saved (e.g., 'df', 'df_house', 'df_apt').

    Returns:
        None
    """
    save_path = '/users/samve/OneDrive/0BeCode/repos/immo-data-processing/models/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model_filename = os.path.join(save_path, f"{model_name}.pkl")
    
    try:
        with open(model_filename, "wb") as file:
            pickle.dump(regressor, file)
        print(f"Trained {model_name} model saved as: {model_filename}")
    except Exception as e:
        print(f"Error saving the model: {e}")
            
def model(df_input, model=1):
    """
    Performs the entire modeling process based on user input.

    Parameters:
        df_input (int): DataFrame choice number. 
                        1 - Entire DataFrame,
                        2 - Houses,
                        3 - Apartments.
        model (int, optional): Model selection option. 
                               1 - XGBoost (default), 
                               2 - Linear Regression.
        scaled (bool, optional): Whether to scale the data using MinMaxScaler. Default is True.

    Returns:
        regressor: Trained machine learning model.
        score (float): R-squared score of the model on the test data.
        mse (float): Mean squared error of the model on the test data.
        cv_scores (np.ndarray): Array of cross-validation scores.
        mean_cv_score (float): Mean of the cross-validation scores.
        std_cv_score (float): Standard deviation of the cross-validation scores.
        fig: Plotly Figure object containing the interactive scatter plot.
    """
    df, model_name= select_df(df_input)
    X, y = set_Xy(df)
    X_train, X_test, y_train, y_test= split_data(X, y, df)
    regressor = train_model(X_train, y_train, model=model)
    score, mse, y_pred = evaluate_model(regressor, X_test, y_test)
    cv_scores, mean_cv_score, std_cv_score = cross_val(regressor, X_train, y_train)
    #fig = create_plot(y_test, y_pred)
    get_save_input(regressor, model_name)
    return score, mse, cv_scores, mean_cv_score, std_cv_score