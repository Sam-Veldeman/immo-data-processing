import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn.model_selection import cross_val_score

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
        regressor = LinearRegression()
    elif model == 2:
        regressor = xgb.XGBRegressor(booster='gbtree', objective="reg:squarederror", random_state=123, n_estimators=2000, learning_rate=0.02, max_depth=6)
    regressor.fit(X_train, y_train)
    return regressor

def evaluate_model(regressor, X_test, y_test):
    y_pred = regressor.predict(X_test)
    score = regressor.score(X_test, y_test)  # Use X_test and y_test for scoring
    MSE = mean_squared_error(y_true=y_test, y_pred=y_pred)
    return score, MSE

def cross_val(regressor, X_train, y_train):
    cv_scores = cross_val_score(regressor, X_train, y_train, cv=5, scoring='r2')
    mean_cv_score = np.mean(cv_scores)
    std_cv_score = np.std(cv_scores)
    return cv_scores, mean_cv_score, std_cv_score

def model(df, columns, model=1, scaled=True):
    X, y = set_Xy(df, columns)
    X_train, X_test, y_train, y_test = split_data(X, y, scaled=scaled)
    regressor = train_model(X_train, y_train, model=model)
    score, MSE = evaluate_model(regressor, X_test, y_test)
    cv_scores, mean_cv_score, std_cv_score = cross_val(regressor, X_train, y_train)
    print(f'The score for the model is: {score}\nThe MSE for this model is {MSE}')
    print("Cross-validation R2 scores:", cv_scores)
    print("Mean cross-validation R2 score:", mean_cv_score)
    print("Standard deviation of cross-validation R2 scores:", std_cv_score)