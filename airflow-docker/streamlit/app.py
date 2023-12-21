import streamlit as st
import pandas as pd
import os
import numpy as np
from pathlib import Path
import joblib
import plotly.express as px



def get_latest_scraped_data_csv(data_dir):
    data_dir = Path(data_dir)
    csv_files = list(data_dir.glob('scraped_data_*.csv'))
    if csv_files:
        latest_csv = max(csv_files, key=os.path.getctime)
        return latest_csv
    else:
        return None
def get_latest_model(data_dir, models_dir):
    data_dir = Path(data_dir)
    models_dir = Path(models_dir)
    csv_file = list(data_dir.glob('cleaned_data_*.csv'))
    model_file = list(models_dir.glob('xgb_model_*.joblib'))
    results_file = list(models_dir.glob('model_results_*.csv'))
    if csv_file and model_file:
        latest_csv = max(csv_file, key=os.path.getctime)
        latest_model = max(model_file, key=os.path.getctime)
        latest_results = max(results_file, key=os.path.getctime)
        return latest_csv, latest_model, latest_results
    else:
        return None

# Define the directory containing scraped data CSV files
data_directory = '/opt/airflow/Data'
models_directory = '/opt/airflow/models'

# Get the path to the latest scraped_data CSV file
latest_scraped_data_csv = get_latest_scraped_data_csv(data_directory)
latest_cleaned_data_csv, model_file, results_file = get_latest_model(data_directory, models_directory)

if latest_scraped_data_csv:
    # Define the input file path using the latest scraped_data CSV file
    
    df = pd.read_csv(latest_scraped_data_csv, index_col='id', skip_blank_lines=True)
else:
    print("No scraped_data CSV file found. Please run the scraper script first.")
    exit(1)
if latest_cleaned_data_csv:
    # Define the input file path using the latest scraped_data CSV file
    df_results = pd.read_csv(results_file)
    df_pred = pd.read_csv(latest_cleaned_data_csv, index_col='id', skip_blank_lines=True)
    model = joblib.load(model_file)
else:
    print("No scraped_data CSV file found. Please run the scraper script first.")
    exit(1)
# Page title and navigation
st.title("Real Estate Analyzer")
page = st.selectbox("Select a page:", ["Data Analysis", "Price Prediction"])

if page == "Data Analysis":
    # Add data analysis code and plots here
    st.header("Data Analysis")

    # Additional Tools
    st.subheader("Data Exploration Tools")

    # Slider for filtering data
    st.sidebar.subheader("Filter Data")
    min_price = st.sidebar.slider("Minimum Price", min_value=df['price'].min(), max_value=df['price'].max(), value=df['price'].min())
    max_price = st.sidebar.slider("Maximum Price", min_value=df['price'].min(), max_value=df['price'].max(), value=df['price'].max())

    # Slider for filtering data by total surface
    min_surface = st.sidebar.slider("Minimum Total Surface", min_value=df['total_surface'].min(), max_value=df['total_surface'].max(), value=df['total_surface'].min())
    max_surface = st.sidebar.slider("Maximum Total Surface", min_value=df['total_surface'].min(), max_value=df['total_surface'].max(), value=df['total_surface'].max())

    # Apply filters to DataFrame
    filtered_df = df[(df['price'] >= min_price) & (df['price'] <= max_price) & (df['total_surface'] >= min_surface) & (df['total_surface'] <= max_surface)]

    # Display filtered data
    st.write(f"Displaying data for prices between € {min_price} and € {max_price} and total surface between {min_surface} and {max_surface} sqm")
    st.write(filtered_df)

    # Additional Plots
    st.subheader("Additional Plots")

    # Histogram of prices with options
    histogram_options = st.checkbox("Show Histogram Options")
    if histogram_options:
        bins = st.slider("Number of Bins", min_value=1, max_value=100, value=20)
        color = st.color_picker("Select Color", value="#3498db")
        title = st.text_input("Plot Title", "Price Distribution")

        st.plotly_chart(px.histogram(filtered_df, x='price', title=title, nbins=bins, color_discrete_sequence=[color]))

    # Box plot of prices by region with options
    boxplot_options = st.checkbox("Show Box Plot Options")
    if boxplot_options:
        y_col = st.selectbox("Select Y-Axis", ['price', 'total_surface', 'bedroom_count'])
        color_col = st.selectbox("Select Color Column", ['region', 'province'])
        title = st.text_input("Plot Title", "Price Distribution by Region")

        st.plotly_chart(px.box(filtered_df, x='region', y=y_col, color=color_col, title=title))

    # Scatter plot of surface vs. price with options
    scatter_options = st.checkbox("Show Scatter Plot Options")
    if scatter_options:
        x_col = st.selectbox("Select X-Axis", ['total_surface', 'bedroom_count'])
        color_col = st.selectbox("Select Color Column", ['region', 'province'])
        title = st.text_input("Plot Title", "Total Surface vs. Price")

        st.plotly_chart(px.scatter(filtered_df, x=x_col, y='price', color=color_col, title=title))

elif page == "Price Prediction":
    # Input fields for categorical features
    cat_cols = ['type', 'postalcode', 'region', 'province']
    selected_cat_values = {}
    for col in cat_cols:
        selected_cat_values[col] = st.selectbox(f"Select {col}:", options=df_pred[col].unique())

    # Input fields for numerical features
    num_cols = ['construction_year', 'total_surface', 'habitable_surface', 'bedroom_count', 'terrace', 'garden_surface', 'facades', 'kitchen_equipped', 'condition_encoded']
    selected_num_values = {}
    for col in num_cols:
        selected_num_values[col] = st.number_input(f"Enter {col}:", min_value=0)

    # Button to make predictions
    if st.button("Predict Price"):
        # Prepare input data as a DataFrame
        input_data = pd.DataFrame({**selected_cat_values, **selected_num_values}, index=[0])
        # Use the trained model to make predictions
        predicted_price = model.predict(input_data)
        st.subheader("Predicted Price:")
        st.write(f"€ {predicted_price[0]:,.2f}")
        # Display the latest evaluation results
        st.subheader("Latest Evaluation Results Model:")
        st.write(df_results)

        # Display specific evaluation metrics
        mae = df_results['MAE'].values[0]
        mse = df_results['MSE'].values[0]
        r2 = df_results['R2'].values[0]

        st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
        st.write(f"Mean Squared Error (MSE): {mse:.2f}")
        st.write(f"R-squared (R2): {r2:.2f}")
    else:
        st.write("No evaluation results found.")