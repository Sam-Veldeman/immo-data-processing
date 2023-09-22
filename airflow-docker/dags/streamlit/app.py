import streamlit as st
import pandas as pd
import os
from pathlib import Path

def get_latest_scraped_data_csv(data_dir, models_dir):
    data_dir = Path(data_dir)
    models_dir = Path(models_dir)
    csv_files = list(data_dir.glob('scraped_data_*.csv'))
    model = list(models_dir.glob(''))
    if csv_files:
        latest_csv = max(csv_files, key=os.path.getctime)
        return latest_csv
    else:
        return None

# Define the directory containing scraped data CSV files
data_directory = '/opt/airflow/dags/Data'
models_directory = 'opt/airflow/dags/models'

# Get the path to the latest scraped_data CSV file
latest_scraped_data_csv = get_latest_scraped_data_csv(data_directory, models_directory)

if latest_scraped_data_csv:
    # Define the input file path using the latest scraped_data CSV file
    input_file_path = latest_scraped_data_csv
else:
    print("No scraped_data CSV file found. Please run the scraper script first.")
    exit(1)
df = pd.read_csv(input_file_path, index_col='id', skip_blank_lines=True)

st.title('Immo Eliza web ui')
st.subheader('This model will predict the listings price in €')
#st.subheader("Train Set Score: {}".format ( round(train_score,3)))
#st.subheader("Test Set Score: {}".format(round(test_score,3)))
name = st.text_input("Name of Passenger ")
sex = st.selectbox("Sex",options=['Male' , 'Female'])
age = st.slider("Age", 1, 100,1)
p_class = st.selectbox("Passenger Class",options=['First Class' , 'Second Class' , 'Third Class'])
