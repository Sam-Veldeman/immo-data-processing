from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime
import scrape
import clean_data
import model

# Define the complete data pipeline DAG
data_pipeline_dag = DAG(
    'complete_data_pipeline',
    schedule_interval='@daily',
    start_date=datetime(2023, 9, 18),
    catchup=False,
)

# Create a Python function for the placeholder task
def do_nothing():
    pass

# Create a PythonOperator as a placeholder
start_task = PythonOperator(
    task_id='start_data_pipeline',
    python_callable=do_nothing,
    dag=data_pipeline_dag,
)

# Define a Python function to run the entire data pipeline
def run_data_pipeline():
    # Step 1: Run the data scraping script
    scrape.run_scraper(num_pages=333, num_workers=10)

# Create a PythonOperator for data scraping
data_scraping_task = PythonOperator(
    task_id='run_data_scraping',
    python_callable=run_data_pipeline,
    dag=data_pipeline_dag,
)

# Define a Python function to run the data cleaning step
def run_data_cleaning():
    # Step 2: Run the data cleaning script
    clean_data.run_cleanup()
    
# Create a PythonOperator for data cleaning
data_cleaning_task = PythonOperator(
    task_id='run_data_cleaning',
    python_callable=run_data_cleaning,
    dag=data_pipeline_dag,
)

# Define a Python function to run the model training step
def run_model_training():
    # Step 3: Run the model training script
    model.train_model()

# Create a PythonOperator for model training
model_training_task = PythonOperator(
    task_id='run_model_training',
    python_callable=run_model_training,
    dag=data_pipeline_dag,
)

# Define the command to run the Streamlit app from the /streamlit folder
streamlit_command = "streamlit run /opt/airflow/streamlit/app.py"

# Create a BashOperator to start the Streamlit app
streamlit_task = BashOperator(
    task_id='run_streamlit_app',
    bash_command=streamlit_command,
    dag=data_pipeline_dag,
)

# Define task dependencies
start_task >> data_scraping_task
data_scraping_task >> data_cleaning_task
data_cleaning_task >> model_training_task
data_cleaning_task >> streamlit_task
model_training_task >> streamlit_task
