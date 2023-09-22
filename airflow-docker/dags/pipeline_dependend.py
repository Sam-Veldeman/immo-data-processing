from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from pathlib import Path
from src import scrape
from src import clean_data
from src import model


# Define the complete data pipeline DAG
data_pipeline_dag = DAG(
    'complete_data_pipeline',
    schedule_interval='@daily',  # Adjust as needed
    start_date=datetime(2023, 9, 18),  # Adjust as needed
    catchup=False,
)

# Define a Python function to run the entire data pipeline
def run_data_pipeline():
    # Step 1: Run the data scraping script
    scrape.run_scraper(num_pages=2, num_workers=10)

# Create a PythonOperator for data scraping
data_scraping_task = PythonOperator(
    task_id='run_data_scraping',
    python_callable=run_data_pipeline,
    dag=data_pipeline_dag,
)

# Define a Python function to run the data cleaning step
def run_data_cleaning():
    # Step 2: Run the data cleaning script
    df = clean_data.run_cleanup()
    
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

# Define task dependencies
data_scraping_task >> data_cleaning_task
data_cleaning_task >> model_training_task

if __name__ == "__main__":
    data_pipeline_dag.cli()
