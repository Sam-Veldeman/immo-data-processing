from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import src.clean_data  # Import the clean_data.py script
from pathlib import Path

# Define the DAG for clean_data.py
clean_data_dag = DAG(
    'clean_data_dag',
    schedule_interval='@daily',  # Adjust as needed
    start_date=datetime(2023, 9, 18),  # Adjust as needed
    catchup=False,
)

# Define a Python function to run the clean_data.py script
def run_clean_data():
    df = src.clean_data.run_cleanup()

# Create a PythonOperator to run the data cleaning function
clean_data_task = PythonOperator(
    task_id='clean_data',
    python_callable=run_clean_data,
    dag=clean_data_dag,
)

if __name__ == "__main__":
    clean_data_dag.cli()
