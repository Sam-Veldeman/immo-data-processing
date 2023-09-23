from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from src.scrape import run_scraper as scrape_function  # Import the scrape.py script

# Define the DAG for scrape.py
scrape_dag = DAG(
    'scrape_data_dag',
    schedule_interval='@daily',  # Adjust as needed
    start_date=datetime(2023, 9, 17),  # Adjust as needed
    catchup=False,
)

# Define a Python function to run the scrape.py script
def run_scrape_task():
    scrape_function(num_pages=2, num_workers=10)

# Create a PythonOperator to run the scraping function
scrape_task = PythonOperator(
    task_id='scrape_data',
    python_callable=run_scrape_task,  # Use the renamed function
    dag=scrape_dag,
)