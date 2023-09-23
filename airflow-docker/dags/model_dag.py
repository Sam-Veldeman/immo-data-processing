from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import src.model  # Import the model.py script

# Define the DAG for model.py
model_dag = DAG(
    'model_training_dag',
    schedule_interval='@daily',  # Adjust as needed
    start_date=datetime(2023, 9, 17),  # Adjust as needed
    catchup=False,
)

# Define a Python function to run the model.py script
def train_model():
    src.model.train_model()

# Create a PythonOperator to run the model training function
train_model_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=model_dag,
)