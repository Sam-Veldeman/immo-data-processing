FROM apache/airflow:latest-python3.10
COPY requirements.txt /requirements.txt
RUN pip install --user --upgrade pip
RUN pip install apache-airflow[amazon]
RUN pip install --no-cache-dir --user -r /requirements.txt
