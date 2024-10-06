import os
from airflow import DAG
from airflow.operators.dummy_operator import DummyOperator
from airflow.contrib.operators.spark_submit_operator import SparkSubmitOperator
from datetime import datetime, timedelta

spark_conn = os.environ.get("spark_conn", "spark_conn")
spark_master = "spark://spark:7077"
postgres_driver_jar = "/usr/local/spark/assets/jars/postgresql-42.2.6.jar"

remote_work_file = "/usr/local/spark/assets/data/user_log.csv"
postgres_db = "jdbc:postgresql://postgres:5432/airflow"
postgres_user = "airflow"
postgres_pwd = "airflow"

now = datetime.now()

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(now.year, now.month, now.day),
    "email": ["airflow@airflow.com"],
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=1)
}

dag = DAG(
    dag_id="hand",
    description="DAG to load score top 5.",
    default_args=default_args,
    # schedule_interval=timedelta(1)
    schedule_interval='0 0 * * 0'
)

# Define tasks
start = DummyOperator(task_id="start", dag=dag)

# Task 00: Upload data to PostgreSQL
load_postgres = SparkSubmitOperator(
    task_id="load_postgres",
    application="/usr/local/spark/applications/load-postgres.py",
    name="upload_data_to_postgres",
    conn_id="spark_conn",
    verbose=1,
    conf={"spark.master": spark_master},
    application_args=[remote_work_file, postgres_db, postgres_user, postgres_pwd],
    jars=postgres_driver_jar,
    driver_class_path=postgres_driver_jar,
    dag=dag
)

# Task 01: Calculate Industry(IT) Hours Worked Per Week
read_postgres = SparkSubmitOperator(
    task_id="read_postgres",
    application="/usr/local/spark/applications/read-postgres.py",
    name="calculate_industry_it_hours_worked_per_week",
    conn_id="spark_conn",
    verbose=1,
    conf={"spark.master": spark_master},
    application_args=[postgres_db, postgres_user, postgres_pwd],
    jars=postgres_driver_jar,
    driver_class_path=postgres_driver_jar,
    dag=dag
)

end = DummyOperator(task_id="end", dag=dag)

# Task dependencies
start >> load_postgres >> read_postgres >> end
# start >> load_postgres >> end
