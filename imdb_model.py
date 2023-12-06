from airflow import DAG
from airflow.providers.amazon.aws.operators.emr import EmrCreateJobFlowOperator, EmrAddStepsOperator, EmrTerminateJobFlowOperator
from airflow.providers.amazon.aws.sensors.emr import EmrStepSensor, EmrJobFlowSensor
from airflow.exceptions import AirflowSkipException
from airflow.utils.dates import days_ago
from airflow.models import Variable
import boto3
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.utils.trigger_rule import TriggerRule
import requests
import pandas as pd
import json
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import io
from pyarrow import parquet as pq
import logging
from datetime import timedelta
from emr_utils import cluster


AWS_ACCESS_KEY = Variable.get('AWS_ACCESS_KEY')
AWS_SECRET_KEY = Variable.get('AWS_SECRET_KEY')
BUCKET = Variable.get('BUCKET_NAME')
FEATURES_FOLDER = Variable.get('FEATURES_FOLDER')
MODEL = Variable.get('MODEL')
WORKER_NODES = int(Variable.get('MODEL_WORKER_NODES'))

s3 = boto3.client('s3',aws_access_key_id=AWS_ACCESS_KEY,aws_secret_access_key=AWS_SECRET_KEY)
emr = boto3.client('emr',aws_access_key_id=AWS_ACCESS_KEY,aws_secret_access_key=AWS_SECRET_KEY,region_name='us-east-2')

def create_emr_cluster(**kwargs):
    cluster_id = cluster.create(emr,'config/model.yml')
    if cluster_id:
        logging.info(f'Cluster {cluster_id} successfully created.')
        return cluster_id
    else:
        raise Exception('Cluster creation FAILED.')

def wait_for_cluster(**kwargs):
    ti = kwargs['ti']
    cluster_id = ti.xcom_pull(task_ids='create_emr_cluster')
    status_code = cluster.wait_launch(cluster_id,emr)

    if status_code == 200:
        logging.info(f'Cluster {cluster_id} LAUNCHED.')
        return cluster_id
    elif status_code == 410:
        raise Exception(f'Cluster {cluster_id} Launch FAILED.')
    else:
        raise Exception('UNKNOWN ERROR')

def model_training(**kwargs):
    script = f'dags/spark_scripts/{MODEL}.py'
    s3.upload_file(script,BUCKET,f'scripts/{MODEL}.py')
    ti = kwargs['ti']
    cluster_id = ti.xcom_pull(task_ids='create_emr_cluster')
    status_code = cluster.step(cluster_id,emr,f'{MODEL}',f's3://imdb-cs777/scripts/{MODEL}.py','')

    if status_code == 200:
        logging.info(f'{MODEL} SUCCESSFULL')
    else:
        raise Exception(f'{MODEL} FAILED.')

def model_evaluation(**kwargs):
    script = f'dags/spark_scripts/eval.py'
    s3.upload_file(script,BUCKET,f'scripts/eval.py')
    ti = kwargs['ti']
    cluster_id = ti.xcom_pull(task_ids='create_emr_cluster')
    status_code = cluster.step(cluster_id,emr,f'{MODEL}_eval',f's3://imdb-cs777/scripts/eval.py',f'{MODEL}')

    if status_code == 200:
        logging.info(f'{MODEL} evaluation SUCCESSFULL')
    else:
        raise Exception(f'{MODEL} evaluation FAILED.')

def terminate_cluster(**kwargs):
    ti = kwargs['ti']
    cluster_id = ti.xcom_pull(task_ids='create_emr_cluster')

    Response = cluster.terminate(cluster_id,emr)

    if Response:
        logging.info(f'Cluster {cluster_id} successfully TERMINATED.')
    else:
        raise Exception(f'Cluster Termination FAILED.')

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': days_ago(1)
}

dag = DAG(
    'imdb_model',
    default_args=default_args,
    description='DAG to train and evaluate the model',
    schedule_interval=None,
    start_date=days_ago(2),
)

create_emr_cluster_task = PythonOperator(
    task_id='create_emr_cluster',
    python_callable=create_emr_cluster,
    dag=dag,
    trigger_rule=TriggerRule.NONE_FAILED
)

wait_for_cluster_task = PythonOperator(
    task_id='wait_for_cluster',
    python_callable=wait_for_cluster,
    dag=dag,
    trigger_rule=TriggerRule.NONE_FAILED
)

model_training_task = PythonOperator(
    task_id=f'model_training',
    python_callable=model_training,
    dag=dag,
    trigger_rule=TriggerRule.NONE_FAILED
)

model_evaluation_task = PythonOperator(
    task_id=f'model_evaluation',
    python_callable=model_evaluation,
    dag=dag,
    trigger_rule=TriggerRule.NONE_FAILED
)

terminate_cluster_task = PythonOperator(
    task_id='terminate_emr_cluster',
    python_callable=terminate_cluster,
    dag=dag,
    trigger_rule = TriggerRule.ALL_DONE,
)

create_emr_cluster_task >> wait_for_cluster_task >> model_training_task >> model_evaluation_task >> terminate_cluster_task

if __name__ == "__main__":
    dag.cli()
