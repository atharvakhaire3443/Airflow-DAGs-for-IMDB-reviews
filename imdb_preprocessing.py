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

RUN_MODE = Variable.get('MODE')
FILE_LIST = Variable.get('FILE_LIST')
AWS_ACCESS_KEY = Variable.get('AWS_ACCESS_KEY')
AWS_SECRET_KEY = Variable.get('AWS_SECRET_KEY')
BUCKET = Variable.get('BUCKET_NAME')
RAW_FOLDER = Variable.get('RAW_FOLDER')
RAW_FILE_PATH = f's3://{BUCKET}/{RAW_FOLDER}'

s3 = boto3.client('s3',aws_access_key_id=AWS_ACCESS_KEY,aws_secret_access_key=AWS_SECRET_KEY)
emr = boto3.client('emr',aws_access_key_id=AWS_ACCESS_KEY,aws_secret_access_key=AWS_SECRET_KEY,region_name='us-east-2')

def check_for_raw_files(**kwargs):
    response = s3.list_objects_v2(Bucket=BUCKET, Prefix=RAW_FOLDER+'/')
    file_list = []
    if 'Contents' in response:
        for item in response['Contents']:
            file_list.append(item['Key'])
    if RUN_MODE == 'full':
        if len(file_list) > 0:
            logging.info(f'Files Found: {file_list}')
        else:
            raise Exception('No Files found')
    elif RUN_MODE == 'select':
        flag = True
        file_list_temp = []
        for file in FILE_LIST.split(','):
            if RAW_FOLDER + '/' + file not in file_list:
                flag = False
            else:
                file_list_temp.append(RAW_FOLDER + '/' + file)
        file_list = file_list_temp
        if flag:
            logging.info(f"Files Found: {file_list}")
        else:
            raise Exception('One or more files not found.')
    else:
        raise Exception('Incorrect RUN MODE')
    result = s3.list_objects_v2(Bucket=BUCKET, Prefix='raw_imdb_reviews')
    logging.info(f'{result}')
    if 'Contents' in result:
        logging.info(f"Folder 'raw_imdb_reviews' exists in bucket '{BUCKET}'.")
    else:
        s3.put_object(Bucket=BUCKET, Key=('raw_imdb_reviews/'))
        logging.info(f"Folder 'raw_imdb_reviews' created in bucket '{BUCKET}'.")
    return file_list

def check_for_FULL_file_in_s3(**kwargs):
    response = s3.list_objects_v2(Bucket=BUCKET, Prefix='raw_imdb_reviews')
    if RUN_MODE == 'full':
        temp_list = []
        if 'Contents' in response:
            for item in response['Contents']:
                temp_list.append(item['Key'])
            logging.info(f'{temp_list}')
        if 'raw_imdb_reviews/raw_imdb_reviews_FULL.parquet' in temp_list:
            return 'create_emr_cluster'
        else:
            return 'json_to_parquet_conversion'

def json_to_parquet_conversion(**kwargs):
    ti = kwargs['ti']
    file_list = ti.xcom_pull(task_ids='check_for_raw_files_in_s3')
    logging.info(f'{file_list}')
    combined_data = []
    for file in file_list:
        file_object = s3.get_object(Bucket=BUCKET, Key=file)
        file_content = file_object['Body'].read().decode('utf-8')
        data = json.loads(file_content)
        logging.info(f'{file} successfully parsed.')
        combined_data = combined_data + data
    logging.info('Files merged')
    df = pd.DataFrame({'review_json': combined_data})
    output_buffer = io.BytesIO()
    df.to_parquet(output_buffer, index=False)
    output_buffer.seek(0)
    if RUN_MODE == 'full':
        s3.put_object(Bucket=BUCKET,Key='raw_imdb_reviews/raw_imdb_reviews_FULL.parquet',Body=output_buffer)
    else:
        s3.put_object(Bucket=BUCKET,Key='raw_imdb_reviews/raw_imdb_reviews_SELECT.parquet',Body=output_buffer)

def create_emr_cluster(**kwargs):
    cluster_id = cluster.create(emr,'config/preprocessing.yml')
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

def transform_json_to_columns(**kwargs):
    script = 'dags/spark_scripts/transform.py'
    s3.upload_file(script,BUCKET,'scripts/transform.py')
    ti = kwargs['ti']
    cluster_id = ti.xcom_pull(task_ids='create_emr_cluster')
    status_code = cluster.step(cluster_id,emr,'transform_json_to_columns','s3://imdb-cs777/scripts/transform.py','')

    if status_code == 200:
        logging.info('Transformations SUCCESSFULL')
    else:
        raise Exception('Transformations FAILED.')

def preprocessing(**kwargs):
    script = 'dags/spark_scripts/preprocessing.py'
    s3.upload_file(script,BUCKET,'scripts/preprocessing.py')
    ti = kwargs['ti']
    cluster_id = ti.xcom_pull(task_ids='create_emr_cluster')
    status_code = cluster.step(cluster_id,emr,'preprocessing','s3://imdb-cs777/scripts/preprocessing.py','')

    if status_code == 200:
        logging.info('Preprocessing SUCCESSFULL')
    else:
        raise Exception('Preprocessing FAILED.')

def extract_tf_idf(**kwargs):
    script = 'dags/spark_scripts/extract_tf_idf.py'
    s3.upload_file(script,BUCKET,'scripts/extract_tf_idf.py')
    ti = kwargs['ti']
    cluster_id = ti.xcom_pull(task_ids='create_emr_cluster')
    status_code = cluster.step(cluster_id,emr,'extract_tf_idf','s3://imdb-cs777/scripts/extract_tf_idf.py','')

    if status_code == 200:
        logging.info('TFIDF Extraction SUCCESSFULL')
    else:
        raise Exception('TFIDF Extraction FAILED.')

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
    'imdb_preprocessing',
    default_args=default_args,
    description='DAG to transform, preprocess and extract tf-idf features of the reviews',
    schedule_interval=None,
    start_date=days_ago(2),
)

check_for_raw_files_task = PythonOperator(
    task_id='check_for_raw_files_in_s3',
    python_callable=check_for_raw_files,
    dag=dag,
)

check_for_FULL_file_in_s3_task = BranchPythonOperator(
    task_id='check_for_FULL_file_in_s3',
    python_callable=check_for_FULL_file_in_s3,
    dag=dag,
)

json_to_parquet_conversion_task = PythonOperator(
    task_id='json_to_parquet_conversion',
    python_callable=json_to_parquet_conversion,
    dag=dag,
)

create_emr_cluster_task = PythonOperator(
    task_id='create_emr_cluster',
    python_callable=create_emr_cluster,
    dag=dag,
    trigger_rule=TriggerRule.NONE_FAILED
)

transform_json_to_columns_task = PythonOperator(
    task_id='transform_json_to_columns',
    python_callable=transform_json_to_columns,
    dag=dag,
    trigger_rule=TriggerRule.NONE_FAILED
)

wait_for_cluster_task = PythonOperator(
    task_id='wait_for_cluster',
    python_callable=wait_for_cluster,
    dag=dag,
    trigger_rule=TriggerRule.NONE_FAILED
)

preprocessing_task = PythonOperator(
    task_id='preprocessing',
    python_callable=preprocessing,
    dag=dag,
    trigger_rule=TriggerRule.NONE_FAILED
)

extract_tf_idf_task = PythonOperator(
    task_id='extract_tf_idf',
    python_callable=extract_tf_idf,
    dag=dag,
    trigger_rule=TriggerRule.NONE_FAILED
)

terminate_cluster_task = PythonOperator(
    task_id='terminate_emr_cluster',
    python_callable=terminate_cluster,
    dag=dag,
    trigger_rule=TriggerRule.ALL_DONE
)

# check_for_raw_files_task >> check_for_FULL_file_in_s3_task
# check_for_FULL_file_in_s3_task >> [json_to_parquet_conversion_task,create_emr_cluster_task]
# json_to_parquet_conversion_task >> create_emr_cluster_task
# create_emr_cluster_task >> wait_for_cluster_task >> terminate_cluster_task

check_for_raw_files_task >> check_for_FULL_file_in_s3_task

check_for_FULL_file_in_s3_task >> json_to_parquet_conversion_task
check_for_FULL_file_in_s3_task >> create_emr_cluster_task

json_to_parquet_conversion_task >> create_emr_cluster_task
create_emr_cluster_task >> wait_for_cluster_task >> transform_json_to_columns_task
transform_json_to_columns_task >> preprocessing_task >> extract_tf_idf_task >> terminate_cluster_task

create_emr_cluster_task.set_downstream(wait_for_cluster_task)

if __name__ == "__main__":
    dag.cli()

