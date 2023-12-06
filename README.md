# Airflow-DAGs-for-IMDB-reviews

Prerequisites -

1. Docker Desktop
2. Apache Airflow
3. AWS EMR and S3 Full Access
4. Code Editor (preferably VS code)

Setup -

Recommended: Refer the airflow docker documentation for detailed installation. - https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.html

Step 1: Check if you have adequate memory

```bash
docker run --rm "debian:bullseye-slim" bash -c 'numfmt --to iec $(echo $(($(getconf _PHYS_PAGES) * $(getconf PAGE_SIZE))))'
```

Step 2: Fetch Docker Compose

```bash
curl -LfO 'https://airflow.apache.org/docs/apache-airflow/2.7.3/docker-compose.yaml'
```

Step 3: Create Airflow Environment

```bash
mkdir -p ./dags ./logs ./plugins ./config
echo -e "AIRFLOW_UID=$(id -u)" > .env
```

Step 4: Initialize the airflow database

```bash
docker compose up airflow-init
```

Step 5: Run airflow

```bash
docker compose up
```

Step 6: Run the airflow scheduler

```bash
airflow scheduler
```

Step 7: Run the airflow webserver

```bash
airflow webserver
```

Now you can open up your browser and enter localhost:8080 to view the Airflow UI.
Your airflow username and password will be 'admin'.

Step 8: Several new folders will be created in your environment. Place all the DAG code in dags/ folder. Place the spark scripts in dags/spark_scripts folder. Place all the cluster config yaml files in configs/ folder.

Step 9: Declare all the airflow variables as needed. 

Step 10: Go to the connections tab and create two connections - aws_default and emr_default.

For aws_default, enter your AWS access key, AWS secret key and provide region_name as -

{
    'region_name':'us-east-2'
}

This should be added to the extra field.

Now you can trigger any DAG you wish to, but remember imdb_preprocessing DAG needs to be run for the data to be available for the imdb_model and imdb_eda DAGs.
