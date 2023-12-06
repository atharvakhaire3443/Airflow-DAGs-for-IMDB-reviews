import boto3
import time
import logging
import yaml

def create(emr, config_file):

    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    
    applications = config['cluster']['applications'].split(',')

    configured_applications = []
    for item in applications:
        configured_applications.append({'Name': item})

    cluster_config = emr.run_job_flow(
    Name=config['cluster']['name'],
    ReleaseLabel=config['cluster']['emr-version'], 
    LogUri = config['cluster']['log-uri'],
    Instances={
        'InstanceGroups': [
            {
                'Name': config['cluster']['instances']['instance-groups']['master']['name'],
                'Market': config['cluster']['instances']['instance-groups']['master']['market'], 
                'InstanceRole': config['cluster']['instances']['instance-groups']['master']['instance-role'],
                'InstanceType': config['cluster']['instances']['instance-groups']['master']['instance-type'],
                'InstanceCount': config['cluster']['instances']['instance-groups']['master']['instance-count'],
            },
            {
                'Name': config['cluster']['instances']['instance-groups']['core']['name'],
                'Market': config['cluster']['instances']['instance-groups']['core']['market'], 
                'InstanceRole': config['cluster']['instances']['instance-groups']['core']['instance-role'],
                'InstanceType': config['cluster']['instances']['instance-groups']['core']['instance-type'],
                'InstanceCount': config['cluster']['instances']['instance-groups']['core']['instance-count'], 
            },
            {
                'Name': config['cluster']['instances']['instance-groups']['task']['name'],
                'Market': config['cluster']['instances']['instance-groups']['task']['market'], 
                'InstanceRole': config['cluster']['instances']['instance-groups']['task']['instance-role'],
                'InstanceType': config['cluster']['instances']['instance-groups']['task']['instance-type'],
                'InstanceCount': config['cluster']['instances']['instance-groups']['task']['instance-count'], 
            }
        ],
        'Ec2SubnetId': config['cluster']['instances']['ec2-subnet-id'],
        'Ec2KeyName': config['cluster']['instances']['ec2-key-name'],
        'KeepJobFlowAliveWhenNoSteps': config['cluster']['instances']['keep-job-flow-alive-when-no-steps'],
        'TerminationProtected': config['cluster']['instances']['termination-protected'],
    },
    BootstrapActions=[
        {
            'Name': config['cluster']['bootstrap-actions']['name'],
            'ScriptBootstrapAction': {
                'Path': config['cluster']['bootstrap-actions']['path'],
            }
        },
    ],
    Applications=configured_applications,
    JobFlowRole=config['cluster']['job-flow-role'],
    ServiceRole=config['cluster']['service-role'],
    )

    cluster_id = cluster_config['JobFlowId']
    return cluster_id

def wait_launch(id,emr):

    while True:
        response = emr.describe_cluster(ClusterId=id)
        status = response['Cluster']['Status']['State']

        logging.info(f"Cluster status: {status}")

        if status in ['WAITING', 'RUNNING']:
            return 200
        elif status in ['TERMINATING', 'TERMINATED', 'TERMINATED_WITH_ERRORS']:
            return 410
        else:
            time.sleep(15)

def step(id,emr,step_name,s3_path, arg):

    step_config = {
        'Name': step_name,
        'ActionOnFailure': 'TERMINATE_CLUSTER',
        'HadoopJarStep': {
        'Jar': 'command-runner.jar',
        'Args': ['spark-submit','--deploy-mode', 'cluster', s3_path, arg]
        }
    }

    response = emr.add_job_flow_steps(JobFlowId=id, Steps=[step_config])
    step_id = response['StepIds'][0]
    logging.info(f"Added step: {step_id}")

    while True:
        step_status = emr.describe_step(ClusterId=id, StepId=step_id)
        state = step_status['Step']['Status']['State']

        logging.info(f"Step status: {state}")

        if state in ['COMPLETED']:
            return 200
        elif state in ['CANCELLED', 'FAILED', 'INTERRUPTED']:
            return 410
        else:
            time.sleep(15)

def terminate(id,emr):

    response = emr.terminate_job_flows(JobFlowIds=[id])
    print(f"Cluster {id} is being terminated.")

    return response