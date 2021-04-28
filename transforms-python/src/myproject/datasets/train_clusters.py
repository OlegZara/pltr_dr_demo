from transforms.api import transform_df, Input, Output, configure
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from abc import ABC, abstractmethod

import datarobot as dr

from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql import functions as F
from py4j.java_gateway import java_import
from tqdm import tqdm
import multiprocessing
from joblib import Parallel, delayed


@configure(profile=['DRIVER_MEMORY_EXTRA_LARGE'])
@transform_df(
    Output("ri.foundry.main.dataset.f424168e-8fc8-4048-a35c-c8fb9f8be430"),
    modelling_data=Input('ri.foundry.main.dataset.3af76283-ed7e-403c-9bd4-df7e6d0b14a2'),
    credentials=Input('ri.foundry.main.dataset.0c96802f-03dd-4ab6-a824-5fd1b8d0d32f')
)
def train_cluster(modelling_data, credentials):

    credentials = credentials.toPandas()
    credentials = dict(zip(credentials.key, credentials.value))

    TOKEN = credentials['TOKEN']
    ENDPOINT = 'https://app.datarobot.com/api/v2'
    MAX_TRAINING_DATE = '2018-10-01'

    spark = SparkSession.builder.getOrCreate()

    def create_project(cluster, modelling_data):

        PROJECT_NAME = 'SKU_forecast_palantir'
        TARGET = 'Qty'

        MAX_WAIT = 3600

        dr.Client(token=TOKEN, endpoint=ENDPOINT)

        project_name = "{project_name} cluster: {cluster} Time: {time}".format(
            project_name=PROJECT_NAME,
            cluster=cluster,
            time=datetime.now().strftime("%y-%m-%d %H:%M:%S"))

        cluster_df = modelling_data.select(
            'RetailPrc', 'Item', 'Qty', 'OnAd', 'FrontPg', 'Display', 'onD5X', 'onBXG',
            'onMega', 'onMB', 'onDC', 'onFFD', 'Store', 'Cat', 'Date', 'cluster_id',
            'lag_3', 'lag_7', 'lag_14', 'lag_28', 'rmean_3_3', 'rmean_7_3', 'rmean_14_3', 
            'rmean_28_3', 'rmean_3_7', 'rmean_7_7', 'rmean_14_7', 'rmean_28_7', 'rmean_3_14', 
            'rmean_7_14', 'rmean_14_14', 'rmean_28_14', 'rmean_3_28', 'rmean_7_28', 
            'rmean_14_28', 'rmean_28_28')

        # cluster_df = cluster_df.filter(cluster_df.cluster_id == int(cluster))

        proj = dr.Project.create(cluster_df.toPandas(),
                                 project_name=project_name,
                                 max_wait=MAX_WAIT,
                                 read_timeout=MAX_WAIT)

        time_partition = dr.DatetimePartitioningSpecification(
            datetime_partition_column='Date',
            disable_holdout=True,
            use_time_series=False,
            backtests=2
        )

        proj.set_target(
            target=TARGET,
            mode=dr.AUTOPILOT_MODE.QUICK,
            partitioning_method=time_partition,
            max_wait=3600,
            worker_count=-1
        )
        return proj

    num_cores = multiprocessing.cpu_count()
    cluster_ids = modelling_data.select("cluster_id").distinct().toPandas().cluster_id.unique()

    inputs_raw = [(str(cluster), modelling_data.filter(modelling_data.cluster_id == int(cluster))) \
                  for cluster in cluster_ids]

    inputs = tqdm(inputs_raw)
    # print("done with data")
    projects = Parallel(n_jobs=num_cores, backend="threading")(delayed(create_project)(i, d) for i, d in inputs)

    # projects = list()
    # for i, d in inputs:
    #     project = create_project(i, d)
    #     projects.append(project)
    #     # break

    for p in projects:
        p.wait_for_autopilot()

    all_project_info = []
    for project in projects:
        info = {}
        info["segmentation_field"] = "cluster_id"
        info["segmentation_value"] = project.project_name.split(" ")[2]
        info["project_name"] = project.project_name
        info["project"] = project.id
        all_project_info.append(info)
    ##
    ## make sure this is the correct server  'https://cfds-ccm-prod.orm.datarobot.com'

    default_prediction_server_id = [dps for dps in dr.PredictionServer.list()
                                    if dps.url == 'https://cfds-ccm-prod.orm.datarobot.com'][0]
    deployments = {}
    for project_info in all_project_info:
        project_id = project_info["project"]
        project_name = project_info["project_name"]
        rec = dr.ModelRecommendation.get(project_id,
                                        recommendation_type=dr.enums.RECOMMENDED_MODEL_TYPE.RECOMMENDED_FOR_DEPLOYMENT)

        segmentation_field = project_info["segmentation_field"]
        segmentation_value = project_info["segmentation_value"]
        print(" {}:{}".format(segmentation_field, segmentation_value))
        print(rec)
        ##########################################

        # model = dr.Model.get(project.id, rec.model_id)
        model = rec.get_model()
        dep = dr.Deployment.create_from_learning_model(
            model.id,
            label=project_name.format(segmentation_field, segmentation_value),
            description=project_name.format(segmentation_field, segmentation_value, rec.recommendation_type),
            default_prediction_server_id=default_prediction_server_id.id
        )

        deployments[project_name] = dep.model
        deployments[project_name]["deployment_id"] = dep.id
        deployments[project_name]["url"] = model.get_leaderboard_ui_permalink()
        # deployments[project_name]["segmentation_field"] = segmentation_field
        deployments[project_name]["cluster_id"] = segmentation_value
        print(project_name.split(" ")[2])

    spark = SparkSession.builder.getOrCreate()
    return spark.createDataFrame(pd.DataFrame(deployments).transpose())
