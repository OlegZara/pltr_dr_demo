import datarobot as dr
import pandas as pd
import numpy as np
from transforms.api import transform_df, Input, Output
from pyspark.sql import SparkSession

@transform_df(
    Output("ri.foundry.main.dataset.ac0d62bf-6109-4ff1-8dfc-40c871aec121"),
    predict_data=Input("ri.foundry.main.dataset.1a4fe048-f4ed-4757-8383-89a6fba427e6"),
    deployments=Input("ri.foundry.main.dataset.f424168e-8fc8-4048-a35c-c8fb9f8be430"),
)
def my_compute_function(predict_data, deployments):


    def create_features(df, series_id, target):

        """
        :param df: times series standard DataFrame, date feature, series id and target as columns
        :param series_id: string
        :param target: string
        :return:
        """
        lags = np.array([3, 7, 14, 28])
        wins = np.array([3, 7, 14, 28])

        for lag in lags:
            # print(df.shape)
            df["lag_{}".format(lag)] = df.groupby([series_id])[target].transform(lambda x: x.shift(lag))

        for win in wins:
            for lag in lags:
                # print(win, lag, df.shape)
                df["rmean_{}_{}".format(lag, win)] = df.groupby([series_id])["lag_{}".format(lag)].transform(
                    lambda x: x.rolling(win).mean())

        # print(df.shape)

        return df

    # inputData = predict_data
    TOKEN = 'NWYzYmYxMjUyYzYxOWMwZTAxZDJiZjQxOlRGeGdnTm5LY1hQUTRYMGdxQjdwcmo4WXVvb0NOMGZv'
    ENDPOINT = 'https://app.datarobot.com/api/v2'

    deployments = deployments.toPandas()
    deployments = deployments.to_dict('records')

    dep = deployments[0]

    client = dr.Client(token=TOKEN, 
                    endpoint=ENDPOINT)

    project_id = dep['project_id']
    model_id = dep['id']
    deployment_id = dep['deployment_id']

    cluster_id = int(dep['segmentation_value'])

    cluster_df = predict_data.filter(predict_data.cluster_id == int(cluster_id))
    cluster_df = cluster_df.toPandas()

    cluster_df = create_features(cluster_df, 'Item', 'Qty')


    # deployment_id = "600743784b7f647c9244a66b"

    job = dr.BatchPredictionJob.score_to_file(
        deployment_id,
        cluster_df,
        'temp.csv',
        passthrough_columns=['Date', 'Item']
    )

    print("started scoring...", job)
    job.wait_for_completion()
    out = pd.read_csv('temp.csv')
    spark = SparkSession.builder.getOrCreate()
    return spark.createDataFrame(out)
