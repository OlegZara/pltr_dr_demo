from transforms.api import transform_df, Input, Output, configure
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

import datarobot as dr

from pyspark.sql import SparkSession
from datetime import datetime, timedelta
from pyspark.sql.window import Window
from pyspark.sql import functions as F
from py4j.java_gateway import java_import
from tqdm import tqdm
from functools import reduce
from joblib import Parallel, delayed


@configure(profile=['DRIVER_MEMORY_EXTRA_LARGE'])
@transform_df(
    Output("ri.foundry.main.dataset.7d468872-685f-4d7c-b632-e924662cb91d"),
    deployments=Input("ri.foundry.main.dataset.f424168e-8fc8-4048-a35c-c8fb9f8be430"),
    predict_data=Input("ri.foundry.main.dataset.1a4fe048-f4ed-4757-8383-89a6fba427e6"),
    project_params=Input("ri.foundry.main.dataset.6d2b8ec2-33da-4d4b-8971-5971f3ed1089"),
    credentials=Input('ri.foundry.main.dataset.0c96802f-03dd-4ab6-a824-5fd1b8d0d32f')
)
def make_predictions(deployments, predict_data, project_params, credentials):

    credentials = credentials.toPandas()
    credentials = dict(zip(credentials.key, credentials.value))

    SERIES_ID = 'Item'
    TARGET = 'Qty'
    TOKEN = credentials['TOKEN']
    ENDPOINT = 'https://app.datarobot.com/api/v2'
    FEATURE_DERIVATION_WINDOW = 59
    NUMBER_PREDICTION_STEPS = 28

    MAX_TRAINING_DATE = '2018-10-01'

    spark = SparkSession.builder.getOrCreate()

    def create_features_spark(df, series_id, target):
        lags = [3, 7, 14, 28]
        wins = [3, 7, 14, 28]

        windowSpec = Window().partitionBy([series_id]).orderBy('Date')

        for lag in lags:
            df = df.withColumn("lag_{}".format(lag), 
                               F.lag(target, lag).over(windowSpec))

        for win in wins:
            for lag in lags:
                windowSpecRoll = windowSpec.rowsBetween(-(win-1), 0)
                df = df.withColumn("rmean_{}_{}".format(lag, win),
                                   F.mean("lag_{}".format(lag))
                                   .over(windowSpecRoll))

        return df

    class RollingForwardOTVBase(ABC):

        @abstractmethod
        def add_custom_features(self, df):
            return df

        @staticmethod
        def make_predictions(data,
                             deployment_id,
                             datarobot_key,
                             user_name,
                             api_key,
                             passthrough_columns=None):

            dr.Client(token=TOKEN,
                      endpoint=ENDPOINT)

            job = dr.BatchPredictionJob.score_to_file(
                deployment_id,
                data,
                'temp.csv',
                passthrough_columns=passthrough_columns,
                download_read_timeout=3600,
                download_timeout=3600
            )
            job.wait_for_completion()
            out = pd.read_csv('temp.csv')
            return out

        def calculate_rolling_forward(self,
                                      data,
                                      dep,
                                      starting_point,
                                      number_prediction_steps):

            forecast_dates = pd.date_range(start=starting_point,
                                           periods=number_prediction_steps + 1)

            java_import(spark._jvm, 'com.datarobot.prediction.Predictors')
            java_import(spark._jvm, 'com.datarobot.prediction.spark.Model')
            java_import(spark._jvm, 'com.datarobot.prediction.spark.Predictors')

            project_id = dep['project_id']
            model_id = dep['id']

            codeGenModel = spark._jvm.com.datarobot.prediction.spark.Predictors\
                .getPredictorFromServer("https://app.datarobot.com/projects/{}/models/{}/blueprint"\
                                        .format(project_id, model_id), TOKEN)

            forecast_data = data.cache()

            for step, forecast_date in enumerate(forecast_dates[1:]):

                forecast_data = self.add_custom_features(forecast_data)

                forecast_point_data = forecast_data.filter(F.col('Date') == forecast_date.date().isoformat())

                java_dataframe_object = codeGenModel.transform(forecast_point_data._jdf)

                from pyspark.sql import DataFrame
                point_forecast = DataFrame(java_dataframe_object, spark)

                point_forecast = point_forecast.select('Date', 'SKU', 'PREDICTION')

                forecast_data = forecast_data.join(point_forecast, on=['SKU', 'Date'], how='left')

                forecast_data = forecast_data.withColumn('Qty',
                                                         F.when(forecast_data['Date'] == forecast_date
                                                                .date().isoformat(),
                                                                F.col('PREDICTION'))
                                                         .otherwise(F.col('Qty')))  # .collect()
                # forecast_data = spark.createDataFrame(forecast_data)

                forecast_data = forecast_data.drop('PREDICTION')

                # if step == 4:
                #     break

            forecast_dates = [f.date().isoformat() for f in forecast_dates[1:]]
            forecast_data = forecast_data.filter(F.col('Date').isin(forecast_dates))

            forecast_data = forecast_data.withColumn('deployment_id', F.lit(dep['deployment_id']))
            return forecast_data

    class RollingForwardOTVBaseCluster(RollingForwardOTVBase):
        """
        The base class, RollingForwardBase, contains all the methods needed to roll-forward
        an OTV model.
        Scores predictions using a supplied deployment_id
        """

        def add_custom_features(self, df):
            df = create_features_spark(df, series_id=SERIES_ID, target=TARGET)
            return df

    def generate_cluster_predictions(cluster_details, data):

        # cluster_info = cluster_assignments.loc[cluster_assignments['cluster_id']==cluster_id]

        start_date = pd.to_datetime(MAX_TRAINING_DATE) - timedelta(FEATURE_DERIVATION_WINDOW)
        start_date = start_date.date().isoformat()
        end_date = pd.to_datetime(MAX_TRAINING_DATE) + timedelta(NUMBER_PREDICTION_STEPS)
        end_date = end_date.date().isoformat()

        rf = RollingForwardOTVBaseCluster()

        forecast_start_date = pd.to_datetime(MAX_TRAINING_DATE) - timedelta(1)
        forecast_start_date = forecast_start_date.date().isoformat()
        # return df

        predictions = rf.calculate_rolling_forward(data=data,
                                                   dep=cluster_details,
                                                   starting_point=forecast_start_date,
                                                   number_prediction_steps=7)
        # predictions = spark.createDataFrame(predictions)
        predictions = predictions.withColumn('cluster_id', F.lit(int(cluster_details['cluster_id'])))

        return predictions

    deployments = deployments.toPandas()
    deployments = deployments.to_dict('records')
    # num_cores = multiprocessing.cpu_count()

    inputs_raw = [(dep, predict_data.filter(predict_data.cluster_id == int(dep['cluster_id'])))\
                  for dep in deployments]

    inputs = tqdm(inputs_raw)
    num_cores = 10
    pred_list = Parallel(n_jobs=num_cores,
                         backend="threading")(delayed(generate_cluster_predictions)(i, d) for i, d in inputs)


    for i, d in inputs:
        generate_cluster_predictions(i, d)

    from pyspark.sql import DataFrame
    preds = reduce(DataFrame.unionAll, pred_list)
    preds = preds.orderBy('SKU', 'Date')
    # preds = preds.select('SKU', 'Date', 'Qty', 'actuals')

    preds = preds.select('SKU',
                         'Date',
                         F.col("Qty").alias("Qty_predicted"),
                         F.col("actuals").alias("Qty_actuals"),
                         'cluster_id',
                         'deployment_id')

    # preds = preds.where(F.col('Qty') != F.col('actuals'))

    return preds
