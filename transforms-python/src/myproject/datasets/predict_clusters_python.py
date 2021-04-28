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
from pyspark.sql import DataFrame
from io import StringIO
import requests
from joblib import Parallel, delayed


@configure(profile=['DRIVER_MEMORY_EXTRA_LARGE'])
@transform_df(
    Output("ri.foundry.main.dataset.4f6c8edd-55f6-4020-a48b-a9eb1b74d7d7"),
    deployments=Input("ri.foundry.main.dataset.f424168e-8fc8-4048-a35c-c8fb9f8be430"),
    predict_data=Input("ri.foundry.main.dataset.1a4fe048-f4ed-4757-8383-89a6fba427e6"),
    project_params=Input("ri.foundry.main.dataset.6d2b8ec2-33da-4d4b-8971-5971f3ed1089"),
    credentials=Input('ri.foundry.main.dataset.0c96802f-03dd-4ab6-a824-5fd1b8d0d32f')
)
def make_predictions_python(deployments, predict_data, project_params, credentials):
    credentials = credentials.toPandas()
    credentials = dict(zip(credentials.key, credentials.value))

    SERIES_ID = 'SKU'
    TARGET = 'Qty'

    ENDPOINT = 'https://app.datarobot.com/api/v2'
    FEATURE_DERIVATION_WINDOW = 59
    NUMBER_PREDICTION_STEPS = 28
    TOKEN = credentials['TOKEN']
    DATAROBOT_KEY = credentials['DATAROBOT_KEY']
    MAX_TRAINING_DATE = '2018-10-01'

    spark = SparkSession.builder.getOrCreate()

    def create_features(df, series_id, target):
        lags = [3, 7, 14, 28]
        wins = [3, 7, 14, 28]

        for lag in lags:
            print(df.shape)
            df["lag_{}".format(lag)] = df.groupby([series_id])[target].transform(lambda x: x.shift(lag))

        for win in wins:
            for lag in lags:
                print(win, lag, df.shape)
                df["rmean_{}_{}".format(lag, win)] = df.groupby([series_id])["lag_{}".format(lag)].transform(
                    lambda x: x.rolling(win).mean())

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

            headers = {'Content-Type': 'text/csv; charset=utf-8',
                       'datarobot-key': datarobot_key,
                       "Accept": "text/csv"}

            url = 'https://datarobot-cfds.dynamic.orm.datarobot.com/predApi/v1.0/deployments/{deployment_id}/' \
                  'predictions'.format(deployment_id=deployment_id)
            # Make API request for predictions
            predictions_response = requests.post(
                url,
                auth=(user_name, api_key),
                data=data.to_csv(),
                headers=headers,
                params={"passthroughColumns": passthrough_columns}
            )

            return pd.read_csv(StringIO(predictions_response.text))
            # return predictions_response]

        def calculate_rolling_forward(self,
                                      data,
                                      cluster,
                                    deployment_id,
                                    datarobot_key,
                                    username,
                                    api_key,
                                    starting_point,
                                    number_prediction_steps):

            forecast_dates = pd.date_range(start=starting_point,
                                        periods=number_prediction_steps + 1)

            forecast_data = data
            SERIES_ID = "SKU"
            for step, forecast_date in enumerate(forecast_dates[1:]):
                print('generating forecast for {}'.format(forecast_date.date().isoformat()))

                forecast_data = self.add_custom_features(forecast_data)

                forecast_point_data = forecast_data.loc[
                    forecast_data.Date == forecast_date.date().isoformat()
                    ]
                print(forecast_point_data.shape)
                point_forecast = self.make_predictions(forecast_point_data,
                                                    deployment_id,
                                                    datarobot_key,
                                                    username,
                                                    api_key,
                                                    passthrough_columns=[SERIES_ID, 'Date'])

                point_forecast = point_forecast.rename(columns={'{}_PREDICTION'.format(TARGET): '{}'.format(TARGET)})
                point_forecast[SERIES_ID] = point_forecast[SERIES_ID].astype(str)

                forecast_data.loc[
                    (forecast_data.Date == forecast_date.date().isoformat())
                    , TARGET] = point_forecast[TARGET].to_list()

            forecast_dates = [f.date().isoformat() for f in forecast_dates]
            forecast_data = forecast_data.loc[
                forecast_data.Date.isin(forecast_dates[1:])]
            forecast_data = forecast_data[['Date', 'SKU', 'Qty', 'actuals']]
            return forecast_data

    class RollingForwardOTVBaseCluster(RollingForwardOTVBase):
        """
        The base class, RollingForwardBase, contains all the methods needed to roll-forward
        an OTV model.
        Scores predictions using a supplied deployment_id
        """

        def add_custom_features(self, df):
            df = create_features(df, series_id=SERIES_ID, target=TARGET)
            return df

    def generate_cluster_predictions(cluster_details, data):

        # cluster_info = cluster_assignments.loc[cluster_assignments['cluster_id']==cluster_id]
        NUMBER_PREDICTION_STEPS = 28
        start_date = pd.to_datetime(MAX_TRAINING_DATE) - timedelta(FEATURE_DERIVATION_WINDOW)
        start_date = start_date.date().isoformat()
        end_date = pd.to_datetime(MAX_TRAINING_DATE) + timedelta(NUMBER_PREDICTION_STEPS)
        end_date = end_date.date().isoformat()

        rf = RollingForwardOTVBaseCluster()

        forecast_start_date = pd.to_datetime(MAX_TRAINING_DATE) - timedelta(1)
        forecast_start_date = forecast_start_date.date().isoformat()

        forecast_start_date = pd.to_datetime(MAX_TRAINING_DATE) - timedelta(1)
        forecast_start_date = forecast_start_date.date().isoformat()

        deployment_id = cluster_details['deployment_id']

        TOKEN = credentials['TOKEN']
        DATAROBOT_KEY = credentials['DATAROBOT_KEY']
        USERNAME = 'oleg.zarakhani@datarobot.com'
        NUMBER_PREDICTION_STEPS = 28

        predictions = rf.calculate_rolling_forward(data=data.toPandas(),
                                             cluster=cluster_details['cluster_id'],
                                             deployment_id=deployment_id,
                                             starting_point=forecast_start_date,
                                             datarobot_key=DATAROBOT_KEY,
                                             username=USERNAME,
                                             api_key=TOKEN,
                                             number_prediction_steps=NUMBER_PREDICTION_STEPS)

        predictions = predictions.rename(columns={'Qty': 'Qty_predicted', 'actuals': 'Qty_actuals'})
        predictions['cluster_id'] = int(cluster_details['cluster_id'])
        predictions['deployment_id'] = cluster_details['deployment_id']

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

    preds = pd.concat(pred_list)

    spark = SparkSession.builder.getOrCreate()
    return spark.createDataFrame(preds)
