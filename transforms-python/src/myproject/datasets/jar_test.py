from transforms.api import transform_df, Input, Output
from py4j.java_gateway import java_import
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql import functions as F


@transform_df(
    Output("ri.foundry.main.dataset.c3547fd9-548d-40af-a2c7-1110ae8b7aa0"),
    predict_data=Input("ri.foundry.main.dataset.1a4fe048-f4ed-4757-8383-89a6fba427e6"),
    deployments=Input("ri.foundry.main.dataset.f424168e-8fc8-4048-a35c-c8fb9f8be430"),
)
def my_compute_function(predict_data, deployments):
    # inputData = predict_data

    def create_features(df, series_id, target):
        lags = [3,7,14,28]
        wins = [3,7,14,28]
        
        windowSpec = Window().partitionBy([series_id]).orderBy('Date')
        
        for lag in lags:
            df = df.withColumn("lag_{}".format(lag),F.lag(target, lag).over(windowSpec))

        for win in wins:
            for lag in lags:
                windowSpecRoll = windowSpec.rowsBetween(-(win-1),0)
                df = df.withColumn("rmean_{}_{}".format(lag, win),F.mean("lag_{}".format(lag)).over(windowSpecRoll))

        return df


    deployments = deployments.toPandas()
    deployments = deployments.to_dict('records')

    dep = deployments[0]

    project_id = dep['project_id']
    model_id = dep['id']
    cluster_id = int(dep['segmentation_value'])

    cluster_df = predict_data.filter(predict_data.cluster_id == int(cluster_id))
    cluster_df = create_features(cluster_df, 'SKU', 'Qty')

    spark = SparkSession.builder.getOrCreate()

    java_import(spark._jvm, 'com.datarobot.prediction.Predictors')
    java_import(spark._jvm, 'com.datarobot.prediction.spark.Model')
    java_import(spark._jvm, 'com.datarobot.prediction.spark.Predictors')
    codeGenModel = spark._jvm.com.datarobot.prediction.spark.Predictors\
        .getPredictorFromServer("https://app.datarobot.com/projects/{}/models/{}/blueprint".format(project_id, model_id)
                                ,"NWYzYmYxMjUyYzYxOWMwZTAxZDJiZjQxOlRGeGdnTm5LY1hQUTRYMGdxQjdwcmo4WXVvb0NOMGZv")
    java_dataframe_object = codeGenModel.transform(cluster_df._jdf)
    # Then I will get the initial dataframe with additional columns with the prediction results
    from pyspark.sql import DataFrame
    output = DataFrame(java_dataframe_object, spark)
    print("After prediction")
    return output
    # output.show()
