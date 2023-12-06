from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import ArrayType, StringType, IntegerType, FloatType
import re
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, RegressionEvaluator
import sys

if __name__ == '__main__':

    model = sys.argv[1]

    spark = SparkSession.builder \
        .appName("Evaluation_IMDB") \
        .config("spark.executor.memory", "10g") \
        .config("spark.driver.memory", "10g") \
        .config("spark.sql.files.maxPartitionBytes", "1g") \
        .config("spark.sql.shuffle.partitions", "200") \
        .getOrCreate()

    predictions_train = spark.read.parquet(f's3://imdb-cs777/{model}_Predictions_Train')
    predictions_test = spark.read.parquet(f's3://imdb-cs777/{model}_Predictions_Test')

    if model in ['LinearRegression','RandomForestRegressor','GBTRegressor','DecisionTreeRegressor']:

        evaluator = RegressionEvaluator(labelCol="rating_category", predictionCol="prediction", metricName="rmse")
        rmse_train = evaluator.evaluate(predictions_train)
        rmse_test = evaluator.evaluate(predictions_test)
        
        rmse_train_df = spark.createDataFrame([(rmse_train,)], ['RMSE'])
        rmse_test_df = spark.createDataFrame([(rmse_test,)], ['RMSE'])

        rmse_train_df.write.mode('overwrite').parquet(f"s3://imdb-cs777/{model}_Training_RMSE")
        rmse_test_df.write.mode('overwrite').parquet(f"s3://imdb-cs777/{model}_Testing_RMSE")
    
    else:

        evaluator = MulticlassClassificationEvaluator(labelCol="rating_category", metricName="accuracy")

        accuracy_train = evaluator.evaluate(predictions_train)
        accuracy_test = evaluator.evaluate(predictions_test)

        accuracy_train_df = spark.createDataFrame([(accuracy_train,)], ['Model_accuracy'])
        accuracy_test_df = spark.createDataFrame([(accuracy_test,)], ['Model_accuracy'])

        accuracy_train_df.write.mode('overwrite').parquet(f"s3://imdb-cs777/{model}_Training_Accuracy")
        accuracy_test_df.write.mode('overwrite').parquet(f"s3://imdb-cs777/{model}_Testing_Accuracy")

spark.stop()
