from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import ArrayType, StringType, IntegerType, FloatType
import re
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

if __name__ == '__main__':

    spark = SparkSession.builder \
        .appName("Semantic_Analysis_IMDB") \
        .config("spark.executor.memory", "10g") \
        .config("spark.driver.memory", "10g") \
        .config("spark.sql.files.maxPartitionBytes", "1g") \
        .config("spark.sql.shuffle.partitions", "200") \
        .getOrCreate()
    
    featured_data = spark.read.parquet("s3://imdb-cs777/Features")

    (train_data, test_data) = featured_data.randomSplit([0.8, 0.2])

    rf = RandomForestRegressor(featuresCol="features", labelCol="rating_category")

    rf_model = rf.fit(train_data)

    predictions_train = rf_model.transform(train_data)

    predictions_test = rf_model.transform(test_data)

    predictions_train.write.mode('overwrite').parquet("s3://imdb-cs777/RandomForestRegressor_Predictions_Train")
    predictions_test.write.mode('overwrite').parquet("s3://imdb-cs777/RandomForestRegressor_Predictions_Test")

spark.stop()
