from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import udf, col, length
from pyspark.sql.types import ArrayType, StringType, IntegerType, FloatType
from pyspark.sql.functions import concat_ws, collect_list
import re
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, CountVectorizer, VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
from pyspark.ml.clustering import LDA, KMeans

if __name__ == '__main__':
    spark = SparkSession.builder \
        .appName("kmeans") \
        .config("spark.executor.memory", "10g") \
        .config("spark.driver.memory", "10g") \
        .config("spark.sql.files.maxPartitionBytes", "1g") \
        .config("spark.sql.shuffle.partitions", "200") \
        .getOrCreate()

    target_df = spark.read.parquet("s3://imdb-cs777/Reviews_Cleaned")

    target_df = target_df.withColumn("review_length", length("cleaned_review"))

    assembler = VectorAssembler(inputCols=["review_length", "rating"], outputCol="features")
    target_df = assembler.transform(target_df)

    scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=False)
    scalerModel = scaler.fit(target_df)
    target_df = scalerModel.transform(target_df)

    k = 5

    kmeans = KMeans(featuresCol="scaledFeatures", k=k)
    model = kmeans.fit(target_df)

    predictions = model.transform(target_df)

    centers = model.clusterCenters()
    centers_data = [Row(cluster_center=[float(value) for value in center]) for center in centers]
    df = spark.createDataFrame(centers_data)

    df.coalesce(1).write.mode("overwrite").option("header", "true").parquet('s3://imdb-cs777/Cluster_Centers')
    
    cluster_counts = predictions.groupBy("prediction").count()
    cluster_counts.coalesce(1).write.mode('overwrite').csv("s3://imdb-cs777/Cluster_Counts")

spark.stop()


    