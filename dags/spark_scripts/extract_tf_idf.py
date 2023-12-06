from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, length
from pyspark.sql.types import ArrayType, StringType, IntegerType, FloatType
import re
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline

if __name__ == '__main__':

    spark = SparkSession.builder \
        .appName("Extract_TFIDF_IMDB") \
        .config("spark.executor.memory", "10g") \
        .config("spark.driver.memory", "10g") \
        .config("spark.sql.files.maxPartitionBytes", "1g") \
        .config("spark.sql.shuffle.partitions", "200") \
        .getOrCreate()

    target_df = spark.read.parquet("s3://imdb-cs777/Reviews_Cleaned")

    tokenizer = Tokenizer(inputCol="cleaned_review", outputCol="words")

    remover = StopWordsRemover(inputCol="words", outputCol="filtered")

    hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures")

    idf = IDF(inputCol="rawFeatures", outputCol="features")

    pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf])

    model = pipeline.fit(target_df)

    featured_data = model.transform(target_df)

    featured_data = featured_data.select('review_id','features','rating_category')

    featured_data.write.mode('overwrite').parquet("s3://imdb-cs777/Features")

spark.stop()
