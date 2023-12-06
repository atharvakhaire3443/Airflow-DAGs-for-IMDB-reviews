from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, length, regexp_replace
from pyspark.sql.types import ArrayType, StringType, IntegerType, FloatType
import re

if __name__ == '__main__':

    spark = SparkSession.builder \
        .appName("Preprocessing_IMDB") \
        .config("spark.executor.memory", "10g") \
        .config("spark.driver.memory", "10g") \
        .config("spark.sql.files.maxPartitionBytes", "1g") \
        .config("spark.sql.shuffle.partitions", "200") \
        .getOrCreate()

    def categorize_rating(rating):
        if rating >= 6.66:
            return 3
        elif rating >= 3.33:
            return 2
        elif rating >= 0:
            return 1
        elif rating == None:
            return None
        else:
            return None

    categorize_rating_udf = udf(categorize_rating, StringType())

    target_df = spark.read.parquet("s3://imdb-cs777/Transformed_Reviews")

    target_df = target_df.filter(target_df.rating.isNotNull())

    target_df = target_df.withColumn('rating',col('rating').cast(FloatType()))

    target_df = target_df.withColumn("rating_category", categorize_rating_udf("rating"))

    target_df = target_df.withColumn("review_detail", regexp_replace("review_detail", "[^A-Za-z0-9 ]", ""))

    def preprocess_text(text):

        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text

    preprocess_udf = udf(preprocess_text, StringType())

    target_df = target_df.withColumn("cleaned_review", preprocess_udf(target_df["review_detail"]))

    target_df = target_df.filter(col("rating_category").isNotNull() & (length(col("cleaned_review")) >= 1000)) \
                        .withColumn("rating_category", col("rating_category").cast(FloatType()))

    target_df.write.mode('overwrite').parquet("s3://imdb-cs777/Reviews_Cleaned")

spark.stop()
