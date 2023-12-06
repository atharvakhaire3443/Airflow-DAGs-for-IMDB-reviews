from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import get_json_object
from pyspark.sql.functions import col


if __name__ == '__main__':

    spark = SparkSession.builder \
    .appName("Transform_IMDB") \
    .config("spark.executor.memory", "10g") \
    .config("spark.driver.memory", "10g") \
    .config("spark.sql.files.maxPartitionBytes", "1g") \
    .config("spark.sql.shuffle.partitions", "200") \
    .getOrCreate()

    raw_df = spark.read.parquet("s3://imdb-cs777/raw_imdb_reviews/raw_imdb_reviews_FULL.parquet")

    target_df = raw_df.select(
    col('review_json.review_id').alias('review_id'),
    col('review_json.reviewer').alias('reviewer'),
    col('review_json.movie').alias('movie'),
    col('review_json.rating').alias('rating'),
    col('review_json.review_summary').alias('review_summary'),
    col('review_json.review_date').alias('review_date'),
    col('review_json.spoiler_tag').alias('spoiler_tag'),
    col('review_json.review_detail').alias('review_detail'),
    col('review_json.helpful').alias('helpful')
)
    
    target_df.write.mode('overwrite').parquet("s3://imdb-cs777/Transformed_Reviews")

    spark.stop()
