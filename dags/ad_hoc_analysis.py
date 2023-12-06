from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import udf, col
from pyspark.sql.types import ArrayType, StringType, IntegerType, FloatType
from pyspark.sql.functions import concat_ws, collect_list, regexp_replace, explode, avg, count, length, corr
import re
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, CountVectorizer
from pyspark.ml import Pipeline
from pyspark.ml.clustering import LDA
import textstat
from textblob import TextBlob

if __name__ == '__main__':
    spark = SparkSession.builder \
        .appName("ad_hoc_analysis") \
        .config("spark.executor.memory", "10g") \
        .config("spark.driver.memory", "10g") \
        .config("spark.sql.files.maxPartitionBytes", "1g") \
        .config("spark.sql.shuffle.partitions", "200") \
        .getOrCreate()

    target_df = spark.read.parquet("s3://imdb-cs777/Reviews_Cleaned")

    # Top 20 most frequent Words

    tokenizer = Tokenizer(inputCol="cleaned_review", outputCol="words")
    wordsData = tokenizer.transform(target_df)

    remover = StopWordsRemover(inputCol="words", outputCol="filtered")
    wordsData = remover.transform(wordsData)

    df_exploded = wordsData.withColumn("word", explode(col("filtered")))
    word_counts = df_exploded.groupby("word").count().sort("count", ascending=False)

    word_counts.limit(20).coalesce(1).write.mode('overwrite').csv("s3://imdb-cs777/Top_20_Words")

    # Average Rating Per Movie

    average_ratings_per_movie = target_df.groupBy("movie").agg(avg("rating").alias("avg_rating"))
    average_ratings_per_movie = average_ratings_per_movie.orderBy("avg_rating", ascending=False)
    average_ratings_per_movie.coalesce(1).write.mode('overwrite').csv("s3://imdb-cs777/Average_Ratings_Per_Movie")

    # Readability Score

    def flesch_reading_ease(text):
        return textstat.flesch_reading_ease(text)

    flesch_udf = udf(flesch_reading_ease, FloatType())
    target_df = target_df.withColumn("flesch_score", flesch_udf("review_detail"))

    # Sentiment Score

    def sentiment_analysis(text):
        return TextBlob(text).sentiment.polarity

    sentiment_udf = udf(sentiment_analysis, FloatType())
    target_df = target_df.withColumn("sentiment_score", sentiment_udf("review_detail"))

    # Review Count per reviewer

    review_count_per_reviewer = target_df.groupBy("reviewer").agg(count("cleaned_review").alias("total_reviews"))
    review_count_per_reviewer.coalesce(1).write.mode('overwrite').csv("s3://imdb-cs777/Review_Count_Per_Reviewer")

    # Average Rating per reviewer

    average_rating_per_reviewer = target_df.groupBy("reviewer").agg(avg("rating").alias("avg_rating"))
    average_rating_per_reviewer.coalesce(1).write.mode('overwrite').csv("s3://imdb-cs777/Average_Rating_Per_Reviewer")

    # Average Sentiment per reviewer

    average_sentiment_per_reviewer = target_df.groupBy("reviewer").agg(avg("sentiment_score").alias("avg_sentiment"))
    average_sentiment_per_reviewer.coalesce(1).write.mode('overwrite').csv("s3://imdb-cs777/Average_Sentiment_Per_Reviewer")

    # Correlation Analysis

    target_df = target_df.withColumn("review_length", length("cleaned_review"))
    target_df = target_df.na.drop(subset=["review_length", "rating"])
    correlation = target_df.select(corr("review_length", "rating"))
    correlation.coalesce(1).write.mode('overwrite').csv("s3://imdb-cs777/Correlation")
    target_df.write.mode('overwrite').parquet("s3://imdb-cs777/Analysed_Reviews")

spark.stop()


