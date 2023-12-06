from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import udf, col
from pyspark.sql.types import ArrayType, StringType, IntegerType, FloatType
from pyspark.sql.functions import concat_ws, collect_list
import re
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, CountVectorizer
from pyspark.ml import Pipeline
from pyspark.ml.clustering import LDA


if __name__ == '__main__':

    spark = SparkSession.builder \
        .appName("LDA") \
        .config("spark.executor.memory", "10g") \
        .config("spark.driver.memory", "10g") \
        .config("spark.sql.files.maxPartitionBytes", "1g") \
        .config("spark.sql.shuffle.partitions", "200") \
        .getOrCreate()
    
    target_df = spark.read.parquet("s3://imdb-cs777/Reviews_Cleaned")

    tokenizer = Tokenizer(inputCol="cleaned_review", outputCol="words")
    wordsData = tokenizer.transform(target_df)

    remover = StopWordsRemover(inputCol="words", outputCol="filtered")
    wordsData = remover.transform(wordsData)

    cv = CountVectorizer(inputCol="filtered", outputCol="rawFeatures")
    cvModel = cv.fit(wordsData)
    featurizedData = cvModel.transform(wordsData)

    idf = IDF(inputCol="rawFeatures", outputCol="features")
    idfModel = idf.fit(featurizedData)
    rescaledData = idfModel.transform(featurizedData)

    rescaledData = rescaledData.select("features")

    num_topics = 10
    lda = LDA(k=num_topics, maxIter=10)
    ldaModel = lda.fit(rescaledData)

    topics = ldaModel.describeTopics(maxTermsPerTopic=10)

    topics_collected = topics.collect()

    vocabulary = cvModel.vocabulary

    topics_data = []
    for topic in topics_collected:
        topic_str = f"Topic {topic.topic}: " + " ".join(
            f"{vocabulary[term_index]}({weight})" for term_index, weight in zip(topic.termIndices, topic.termWeights)
        )
        topics_data.append(Row(topic_str))

    df = spark.createDataFrame(topics_data, ["TopicDetails"])

    ldaModel.describeTopics().coalesce(1).write.mode('overwrite').parquet("s3://imdb-cs777/IMDB_LDA")
    df.coalesce(1).write.mode('overwrite').option("header", "false").text("s3://imdb-cs777/IMDB_10_Topics_top_words")

spark.stop()