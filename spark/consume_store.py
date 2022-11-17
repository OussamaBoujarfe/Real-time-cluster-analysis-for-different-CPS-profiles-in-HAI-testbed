from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

from influx_writer import InfluxDBWriter
from influxdb_client import InfluxDBClient

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime

if __name__=="__main__":
    os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.2.0 pyspark-shell'

    # Spark Instancee
    spark = SparkSession.builder.master('local[*]').getOrCreate()
    spark.sparkContext.setLogLevel('ERROR')
 
    # Define an input stream
    cols = ['Time']

    fields = [StructField(col_name, StringType(), True) for col_name in cols]
    schema = StructType(fields)

    # Read stream from json and fit schema
    inputStream = spark\
        .readStream\
        .format("kafka")\
        .option("kafka.bootstrap.servers", "kafka:9092") \
        .option("subscribe", "HAI")\
        .load()

    inputStream = inputStream.select(col("value").cast("string").alias("data"))
    # inputStream.withWatermark("Time", "1 minute")
    inputStream.printSchema()

  
    # Read stream and process
    print(f"> Reading the stream and storing ...")
    query = (inputStream
        .writeStream
        .outputMode("append")
        .foreach(InfluxDBWriter( approaches = sys.argv[1:] ))
        .option("checkpointLocation", "checkpoints")
        .start())

    spark.streams.awaitAnyTermination()