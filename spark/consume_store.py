from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

#from influx_writer import InfluxDBWriter
#from influxdb_client import InfluxDBClient
#influxdb-client

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
    cols = ["time", "P1_B2004", "P1_B2016", "P1_B3004", "P1_B3005", 
        "P1_B4002", "P1_B4005", "P1_B400B", "P1_B4022", "P1_FCV01D",
        "P1_FCV01Z", "P1_FCV02D", "P1_FCV02Z", "P1_FCV03D", "P1_FCV03Z",
        "P1_FT01", "P1_FT01Z", "P1_FT02", "P1_FT02Z", "P1_FT03", 
        "P1_FT03Z", "P1_LCV01D", "P1_LCV01Z", "P1_LIT01", "P1_PCV01D", 
        "P1_PCV01Z", "P1_PCV02D", "P1_PCV02Z", "P1_PIT01", "P1_PIT02", 
        "P1_TIT01", "P1_TIT02", "P2_24Vdc", "P2_Auto", "P2_Emgy", 
        "P2_On", "P2_SD01", "P2_SIT01", "P2_TripEx", "P2_VT01e", 
        "P2_VXT02", "P2_VXT03", "P2_VYT02", "P2_VYT03", "P3_LCP01D", 
        "P3_LCV01D", "P3_LH", "P3_LL", "P3_LT01", "P4_HT_FD", 
        "P4_HT_LD", "P4_HT_PO", "P4_HT_PS", "P4_LD", "P4_ST_FD", 
        "P4_ST_LD", "P4_ST_PO", "P4_ST_PS", "P4_ST_PT01", "P4_ST_TT01", 
        "attack", "attack_P1", "attack_P2", "attack_P3"]

    fields = [StructField('time', TimestampType(), False)]
    for col_name in cols[1:-4]:
        fields.append(StructField(col_name, DecimalType(10,5), False))
    for col_name in cols[-4:]:
        fields.append(StructField(col_name, StringType(), False))
        
    schema = StructType(fields)
    

    # Read stream from json and fit schema
    inputStream = spark\
        .readStream\
        .format("kafka")\
        .option("kafka.bootstrap.servers", "kafka:9092")\
        .option("subscribe", "HAI")\
        .load()
    
    
    inputStream.printSchema()
    
    inputStream = inputStream.select(col("value").cast("string").alias("data"))
                                     
    inputStream = inputStream.select(from_json(col("data"), schema,{"mode" : "PERMISSIVE"}).alias("data")).select("data.*")
    inputStream = inputStream.withColumn("attack", inputStream["attack"].cast(BooleanType()))
    inputStream = inputStream.withColumn("attack_P1", inputStream["attack_P1"].cast(BooleanType()))
    inputStream = inputStream.withColumn("attack_P2", inputStream["attack_P1"].cast(BooleanType()))
    inputStream = inputStream.withColumn("attack_P3", inputStream["attack_P1"].cast(BooleanType()))
    
    
   
    # inputStream.withWatermark("Time", "1 minute")
    inputStream.printSchema()

        
    # Read stream and process
    print(f"> Reading the stream and storing ...")
    query = inputStream \
        .writeStream \
        .format("console") \
        .start()
        #.option("truncate", "false")\
        

    spark.streams.awaitAnyTermination()
