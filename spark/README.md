# Spark Streaming
This directory contains the stream processing part of the project.
Data stread is read from a kafka topic and then split into batches. Each batch is processed and stored in InfluxDB.
We are using PySpark for the streaming and InfluxDB as a time series database

