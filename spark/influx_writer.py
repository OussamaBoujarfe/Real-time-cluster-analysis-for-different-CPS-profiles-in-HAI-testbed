 
from pyspark.sql import Row

import pandas as pd

from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS


URL = "http://influxdb:8086"
INFLUX_TOKEN = "fu7xGKqYTlxb3AFRJT48z_-6CYTPB03aTdiR4pw-sPO0aV3F6INzrlIRA0xVEz4Pxm0p96aS8D9kM7udMUkI9g=="
ORG = "primary"
BUCKET = "hai"
#INFLUX_TOKEN =  'Vp7vp5ddGYs9vh4A4NVM6N7MmJvNGN79tODBCCPTxx_DAe7Xty8Vmi7kh-unIZ4QKbd3o7r2bUNTcj0OaSeyWg=='
#_vueR6p82n7khVlskwa91RB_mhU5Ftte71K0M0-Ifm6gZjheEukfAIbHJXKl_brh5YvP3Sx1sjtCva6Oq2Z7dA==

class InfluxDBWriter:
    def __init__(self):
        self.url = URL
        self.token = INFLUX_TOKEN
        self.org = ORG
        self.bucket = BUCKET
        self.client = InfluxDBClient(url=self.url, token=self.token, org=self.org)
        self.write_api = self.client.write_api()
        self.delete_api = self.client.delete_api()


    def saveToInfluxDB(self, df):
        try:
            self.write_api.write(bucket=self.bucket, org=self.org, record=df,
             data_frame_timestamp_column="time", data_frame_measurement_name="attack" )
            print(f"Writen {len(df)} data points to influxdb")            
        
        except Exception as ex:
            print(f"[x] Error {str(ex)}")

    def flushInfluxDB(self):
        start = "1970-01-01T00:00:00Z"
        stop = "2022-12-05T08:00:00Z"
        self.delete_api.delete(start, stop, '_measurement="attack"', bucket=self.bucket, org=self.org)
        print("Old data flushed")
