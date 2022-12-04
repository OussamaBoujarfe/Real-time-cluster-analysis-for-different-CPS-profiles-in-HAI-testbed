 
from pyspark.sql import Row

import pandas as pd

from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS


URL = "http://influxdb:8086"
INFLUX_TOKEN = "JXe0HqUucKEzH1sixt-SZBesF3wayT-wGWq6RyLcKPcoaVUBfTdWRT_rx9Ul6f4uIq1qjlDQrYDp64BZ5LqQBw=="
ORG = "primary"
BUCKET = "hai"
#INFLUX_TOKEN =  'Vp7vp5ddGYs9vh4A4NVM6N7MmJvNGN79tODBCCPTxx_DAe7Xty8Vmi7kh-unIZ4QKbd3o7r2bUNTcj0OaSeyWg=='


class InfluxDBWriter:
    def __init__(self):
        self.url = URL
        self.token = INFLUX_TOKEN
        self.org = ORG
        self.bucket = BUCKET
        self.client = InfluxDBClient(url=self.url, token=self.token, org=self.org)
        self.write_api = self.client.write_api()
    

    def saveToInfluxDB(self, df):
        try:
            self.write_api.write(bucket=self.bucket, org=self.org, record=df,
             data_frame_timestamp_column="time", data_frame_measurement_name="attack" )
            print(f"Writen {len(df)} data points to influxdb")            
        
        except Exception as ex:
            print(f"[x] Error {str(ex)}")

        