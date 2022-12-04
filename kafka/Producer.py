from kafka import KafkaProducer
import json
import csv
import sys
import os 
import time
import datetime

if __name__ == "__main__":
    BOOTSTRAP_SERVER = 'kafka:9092'
    TOPIC = 'HAI'
    DATA_PATH = './data/test2.csv'
    # Load the dataa
    print('Loading data')
    data = csv.DictReader(open(DATA_PATH),delimiter=';') 
    # Initialize producer
    producer = KafkaProducer(bootstrap_servers = BOOTSTRAP_SERVER)
    # Time interval
    startTime = time.time()
    waitSeconds = .1

    """
    t= time.strptime("2019-09-11 20:02:56", '%Y-%m-%d %H:%M:%S')
    n= time.localtime()
    time1 = time.mktime(t)
    time2 = time.mktime(n)
    print(time2-time1)
    """



    for row in data:
        #print(row)
        # Convert to a JSON format
        current_time = time.localtime()
        old_time = datetime.datetime.fromisoformat(row['time'])
        #new_time = old_time + datetime.timedelta(seconds=101271679)
        new_time = old_time + datetime.timedelta(seconds=97209960)
        row['time'] =   new_time.isoformat(' ')
        payload = json.dumps(row)
        # Produce
        producer.send(TOPIC, value=payload.encode('utf-8'))
        #print(payload)
        # Wait a number of second until next message
        time.sleep(waitSeconds - ((time.time() - startTime) % waitSeconds))
    print("Emptied.")