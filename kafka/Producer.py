from kafka import KafkaProducer
import json
import csv
import sys
import os 
import time

if __name__ == "__main__":
    BOOTSTRAP_SERVER = 'kafka:9092'
    TOPIC = 'HAI'
    DATA_PATH = './data/All1.csv'
    # Load the dataa
    print('Loading data')
    data = csv.DictReader(open(DATA_PATH),delimiter=';') 
    # Initialize producer
    producer = KafkaProducer(bootstrap_servers = BOOTSTRAP_SERVER)
    # Time interval
    startTime = time.time()
    waitSeconds = .6

    
    for row in data:
        #print(row)
        # Convert to a JSON format
        payload = json.dumps(row)
        # Produce
        producer.send(TOPIC, value=payload.encode('utf-8'))
        #print(payload)
        # Wait a number of second until next message
        time.sleep(waitSeconds - ((time.time() - startTime) % waitSeconds))
    print("Emptied.")