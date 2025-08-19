import os
import sys
import json

from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")
print(MONGO_DB_URL)

import certifi  #types of connection certification 
ca = certifi.where()   #certificate authorities trusted ssl,tls connections
#server you are connected has a trusted certificate ensured.


import pandas as pd
import numpy as np
import pymongo
from networksecurity import NetworkSecurityException
from networksecurity.logging.logger import logging

class NetworkDataExtract():
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def csv_to_json(self,file_path):
        try:
            data = pd.read_csv(file_path)
            data.reset_index(drop=True,inplace=True)
            records = list(json.loads(data.T.to_json()).values()) #converts csv to list of json [{a:1,b:2,c:3},{a:8,b:9,c:0},....]
            return records
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def insert_data_mongodb(self,records,database,collection):
        try:
            self.database = database
            self.collection = collection
            self.records = records

            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            self.database = self.mongo_client[self.database]

            self.collection = self.database[self.collection]
            self.collection.insert_many(self.records)
            return (len(self.records))
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        

if __name__ == '__main__':
    FILE_PATH = "Network_Data/phisingData.csv"
    DATABASE = "enugulamaruthi"
    Collection = "NetworkData"
    networkobj = NetworkDataExtract()
    records = networkobj.csv_to_json(file_path=FILE_PATH)
    print(records)
    no_of_records = networkobj.insert_data_mongodb(records,DATABASE,Collection)
    print(no_of_records)

        
        
        