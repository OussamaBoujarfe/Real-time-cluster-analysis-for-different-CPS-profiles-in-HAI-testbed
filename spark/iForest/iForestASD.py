from pysad.models import IForestASD
import numpy as np
import random

import os, time, datetime, io, csv
import pandas as pd
import pickle

class iForestASD:
    def __init__(self):
        self.iforest_model = pickle.load(open('./iForest/iforestasd_2048.pickle', 'rb')) 
    
    def model(self, df):
        iforestasd_labels = self.iforest_model.fit_score(df.to_numpy())
        iforestasd_labels_df = pd.DataFrame(iforestasd_labels, columns=["iForestASD anomaly score"])
        
        
        return iforestasd_labels_df