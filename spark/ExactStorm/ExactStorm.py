from pysad.models import ExactStorm
import numpy as np
import random

import os, time, datetime, io, csv
import pandas as pd
import pickle

class ExactStorm:
    def __init__(self):
        self.exactstorm_model = pickle.load(open('./ExactStorm/exactstorm.pickle', 'rb')) 
    
    def model(self, df):
        exactstorm_labels = self.exactstorm_model.fit_score(df.to_numpy())
        exactstorm_labels_df = pd.DataFrame(exactstorm_labels, columns=["ExactStorm anomaly score"])
        
        
        return exactstorm_labels_df