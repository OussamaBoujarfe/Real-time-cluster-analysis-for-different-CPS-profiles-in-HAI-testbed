import pickle
from river import stream
import pandas as pd

class STREAMKmeans:
    def __init__(self):
        self.streamkmeans_model = pickle.load(open('./streamkmeans.pickle', 'rb')) 
    
    def model(self, df):
        streamkmeans_labels=[]
        for x, _ in stream.iter_pandas(df): 
            streamkmeans_labels.append(self.streamkmeans_model.predict_one(x))
            self.streamkmeans_model = self.streamkmeans_model.learn_one(x)
        skm_l_df = pd.DataFrame(streamkmeans_labels, columns=["STREAMKmeans labels"])
        
        
        return skm_l_df