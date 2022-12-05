import pickle
from river import stream
import pandas as pd

class DenStream:
    def __init__(self):
        self.denstream_model = pickle.load(open('./Denstream/denstream.pkl', 'rb')) 
    
    def model(self, df):
        denstream_labels=[]
        for x, _ in stream.iter_pandas(df): 
            denstream_labels.append(self.denstream_model.predict_one(x))
            self.streamkmeans_model = self.denstream_model.learn_one(x)
        skm_l_df = pd.DataFrame( denstream_labels, columns=["DeanStream labels"])
        
        
        return skm_l_df