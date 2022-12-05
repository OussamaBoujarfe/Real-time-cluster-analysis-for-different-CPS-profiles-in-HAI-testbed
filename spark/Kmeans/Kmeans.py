import pickle
from river import stream
import pandas as pd

class Kmeans:
    def __init__(self):
        self.kmeans_model = pickle.load(open('./Kmeans/kmeans.pickle', 'rb')) 
    
    def model(self, df):
        kmeans_labels=[]
        for x, _ in stream.iter_pandas(df): 
            kmeans_labels.append(self.kmeans_model.predict_one(x))
            self.kmeans_model = self.kmeans_model.learn_one(x)
        km_l_df = pd.DataFrame(kmeans_labels, columns=["Kmeans labels"])
        
        
        return km_l_df