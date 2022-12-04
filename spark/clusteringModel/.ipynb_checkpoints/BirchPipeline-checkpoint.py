from sklearn.cluster import Birch
import numpy as np 
import pandas as pd 
import glob
import pickle
import matplotlib.pyplot as plt
from sklearn.datasets import load_svmlight_file
import warnings
warnings.filterwarnings('ignore')

pca = pickle.load(open('../transformers/pca_2.pickle', 'rb'))
sc = pickle.load(open('../transformers/scaler_2.pickle', 'rb'))    
cont=['P1_B2004', 'P1_B2016', 'P1_B3004', 'P1_B3005', 'P1_B4002', 'P1_B4005', 'P1_B400B', 'P1_B4022', 'P1_FCV01D', 'P1_FCV01Z', 'P1_FCV02D', 'P1_FCV02Z', 'P1_FCV03D', 'P1_FCV03Z', 'P1_FT01', 'P1_FT01Z', 
'P1_FT02', 'P1_FT02Z', 'P1_FT03', 'P1_FT03Z', 'P1_LCV01D', 'P1_LCV01Z', 'P1_LIT01', 'P1_PCV01D', 'P1_PCV01Z', 'P1_PCV02Z', 'P1_PIT01', 'P1_PIT02', 'P1_TIT01', 'P1_TIT02', 'P2_24Vdc', 'P2_SIT01', 'P2_VT01e', 'P2_VXT02', 'P2_VXT03', 'P2_VYT02', 'P2_VYT03', 'P3_LCP01D', 'P3_LCV01D', 'P3_LT01', 'P4_HT_FD', 'P4_HT_LD', 'P4_HT_PO', 'P4_LD', 'P4_ST_FD', 'P4_ST_LD', 'P4_ST_PO', 'P4_ST_PS', 'P4_ST_PT01', 'P4_ST_TT01']   
    
#load the data in df 
train_files = glob.glob("../data/train*.csv")
df = pd.concat((pd.read_csv(f,delimiter=';') for f in train_files))
print(df)   

pd_df = df[cont]
data_norm = pd.DataFrame( sc.transform(pd_df) , columns=cont)
reduced = pca.transform( data_norm )
print(reduced)   

from sklearn.cluster import KMeans
cs = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(reduced)
    cs.append(kmeans.inertia_)
plt.plot(range(1, 11), cs)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()



model = Birch(branching_factor = 50, n_clusters = 4, threshold = 1.5)
model.fit(reduced)


pickle.dump(model, open('BirchModel.pkl', 'wb'))
#pickle the model 