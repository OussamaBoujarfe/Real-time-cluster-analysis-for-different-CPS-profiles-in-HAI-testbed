from river import cluster
from river import stream
import numpy as np
import random

import os, time, datetime, io, csv
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score




pca = pickle.load(open('./transformers/pca_2.pickle', 'rb'))
sc = pickle.load(open('./transformers/scaler_2.pickle', 'rb'))    
cont=['P1_B2004', 'P1_B2016', 'P1_B3004', 'P1_B3005', 'P1_B4002', 'P1_B4005', 'P1_B400B', 'P1_B4022', 'P1_FCV01D', 'P1_FCV01Z', 'P1_FCV02D', 'P1_FCV02Z', 'P1_FCV03D', 'P1_FCV03Z', 'P1_FT01', 'P1_FT01Z', 
'P1_FT02', 'P1_FT02Z', 'P1_FT03', 'P1_FT03Z', 'P1_LCV01D', 'P1_LCV01Z', 'P1_LIT01', 'P1_PCV01D', 'P1_PCV01Z', 'P1_PCV02Z', 'P1_PIT01', 'P1_PIT02', 'P1_TIT01', 'P1_TIT02', 'P2_24Vdc', 'P2_SIT01', 'P2_VT01e', 'P2_VXT02', 'P2_VXT03', 'P2_VYT02', 'P2_VYT03', 'P3_LCP01D', 'P3_LCV01D', 'P3_LT01', 'P4_HT_FD', 'P4_HT_LD', 'P4_HT_PO', 'P4_LD', 'P4_ST_FD', 'P4_ST_LD', 'P4_ST_PO', 'P4_ST_PS', 'P4_ST_PT01', 'P4_ST_TT01']    


LOCAL_PATH = './data/train1.csv'
df = pd.read_csv(LOCAL_PATH,delimiter=';')
pd_df = df[cont]






data_norm = pd.DataFrame( sc.transform(pd_df) , columns=cont)
# PCA
reduced = pca.transform( data_norm )
principal_components =  pd.DataFrame(data = reduced, columns=[f"P{col + 1}" for col in range(reduced.shape[1])])


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


"""
streamkmeans_1 = cluster.STREAMKMeans(chunk_size=390, n_clusters=4, halflife=0.5, sigma=2.5, seed=0)
streamkmeans_2 = cluster.STREAMKMeans(chunk_size=20, n_clusters=3, halflife=0.5, sigma=2.5, seed=0)
streamkmeans_3 = cluster.STREAMKMeans(chunk_size=300, n_clusters=10, halflife=1.5, sigma=2.5, seed=0)
streamkmeans_4 = cluster.STREAMKMeans(chunk_size=30, n_clusters=5, halflife=1.5, sigma=1.5, seed=0)
streamkmeans_5 = cluster.STREAMKMeans(chunk_size=3000, n_clusters=3, halflife=1.5, sigma=1.3, seed=0)

tim= time.time()

labels=[]
for x, _ in stream.iter_pandas(principal_components):
    streamkmeans_1 = streamkmeans_1.learn_one(x)
    labels.append(streamkmeans_1.predict_one(x))
print(pd.DataFrame(labels).value_counts())
    
    #streamkmeans_2 = streamkmeans_2.learn_one(x)
    #streamkmeans_3 = streamkmeans_3.learn_one(x)
    #streamkmeans_4 = streamkmeans_4.learn_one(x)
    #streamkmeans_5 = streamkmeans_5.learn_one(x)

tim2= time.time()

print(tim2-tim)

with open('./models/streamkmeans.pickle', 'wb') as f:
    pickle.dump(streamkmeans_1, f)"""