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

import warnings
warnings.filterwarnings('ignore')


pca = pickle.load(open('./transformers/pca_2.pickle', 'rb'))
sc = pickle.load(open('./transformers/scaler_2.pickle', 'rb'))    
cont=['P1_B2004', 'P1_B2016', 'P1_B3004', 'P1_B3005', 'P1_B4002', 'P1_B4005', 'P1_B400B', 'P1_B4022', 'P1_FCV01D', 'P1_FCV01Z', 'P1_FCV02D', 'P1_FCV02Z', 'P1_FCV03D', 'P1_FCV03Z', 'P1_FT01', 'P1_FT01Z', 
'P1_FT02', 'P1_FT02Z', 'P1_FT03', 'P1_FT03Z', 'P1_LCV01D', 'P1_LCV01Z', 'P1_LIT01', 'P1_PCV01D', 'P1_PCV01Z', 'P1_PCV02Z', 'P1_PIT01', 'P1_PIT02', 'P1_TIT01', 'P1_TIT02', 'P2_24Vdc', 'P2_SIT01', 'P2_VT01e', 'P2_VXT02', 'P2_VXT03', 'P2_VYT02', 'P2_VYT03', 'P3_LCP01D', 'P3_LCV01D', 'P3_LT01', 'P4_HT_FD', 'P4_HT_LD', 'P4_HT_PO', 'P4_LD', 'P4_ST_FD', 'P4_ST_LD', 'P4_ST_PO', 'P4_ST_PS', 'P4_ST_PT01', 'P4_ST_TT01']    


LOCAL_PATH = './data/train_0.csv'
df = pd.read_csv(LOCAL_PATH,delimiter=';')
pd_df = df[cont]


data_norm = pd.DataFrame( sc.transform(pd_df) , columns=cont)
# PCA
reduced = pca.transform( data_norm )
principal_components =  pd.DataFrame(data = reduced, columns=[f"P{col + 1}" for col in range(reduced.shape[1])])






streamkmeans_1 = cluster.STREAMKMeans(chunk_size=390, n_clusters=4, halflife=0.5, sigma=2.5, seed=0)

tim= time.time()

labels=[]
for x, _ in stream.iter_pandas(principal_components):
    streamkmeans_1 = streamkmeans_1.learn_one(x)
    labels.append(streamkmeans_1.predict_one(x))
print(pd.DataFrame(labels).value_counts())

tim2= time.time()

print(tim2-tim)

with open('./streamkmeans.pickle', 'wb') as f:
    pickle.dump(streamkmeans_1, f)

    
"""
def nltk_inertia(feature_matrix, centroid, labels):
    sum_ = []
    for i in range(feature_matrix.shape[0]):
        sum_.append(np.sum((feature_matrix[i] - centroid[labels[i]])**2))  #here implementing inertia as given in the docs of scikit i.e sum of squared distance..
    return sum(sum_)

Sum_of_squared_distances = []
sil_scores = []
ch_scores = []
db_scores = []
feature_matrix = principal_components.to_numpy()
K=range(2,16)
for k in K:
    tim= time.time()

    streamkmeans_1 = cluster.STREAMKMeans(chunk_size=30, n_clusters=k, halflife=1.5, sigma=1.5, seed=0)
    labels=[]
    for x, _ in stream.iter_pandas(principal_components):
        streamkmeans_1 = streamkmeans_1.learn_one(x)
        labels.append(streamkmeans_1.predict_one(x))
    lc= []  #centroids array-d
    for kn in range(k):
        
        lc.append([streamkmeans_1.centers[kn]['P1'], streamkmeans_1.centers[kn]['P2']])
    lc = np.array(lc)
    labels = np.array(labels)
    Sum_of_squared_distances.append(nltk_inertia(feature_matrix, lc, labels))
    print("ping?")

    ch_scores.append(calinski_harabasz_score(principal_components, labels))
    #print("b")
    db_scores.append(davies_bouldin_score(principal_components, labels))
    #print("c")
    #sil_scores.append(silhouette_score(principal_components, labels))
    #print("a")
    tim2= time.time()
    print(tim2-tim)
    
plt.figure(104)
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()



plt.figure(106)
plt.plot(K, ch_scores, 'bx-')
plt.xlabel('k')
plt.ylabel('calinski_harabasz_score')
plt.title('max best')
plt.show()

plt.figure(107)
plt.plot(K, db_scores, 'bx-')
plt.xlabel('k')
plt.ylabel('davies_bouldin_score')
plt.title('min best')
plt.show()

"""


