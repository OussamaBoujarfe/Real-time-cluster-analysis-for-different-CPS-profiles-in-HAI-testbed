
import os, time, datetime, io, csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



# Importing the datase
def load_data(path):
  return pd.read_csv(path,delimiter=';')

# Rewrite column names
def rewrite_cols(data):
  cols = list(data.columns)
  cols[1:] = [i.split('_')[1] for i in cols[1:]]
  data.columns = cols
  return data


def preprocess_2(data):
  permitted_cols=set(data.columns).difference({'time','attack','attack_P1','attack_P2','attack_P3'})
  for col in ['time','attack','attack_P1','attack_P2','attack_P3']:
    print(f"> Dropping {col}")
    data = data.drop(col, axis=1)

  CATEG_RANGE = 3
  cat, cont = [], []

  for col in data.columns:
    if len(data[col].value_counts()) <= CATEG_RANGE:
      cat.append(col)
    else:
      cont.append(col)

  return data, cat, cont
""" 
> Dropping P1_PCV02D
> Dropping P2_Auto
> Dropping P2_Emgy
> Dropping P2_On
> Dropping P2_SD01
> Dropping P2_TripEx
> Dropping P3_LH
> Dropping P3_LL
> Dropping P4_HT_PS
> Dropping attack_P2
> Dropping attack_P3 
"""


def Standardize_2(data):
  sc = StandardScaler()
  permitted_cols=set(data.columns).difference({'time','attack','attack_P1','attack_P2','attack_P3'})
  permitted_cols= ['P1_B2004', 'P1_B2016', 'P1_B3004', 'P1_B3005', 'P1_B4002', 'P1_B4005', 'P1_B400B', 'P1_B4022', 'P1_FCV01D', 'P1_FCV01Z', 'P1_FCV02D', 'P1_FCV02Z', 'P1_FCV03D', 'P1_FCV03Z', 'P1_FT01', 'P1_FT01Z', 
'P1_FT02', 'P1_FT02Z', 'P1_FT03', 'P1_FT03Z', 'P1_LCV01D', 'P1_LCV01Z', 'P1_LIT01', 'P1_PCV01D', 'P1_PCV01Z', 'P1_PCV02Z', 'P1_PIT01', 'P1_PIT02', 'P1_TIT01', 'P1_TIT02', 'P2_24Vdc', 'P2_SIT01', 'P2_VT01e', 'P2_VXT02', 'P2_VXT03', 'P2_VYT02', 'P2_VYT03', 'P3_LCP01D', 'P3_LCV01D', 'P3_LT01', 'P4_HT_FD', 'P4_HT_LD', 'P4_HT_PO', 'P4_LD', 'P4_ST_FD', 'P4_ST_LD', 'P4_ST_PO', 'P4_ST_PS', 'P4_ST_PT01', 'P4_ST_TT01']
  sc.fit(data[permitted_cols])
  
  data_norm = pd.DataFrame( sc.transform(data[permitted_cols]))
  data_norm_c = data_norm.copy()
  data_norm.columns = data[permitted_cols].columns
  #data_norm.insert(0, 'time', data['time'])
  print(data_norm.describe())

  with open('./transformers/scaler_2.pickle', 'wb') as f:
    pickle.dump(sc, f)
  return data_norm

def apply_pca_2(data, cont):
  data_scaled = Standardize_2(data[cont])
  data_ = data_scaled
  permitted_cols=set(data.columns).difference({'time','attack','attack_P1','attack_P2','attack_P3'})
  permitted_cols= ['P1_B2004', 'P1_B2016', 'P1_B3004', 'P1_B3005', 'P1_B4002', 'P1_B4005', 'P1_B400B', 'P1_B4022', 'P1_FCV01D', 'P1_FCV01Z', 'P1_FCV02D', 'P1_FCV02Z', 'P1_FCV03D', 'P1_FCV03Z', 'P1_FT01', 'P1_FT01Z', 
'P1_FT02', 'P1_FT02Z', 'P1_FT03', 'P1_FT03Z', 'P1_LCV01D', 'P1_LCV01Z', 'P1_LIT01', 'P1_PCV01D', 'P1_PCV01Z', 'P1_PCV02Z', 'P1_PIT01', 'P1_PIT02', 'P1_TIT01', 'P1_TIT02', 'P2_24Vdc', 'P2_SIT01', 'P2_VT01e', 'P2_VXT02', 'P2_VXT03', 'P2_VYT02', 'P2_VYT03', 'P3_LCP01D', 'P3_LCV01D', 'P3_LT01', 'P4_HT_FD', 'P4_HT_LD', 'P4_HT_PO', 'P4_LD', 'P4_ST_FD', 'P4_ST_LD', 'P4_ST_PO', 'P4_ST_PS', 'P4_ST_PT01', 'P4_ST_TT01']
  
  pca = PCA(n_components = 2)
  pca.fit(data_[permitted_cols])
  reduced = pca.transform(data_[permitted_cols])
  principal_components = pd.DataFrame(data = reduced, columns=[f"P{col + 1}" for col in range(reduced.shape[1])])
  with open('./transformers/pca_2.pickle', 'wb') as f:
      pickle.dump(pca, f)
  return principal_components  

# Merge extracted features with scaled data
def append_cols(A, B, cols):
  for col in cat:
    A[col] = B[col]
  #print(A.columns)
  #print(B.columns)
  A.insert(0, 'time', B['time'])
  return A

def split_save(data):
  # Split the data
  i = round(len(data)*.8)
  train = data[0:i] # 80% of the data
  test = data[i+1:len(data)] # 20% of the data
  print(len(train), len(test))

  # Split the data into two sets and save locally
  train.to_csv(f"./data/train_1.csv", sep = ',', encoding = 'utf-8', index = False)
  test.to_csv(f"./data/test_1.csv", sep = ',', encoding = 'utf-8', index = False)

if __name__=="__main__":
  # Load data
  LOCAL_PATH = './data/train_0.csv'
  data = load_data(LOCAL_PATH)
  # Rewrite cols
  #data = rewrite_cols(data)

  # Preprocess
  data, cat, cont = preprocess_2(data)
  print(data.columns)
  print(cat)
  print(cont)
  # PCA
  data_processed = apply_pca_2(data, cont)


  plt.axes().set_aspect('equal','box')
  plt.scatter(data_processed['P1'], data_processed['P2'])
  plt.show()
  # Merge
  #R = append_cols(data_processed, data, cat)
  # Split and save the data
  #split_save(R)