# Data Preprocessing of SWaT Dataset
import os, time, datetime, io, csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


from influx_writer import InfluxDBWriter

# Importing the datase
def load_data(path):
  return pd.read_csv(path,delimiter=';')

# Rewrite column names
def rewrite_cols(data):
  cols = list(data.columns)
  cols[1:] = [i.split('_')[1] for i in cols[1:]]
  data.columns = cols
  return data

# Data Exploration
def EDA(data):
  ## Data shape
  print(f'[>] Data shape is {data.shape}')
  ## Data types
  print(f'[>] Data types are: ')
  print(data.dtypes)
  ## Missing values 
  print(f'[>] Missing values: ')
  print(data.isnull().sum().sum())
  ## Duplicated rows
  print(f'[>] Duplicated values: ')
  print(data.duplicated().sum())
  ## Values to datetime
  print(f'[...] Converting string time to timestamp')
  data['tcdime'] = [datetime.datetime.strptime(t, "%d/%m/%Y %H:%M:%S.%f %p") for t in data['time']] 
  ## Distribution by days
  print(f'[>] Distribution by days: ')
  day = data['time'].dt.day
  cnt_srs = day.value_counts()
  plt.figure(figsize=(12,6))
  sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=sns.color_palette()[3])
  plt.xticks(rotation='vertical')
  plt.xlabel('Date', fontsize=12)
  plt.ylabel('Number of Occurrences', fontsize=12)
  plt.show()
  ## Distribution by hours
  print(f'[>] Distribution by hours: ')
  hours = data['time'].dt.hour
  cnt_srs = hours.value_counts()
  plt.figure(figsize=(12,6))
  sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=sns.color_palette()[3])
  plt.xticks(rotation='vertical')
  plt.xlabel('Hour', fontsize=12)
  plt.ylabel('Number of Occurrences', fontsize=12)
  plt.show()
  ## Plots
  print(f'[>] Plot columns by timestamp: ')
  j = 1
  while j < len(cols)/2:
    plt.figure(figsize = (15, 10))
    for i, col in enumerate(cols[j:j+3]):
      plt.subplot(2, 3, i + 1)    
      plt.scatter(data['time'], data[col], s=10)
      plt.xlabel('time')
      plt.title(str(col))
      plt.xticks(rotation=40)
      j += 1
    plt.show()

# Preprocess data
def preprocess(data):
  # Find columns with only one values. We will drop them since they don't have any effect.
  for col in data.columns:
    if len(data[col].value_counts()) == 1:
      print(f"> Dropping {col}")
      data = data.drop(col, axis=1)
  # Get categorical columns
  CATEG_RANGE = 3
  cat, cont = [], []

  for col in data.columns:
    if len(data[col].value_counts()) <= CATEG_RANGE:
      cat.append(col)
    else:
      cont.append(col)

  return data, cat, cont

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



# Standardize values
def Standardize(data):
  sc = StandardScaler()
  sc.fit(data.iloc[:, 1:len(data.columns)+1])
  data_norm = pd.DataFrame( sc.transform(data.iloc[:, 1:len(data.columns)+1]) )
  data_norm_c = data_norm.copy()
  data_norm.columns = data.iloc[:, 1:len(data.columns)+1].columns
  data_norm.insert(0, 'time', data.iloc[:,0])
  # It is important to use binary access
  with open('./transformers/scaler.pickle', 'wb') as f:
    pickle.dump(sc, f)
  return data_norm

# Principal Component Analysis
def apply_pca(data, cont):
  data_scaled = Standardize(data[cont])
  data_ = data_scaled
  pca = PCA(n_components = 0.95)
  pca.fit(data_.iloc[:, data.columns!='time'])
  reduced = pca.transform(data_.iloc[:, 1:len(data.columns)+1])
  principal_components = pd.DataFrame(data = reduced, columns=[f"P{col + 1}" for col in range(reduced.shape[1])])
  with open('./transformers/pca.pickle', 'wb') as f:
      pickle.dump(pca, f)
  return principal_components

def Standardize_2(data):
  sc = StandardScaler()
  permitted_cols=set(data.columns).difference({'time','attack','attack_P1','attack_P2','attack_P3'})
  permitted_cols= ['P1_B2004', 'P1_B2016', 'P1_B3004', 'P1_B3005', 'P1_B4002', 'P1_B4005', 'P1_B400B', 'P1_B4022', 'P1_FCV01D', 'P1_FCV01Z', 'P1_FCV02D', 'P1_FCV02Z', 'P1_FCV03D', 'P1_FCV03Z', 'P1_FT01', 'P1_FT01Z', 
'P1_FT02', 'P1_FT02Z', 'P1_FT03', 'P1_FT03Z', 'P1_LCV01D', 'P1_LCV01Z', 'P1_LIT01', 'P1_PCV01D', 'P1_PCV01Z', 'P1_PCV02Z', 'P1_PIT01', 'P1_PIT02', 'P1_TIT01', 'P1_TIT02', 'P2_24Vdc', 'P2_SIT01', 'P2_VT01e', 'P2_VXT02', 'P2_VXT03', 'P2_VYT02', 'P2_VYT03', 'P3_LCP01D', 'P3_LCV01D', 'P3_LT01', 'P4_HT_FD', 'P4_HT_LD', 'P4_HT_PO', 'P4_LD', 'P4_ST_FD', 'P4_ST_LD', 'P4_ST_PO', 'P4_ST_PS', 'P4_ST_PT01', 'P4_ST_TT01']
  sc.fit(data[permitted_cols])
  print(len(data.columns), len(permitted_cols))
  data_norm = pd.DataFrame( sc.transform(data[permitted_cols]))
  data_norm_c = data_norm.copy()
  data_norm.columns = data[permitted_cols].columns
  #data_norm.insert(0, 'time', data['time'])

  with open('./transformers/scaler_2.pickle', 'wb') as f:
    pickle.dump(sc, f)
  return data_norm

def apply_pca_2(data, cont):
  data_scaled = Standardize_2(data[cont])
  data_ = data_scaled
  permitted_cols=set(data.columns).difference({'time','attack','attack_P1','attack_P2','attack_P3'})
  permitted_cols= ['P1_B2004', 'P1_B2016', 'P1_B3004', 'P1_B3005', 'P1_B4002', 'P1_B4005', 'P1_B400B', 'P1_B4022', 'P1_FCV01D', 'P1_FCV01Z', 'P1_FCV02D', 'P1_FCV02Z', 'P1_FCV03D', 'P1_FCV03Z', 'P1_FT01', 'P1_FT01Z', 
'P1_FT02', 'P1_FT02Z', 'P1_FT03', 'P1_FT03Z', 'P1_LCV01D', 'P1_LCV01Z', 'P1_LIT01', 'P1_PCV01D', 'P1_PCV01Z', 'P1_PCV02Z', 'P1_PIT01', 'P1_PIT02', 'P1_TIT01', 'P1_TIT02', 'P2_24Vdc', 'P2_SIT01', 'P2_VT01e', 'P2_VXT02', 'P2_VXT03', 'P2_VYT02', 'P2_VYT03', 'P3_LCP01D', 'P3_LCV01D', 'P3_LT01', 'P4_HT_FD', 'P4_HT_LD', 'P4_HT_PO', 'P4_LD', 'P4_ST_FD', 'P4_ST_LD', 'P4_ST_PO', 'P4_ST_PS', 'P4_ST_PT01', 'P4_ST_TT01']
  
  pca = PCA(n_components = 0.95)
  pca.fit(data[permitted_cols])
  reduced = pca.transform(data[permitted_cols])
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
  # EDA
  # EDA(data)
  # Preprocess
  data, cat, cont = preprocess_2(data)
  print(data.columns)
  print(cat)
  print(cont)
  # PCA
  data_processed = apply_pca_2(data, cont)
  # Merge
  #R = append_cols(data_processed, data, cat)
  # Split and save the data
  #split_save(R)