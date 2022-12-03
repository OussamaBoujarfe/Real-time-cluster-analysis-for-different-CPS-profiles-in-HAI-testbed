from sklearn.cluster import Birch
import numpy as np 
import pandas as pd 
import glob
import pickle

pca = pickle.load(open('../transformers/pca_2.pickle', 'rb'))
sc = pickle.load(open('../transformers/scaler_2.pickle', 'rb'))    
cont=['P1_B2004', 'P1_B2016', 'P1_B3004', 'P1_B3005', 'P1_B4002', 'P1_B4005', 'P1_B400B', 'P1_B4022', 'P1_FCV01D', 'P1_FCV01Z', 'P1_FCV02D', 'P1_FCV02Z', 'P1_FCV03D', 'P1_FCV03Z', 'P1_FT01', 'P1_FT01Z', 
'P1_FT02', 'P1_FT02Z', 'P1_FT03', 'P1_FT03Z', 'P1_LCV01D', 'P1_LCV01Z', 'P1_LIT01', 'P1_PCV01D', 'P1_PCV01Z', 'P1_PCV02Z', 'P1_PIT01', 'P1_PIT02', 'P1_TIT01', 'P1_TIT02', 'P2_24Vdc', 'P2_SIT01', 'P2_VT01e', 'P2_VXT02', 'P2_VXT03', 'P2_VYT02', 'P2_VYT03', 'P3_LCP01D', 'P3_LCV01D', 'P3_LT01', 'P4_HT_FD', 'P4_HT_LD', 'P4_HT_PO', 'P4_LD', 'P4_ST_FD', 'P4_ST_LD', 'P4_ST_PO', 'P4_ST_PS', 'P4_ST_PT01', 'P4_ST_TT01']   
    
#load the data in df 
train_files = glob.glob("../data/train*.csv")
df = pd.concat((pd.read_csv(f,delimiter=';') for f in train_files))
print(df)   

data_norm = df.DataFrame( sc.transform(pd_df) , columns=cont)
reduced = pca.transform( data_norm )
processed =  pd.DataFrame(data = reduced, columns=[f"P{col + 1}" for col in range(reduced.shape[1])])
print(processed)   

#train data in procedssed where processed has columns p1 and p2
#pickle the model 