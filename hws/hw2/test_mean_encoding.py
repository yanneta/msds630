import numpy as np
import pandas as pd
from mean_encoding import *  

def test_reg_target_encoding():
   train = pd.read_csv("tiny_data/avazu_train_tiny.csv")
   reg_target_encoding(train)
   encoded_feature = train["device_type_mean_enc"].values
   corr = np.corrcoef(train["click"].values, encoded_feature)[0][1]
   assert(np.around(corr, decimals=4) == 0.1505)

   
def test_target_encoding_validation():
   train = pd.read_csv("tiny_data/avazu_train_tiny.csv")
   valid = pd.read_csv("tiny_data/avazu_valid_tiny.csv")
   mean_encoding_test(valid, train)
   encoded_feature_mean = valid["device_type_mean_enc"].values.mean()
   assert(np.around(encoded_feature_mean, decimals=4) == 0.1785)

