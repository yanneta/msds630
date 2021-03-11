import numpy as np
import pandas as pd


from sklearn.model_selection import KFold

def reg_target_encoding(train, col="device_type", target="click", splits=5):
    """ Computes regularize mean encoding.
    Inputs:
       train: training dataframe
       
    """
    kf = KFold(n_splits=splits, shuffle=False)
    new_col = col + "_" + "mean_enc"
    ### BEGIN SOLUTION
    
    ### END SOLUTION


def mean_encoding_test(test, train, col="device_type", target="click"):
    """ Computes target enconding for test data.

    This is similar to how we do validation
    """
    ### BEGIN SOLUTION
    
    ### END SOLUTION

