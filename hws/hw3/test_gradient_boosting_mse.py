import numpy as np

from gradient_boosting_mse import *


def test_train_predict():
   X_train, y_train = load_dataset("data/tiny.rent.train")
   X_val, y_val = load_dataset("data/tiny.rent.test")

   y_mean, trees = gradient_boosting_mse(X_train, y_train, 5, max_depth=2, nu=0.1)
   assert(np.around(y_mean, decimals=4)== 3839.1724)
   
   y_hat_train = gradient_boosting_predict(X_train, trees, y_mean, nu=0.1)
   assert(np.around(r2_score(y_train, y_hat_train), decimals=4)==0.5527)

   y_hat = gradient_boosting_predict(X_val, trees, y_mean, nu=0.1)
   assert(np.around(r2_score(y_val, y_hat), decimals=4)==0.5109)
