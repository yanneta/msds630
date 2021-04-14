from boosting_nn import *

def test_normalize():
   X, Y = parse_spambase_data("data/tiny.spam.train")
   X_val, Y_val = parse_spambase_data("data/tiny.spam.test")
   X, X_val = normalize(X, X_val)
   xx = np.around(X_val[0, :3],3)
   assert(np.array_equal(xx, np.array([-0.433, -0.491, -0.947])))

def test_compute_residual():
   y = np.array([-1, -1, 1, 1])
   fm = np.array([-0.4, .1, -0.3 , 2])
   res = compute_pseudo_residual(y, fm)
   xx = np.around(res, 3)
   actual = np.array([-0.401, -0.525,  0.574,  0.119])
   assert(np.array_equal(xx, actual))

def test_boosting():
   X, Y = parse_spambase_data("data/tiny.spam.train")
   X_val, Y_val = parse_spambase_data("data/tiny.spam.test")
   X, X_val = normalize(X, X_val)
   nu = .1
   f0, models = boostingNN(X, Y, num_iter=10, nu=nu)
   y_hat = gradient_boosting_predict(X, f0, models, nu=nu)
   acc_train = accuracy(Y, y_hat)
   assert(acc_train==1)
   y_hat = gradient_boosting_predict(X_val, f0, models, nu=nu)
   acc_val = accuracy(Y_val, y_hat)
   assert(acc_val==0.8)
