from adaboost import *


def test_parse_spambase_data():
    y_test = np.array([1., -1., 1., 1., -1., -1., 1., 1., 1., -1.])
    X, Y = parse_spambase_data("data/tiny.spam.train")
    assert(np.array_equal(Y, y_test))
    n, m = X.shape
    assert(n == 10)
    assert(m == 57)

def test_adaboost():
    X, Y = parse_spambase_data("data/tiny.spam.train")
    trees, weights = adaboost(X, Y, 2)
    y_hat_0 = trees[0].predict(X)
    assert(len(trees) == 2)
    assert(len(weights) == 2)
    assert(isinstance(trees[0], DecisionTreeClassifier))
    y_hat_true = np.array([ 1., -1.,  1.,  1., -1., -1., -1.,  1.,  1., -1.])
    assert(np.array_equal(y_hat_0, y_hat_true))

def test_adaboost_predict():
    x = np.array([[0, -1], [1, 0], [-1, 0]])
    y = np.array([-1, 1, 1])
    trees, weights = adaboost(x, y, 1)
    pred = adaboost_predict(x, trees, weights)
    assert(np.array_equal(pred, y))
