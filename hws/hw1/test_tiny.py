from cf import *

def test_encoding():
    df = pd.read_csv("tiny_training.csv")
    df, num_users, num_movies = encode_data(df)
    users = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 6, 6])
    movies = np.array([0, 1, 1, 2, 0, 1, 0, 3, 0, 3, 3, 1, 3])
    np.testing.assert_equal(df["userId"].values, users)
    np.testing.assert_equal(df["movieId"].values, movies)

def test_cost():
    df = pd.read_csv("tiny_training.csv")
    df, num_users, num_movies = encode_data(df)
    emb_user = np.ones((7, 5))
    emb_movie = np.ones((4, 5))
    error = cost(df, emb_user, emb_movie)
    assert(np.around(error, decimals=2) == 4.08)

def test_gradient():
    K = 5
    df = pd.read_csv("tiny_training.csv")
    df, num_users, num_movies = encode_data(df)
    emb_user = create_embedings(7, K)
    emb_movie = create_embedings(4, K)
    Y = df2matrix(df, emb_user.shape[0], emb_movie.shape[0])
    grad_user, grad_movie = gradient(df, Y, emb_user, emb_movie)
    approx = np.array([finite_difference(df, emb_user, emb_movie, ind_u=2, k=i) for i in range(K)])
    assert(np.all(np.abs(grad_user[2] - approx) < 0.0001))
    approx = np.array([finite_difference(df, emb_user, emb_movie, ind_m=3, k=i) for i in range(K)])
    assert(np.all(np.abs(grad_movie[3] - approx) < 0.0001))

def test_gd():
    df = pd.read_csv("tiny_training.csv")
    df, num_users, num_movies = encode_data(df)
    emb_user = create_embedings(7, 7)
    emb_movie = create_embedings(5, 7)
    emb_user, emb_movie = gradient_descent(df, emb_user, emb_movie, iterations=200, learning_rate=0.01)
    train_mse = cost(df, emb_user, emb_movie)
    assert(np.around(train_mse, decimals=2) == 0.59)
