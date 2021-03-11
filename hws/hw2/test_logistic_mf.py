from logistic_mf import *


def init_model_and_data():
    train = pd.read_csv("tiny_data/train_books_ratings_tiny.csv")
    valid = pd.read_csv("tiny_data/valid_books_ratings_tiny.csv")
    train_df = encode_data(train, train=None)
    valid_df = encode_data(valid, train=train)
    num_users = len(train.user.unique())
    num_items = len(train.item.unique())
    model = MF(num_users, num_items)
    return model, train_df, valid_df


def test_model():
    model, df, _  = init_model_and_data() 
    u = torch.LongTensor(df.user.values)
    v = torch.LongTensor(df.item.values)
    y_hat = model(u, v)
    y_hat_true = np.array(
    [0.5193, 0.5127, 0.5133, 0.5136, 0.5124, 0.5120, 0.5186, 0.5129, 0.5152, 0.5151])
    assert(np.all(np.abs(y_hat.detach().numpy()[:10] - y_hat_true) < 0.0001))


def test_val_metrics():
    model, df, _  = init_model_and_data()
    loss, acc = valid_metrics(model, df)
    assert(np.around(loss, decimals=2) == 0.73)


def test_train_one_epoch():
    model, df, _  = init_model_and_data()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=0)
    for i in range(3):
        train_loss = train_one_epoch(model, df, optimizer)
    assert(np.around(train_loss, decimals=2) == 0.53)

