import pandas as pd
import day_18
import gdown
from gdown.exceptions import FileURLRetrievalError


def get_table_data_set():
    url = "https://drive.google.com/uc?id=1iA0WmVfW88HyJvTBSQDI5vesf-pgKabq"
    try:
        gdown.download(url)
    except FileURLRetrievalError:
        print("File cannot download")

    dataset = pd.read_csv('advertising.csv')

    return dataset


def predict(x, w, b):
    return x*w + b


def gradient(y_hat, y, x):
    dw = 2 * x * (y_hat-y)
    db = 2 * (y_hat-y)

    return dw, db


def update_weight(w, b, lr, dw, db, sample: int):
    w_new = w - lr*dw/sample
    b_new = b - lr*db/sample

    return w_new, b_new


def run():
    b = 1
    w1 = 0
    w2 = 0
    w3 = 0
    lr = 0.00001
    epochs = 1000
    day_18.init(
        project="demo-linear-regression",
        config={
            "learning_rate": lr,
            "epochs": epochs
        }
    )

    dataset = get_table_data_set()
    x1_train = dataset["TV"]
    x2_train = dataset["Radio"]
    x3_train = dataset["Newspaper"]
    y_train = dataset["Sales"]
    sample = len(y_train)
    day_18.run.log({"Dataset": day_18.Table(dataframe=dataset)})

    for _ in range(epochs):
        y_hat = x1_train*w1 + x2_train*w2 + x3_train*w3 + b
        y = y_train
        loss = (y_hat - y) * (y_hat - y)
        day_18.log({"loss": loss.to_numpy()})

        dw1, db1 = gradient(y_hat, y, x1_train)
        dw2, db2 = gradient(y_hat, y, x2_train)
        dw3, db3 = gradient(y_hat, y, x3_train)

        (w1, b) = update_weight(w1, b, lr, dw1, db1, sample)
        (w2, b) = update_weight(w2, b, lr, dw2, db2, sample)
        (w3, b) = update_weight(w3, b, lr, dw3, db3, sample)

    day_18.finish()


run()
