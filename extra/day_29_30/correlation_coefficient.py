import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def correlation_coefficient(x, y):
    x_m = np.mean(x)
    y_m = np.mean(y)

    r_numerator = 0
    for idx, _ in enumerate(x):
        r_numerator += (x[idx] - x_m) * (y[idx] - y_m)

    sum_x = np.sum((x - x_m) * (x - x_m))
    sum_y = np.sum((y - y_m) * (y - y_m))
    r_denominator = np.sqrt(sum_x * sum_y)

    return r_numerator / r_denominator


X = np.array([10, 20, 30, 40, 50])
Y = np.array([5, 15, 25, 35, 45])
print(correlation_coefficient(X, Y))


def run():
    data = pd.read_csv("advertising.csv")
    list_feature = ["TV", "Radio", "Newspaper", "Sales"]
    r_matrix = np.ones((len(list_feature), len(list_feature)))
    for i, f in enumerate(list_feature):
        for j, f_reversed in enumerate(list_feature):
            data_i = data[f]
            data_j = data[f_reversed]
            r_matrix[i][j] = correlation_coefficient(data_i, data_j)

    result = pd.DataFrame(r_matrix, index=list_feature, columns=list_feature)
    plt.figure(figsize=(10, 8))
    sns.heatmap(result, annot=True, fmt=".2f", linewidth=.5)
    plt.show()


run()
