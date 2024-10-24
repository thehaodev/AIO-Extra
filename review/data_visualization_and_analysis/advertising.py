import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def run():
    # Read data
    df = pd.read_csv('advertising_simple.csv')

    # Show plot pie of feature number
    plt.figure(1)
    sample1 = df.loc[0][:3]
    print(sample1)
    print(sample1.info())
    sample1.plot.pie()

    # Show plot pie of df[TV]
    plt.figure(2)
    df['TV'].plot.pie()

    # Show 3 feature with kind = bar
    plt.figure(3)
    selected_columns = df.iloc[:, :3]
    print(selected_columns)
    selected_columns.plot(kind='bar')

    # Show 3 feature with kind = line
    plt.figure(4)
    selected_columns.plot(kind="line")

    # Show 3 feature with kind = barh
    plt.figure(5)
    selected_columns.plot(kind='barh')

    # Show plot kind scatter of TV and Sales
    plt.figure(6)
    df.plot(x='TV', y='Sales', kind='scatter')

    # Show plot kind scatter of Newspaper and Sales
    plt.figure(7)
    df.plot(x='Newspaper', y='Sales', kind='scatter')

    # Show plot kind scatter of Radio and Sales
    plt.figure(8)
    df.plot(x='Radio', y='Sales', kind='scatter')

    plt.show()


run()
