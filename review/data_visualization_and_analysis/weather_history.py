import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('weatherHistory_v1.csv')
data_series = df['Temperature (C)']
data_series.info()


def run_simple_analysis():

    print(data_series.size)
    print(data_series.dtypes)
    print(data_series.shape)

    # Show Density
    plt.figure(1)
    data_series.plot.kde(0.2)

    # Show Distribution
    plt.figure(2)
    data_series.hist()

    # Show box plot
    plt.figure(3)
    data_series.plot.box()

    plt.show()


def visualization():
    plt.figure(1)
    data_series.plot()

    plt.figure(2)
    data_series[:100].plot()

    # Fill data
    plt.figure(3)
    data_filled = data_series.ffill()
    data_filled[:100].plot()

    # Smooth data
    plt.figure(4)
    data_smooth = data_filled.rolling(window=5).mean()
    data_smooth[:100].plot()

    # Smooth data 2
    plt.figure(5)
    data_smooth = data_filled.ewm(com=2).mean()
    data_smooth[:100].plot()

    plt.show()


visualization()
