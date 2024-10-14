import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree


def run_simple_visualization():
    # Read data
    dataset = pd.read_csv('Position_Salaries.csv')

    # Get data for class = x and target = y
    x = dataset.iloc[:, 1:2].values
    y = dataset.iloc[:, 2].values

    # Define model
    # We create two model: One with depth = 3 and depth = 4
    regressor_max_depth_three = DecisionTreeRegressor(random_state=0, max_depth=3, ccp_alpha=0.0)
    regressor_max_depth_three.fit(x, y)

    regressor_min_samples_leaf_fourth = DecisionTreeRegressor(random_state=0, min_samples_leaf=4, ccp_alpha=0.0)
    # Fit the regressor object to the dataset.
    regressor_min_samples_leaf_fourth.fit(x, y)

    # visualization
    _, ax = plt.subplots(figsize=(20, 20))
    tree.plot_tree(regressor_max_depth_three, ax=ax, feature_names=["Level"], filled=True)

    plt.show()


run_simple_visualization()
