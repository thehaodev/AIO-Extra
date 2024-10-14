import pandas as pd
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.datasets import load_iris


def run_simple_visualization():
    attribute_names = ['love_math', 'love_art', 'love_english']
    data = {
        'love_math': ['yes', 'yes', 'no', 'no', 'yes', 'yes', 'no'],
        'love_art': ['yes', 'no', 'yes', 'yes', 'yes', 'no', 'no'],
        'love_english': ['no', 'no', 'yes', 'yes', 'yes', 'yes', 'yes'],
        'love_ai': ['no', 'no', 'yes', 'yes', 'yes', 'no', 'no']}
    df = pd.DataFrame(data, columns=data.keys())

    # Define model
    classifier = tree.DecisionTreeClassifier(criterion="gini",
                                             max_depth=4, min_samples_leaf=10)

    # get data for class = x, for feature = y
    one_hot_data = pd.get_dummies(df[["love_math", "love_art",
                                      "love_english"]], drop_first=True)
    x = one_hot_data.iloc[:, :].values
    y = df['love_ai'].values

    # Train model
    classifier.fit(x, y)

    # Visualization
    _, ax = plt.subplots(figsize=(10, 10))
    tree.plot_tree(classifier, ax=ax, feature_names=attribute_names, filled=True)

    # Predict data with yes, yes, no
    result = classifier.predict([[1, 1, 0]])
    print(result)

    plt.show()


def run_simple_visualization_iris():
    # Get data for class = x and features = y
    dataset = load_iris()
    x = dataset.data
    y = dataset.target

    # Define model
    classifier = tree.DecisionTreeClassifier(criterion="gini",
                                             max_depth=4, min_samples_leaf=10)

    # Train
    classifier.fit(x, y)

    # Visualization
    _, ax = plt.subplots(figsize=(10, 10))
    tree.plot_tree(classifier, ax=ax, feature_names=["sepal length", "sepal width",
                                                     "petal length", "petal width"],
                   filled=True)

    plt.show()


run_simple_visualization_iris()
