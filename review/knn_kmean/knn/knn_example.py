import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

# Load iris data from sklearn
iris = datasets.load_iris()
iris_x = iris.data
iris_y = iris.target

print('Number of classes: %d' % len(np.unique(iris_y)))
print('Number of data points: %d' % len(iris_y))

x_0 = iris_x[iris_y == 0, :]
print('\nSample from class 0', x_0)

x_1 = iris_x[iris_y == 1, :]
print('\nSample from class 1', x_1)

x_2 = iris_x[iris_y == 2, :]
print('\nSample from class 2', x_2)

# Split train and test sets.
# Method train_test_split allow us to choose random test_size of sample from data x and target y
x_train, x_test, y_train, y_test = train_test_split(iris_x, iris_y, test_size=100, random_state=1)

print(f"Training size: {len(y_train)}")
print(f"Training size: {len(y_test)}")

# Simple test1: Uniform weight
# p = 2 mean to used euclidean_distance
clf = neighbors.KNeighborsClassifier(n_neighbors=10, p=2)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

# observe result
print("Print result for 20 test data points:")
print("Predict labels: ", y_pred[20:40])
print("Ground truth: ", y_test[20:40])
# In result ground truth mean the real label of data in test data.
# We can know exactly because we split data from one dataset
# So we can check by print the same data in test and prediction

# Evaluation method
# Accuracy
print(f'Accuracy of 10NN {100*accuracy_score(y_test, y_pred)}%')
print(f'F1 Score of 10NN {100*f1_score(y_test, y_pred, average='micro')}%')

# Simple test2: Distance weight
clf2 = neighbors.KNeighborsClassifier(n_neighbors=10, p=2, weights='distance')
clf2.fit(x_train, y_train)
y_pred2 = clf2.predict(x_test)

print(f'Accuracy of 10NN Distance Weight {accuracy_score(y_test, y_pred2)}')


# Another weight example
def my_weight(distances):
    sigma2 = 0.5
    return np.exp(-distances**2/sigma2)


clf3 = neighbors.KNeighborsClassifier(n_neighbors=10, p=2, weights=my_weight)
clf3.fit(x_train, y_train)
y_pred3 = clf3.predict(x_test)

print(f'Accuracy of 10NN Custom Weight {accuracy_score(y_test, y_pred3)}')








