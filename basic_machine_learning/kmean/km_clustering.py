import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

generator = np.random.default_rng(11)

# Create mean and point around them
means = np.asarray([[2, 2], [8, 3], [3, 6]])
cov = np.asarray([[1, 0], [0, 1]])
N = 500
X0 = generator.multivariate_normal(means[0], cov, N)
X1 = generator.multivariate_normal(means[1], cov, N)
X2 = generator.multivariate_normal(means[2], cov, N)

X = np.concatenate((X0, X1, X2), axis=0)
K = 3

original_label = np.asarray([0] * N + [1] * N + [2] * N).T


def kmeans_display(label):
    x0_label = X[label == 0, :]
    x1_label = X[label == 1, :]
    x2_label = X[label == 2, :]

    plt.plot(x0_label[:, 0], x0_label[:, 1], 'b^', markersize=4, alpha=.8)
    plt.plot(x1_label[:, 0], x1_label[:, 1], 'go', markersize=4, alpha=.8)
    plt.plot(x2_label[:, 0], x2_label[:, 1], 'rs', markersize=4, alpha=.8)

    plt.axis('equal')
    plt.plot()
    plt.show()


def kmeans_init_centers(k):
    # randomly pick k rows of X as initial centers
    return X[generator.choice(X.shape[0], k, replace=False)]


def kmeans_assign_labels(centers):
    # calculate pairwise distances btw data and centers
    D = cdist(X, centers)
    # return index of the closest center
    return np.argmin(D, axis=1)


def kmeans_update_centers(labels):
    centers = np.zeros((K, X.shape[1]))
    for k in range(K):
        # collect all points assigned to the k-th cluster
        xk = X[labels == k, :]
        # take average
        centers[k, :] = np.mean(xk, axis=0)
    return centers


def has_converged(centers, new_centers):
    # return True if two sets of centers are the same
    return (set([tuple(a) for a in centers]) ==
            set([tuple(a) for a in new_centers]))


def kmeans():
    centers = [kmeans_init_centers(K)]
    labels = []
    it = 0
    while True:
        labels.append(kmeans_assign_labels(centers[-1]))
        new_centers = kmeans_update_centers(labels[-1])
        if has_converged(centers[-1], new_centers):
            break
        centers.append(new_centers)
        it += 1
    return centers, labels, it


def run():
    (centers, labels, it) = kmeans()
    print('Centers found by our algorithm:')
    print(centers[-1])

    kmeans_display(labels[-1])


def run_sklearn():
    kmeans_sklearn = KMeans(n_clusters=3, random_state=0).fit(X)
    print('Centers found by scikit-learn:')
    print(kmeans_sklearn.cluster_centers_)
    pred_label = kmeans_sklearn.predict(X)
    kmeans_display(pred_label)


run_sklearn()
