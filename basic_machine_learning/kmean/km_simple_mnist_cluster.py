import numpy as np
from mnist import MNIST
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from basic_machine_learning.display_network import display_network
import scipy.misc


def run():
    # Load MNIST data
    mndata = MNIST('../data/MNIST')
    mndata.load_testing()
    X = mndata.test_images
    X0 = np.asarray(X)[:1000, :] / 256.0
    X = X0

    K = 10
    kmeans = KMeans(n_clusters=K, random_state=42)
    kmeans.fit(X)
    pred_label = kmeans.predict(X)

    print(type(kmeans.cluster_centers_.T))
    print(kmeans.cluster_centers_.T.shape)
    A = display_network(kmeans.cluster_centers_.T, K, 1)

    f1 = plt.imshow(A, interpolation='nearest', cmap="jet")
    f1.axes.get_xaxis().set_visible(False)
    f1.axes.get_yaxis().set_visible(False)
    plt.show()
    # plt.savefig('a1.png', bbox_inches='tight')

    # a colormap and a normalization instance
    cmap = plt.get_cmap('jet')
    norm = plt.Normalize(vmin=A.min(), vmax=A.max())

    # map the normalized data to colors
    # image is now RGBA (512x512x4)
    image = cmap(norm(A))
    scipy.misc.imsave('aa.png', image)

    print(type(pred_label))
    print(pred_label.shape)
    print(type(X0))


run()
