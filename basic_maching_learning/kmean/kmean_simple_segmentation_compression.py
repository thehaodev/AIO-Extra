import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans


def run():
    # REad image
    # imgplot = plt.imshow(img)
    img = mpimg.imread('girl3.jpg')
    plt.imshow(img)

    plt.axis('off')
    plt.show()

    X = img.reshape((img.shape[0] * img.shape[1], img.shape[2]))
    for K in [2, 5, 10, 15, 20]:
        kmeans = KMeans(n_clusters=K, random_state=42)

        label = kmeans.predict(X)

        img4 = np.zeros_like(X)
        # replace each pixel by its center
        for k in range(K):
            img4[label == k] = kmeans.cluster_centers_[k]
        # reshape and display output image
        img5 = img4.reshape((img.shape[0], img.shape[1], img.shape[2]))
        plt.imshow(img5, interpolation='nearest')
        plt.axis('off')
        plt.show()

    for K in [3]:
        kmeans = KMeans(n_clusters=K, random_state=42)
        kmeans.fit(X)
        label = kmeans.predict(X)

        img4 = np.zeros_like(X)
        # replace each pixel by its center
        for k in range(K):
            img4[label == k] = kmeans.cluster_centers_[k]
        # reshape and display output image
        img5 = img4.reshape((img.shape[0], img.shape[1], img.shape[2]))
        plt.imshow(img5, interpolation='nearest')
        plt.axis('off')
        plt.show()
