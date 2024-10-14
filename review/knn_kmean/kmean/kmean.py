import numpy as np
from sklearn.cluster import KMeans


def example_km(data_input, k):

    # 1. Randomly select centroids
    centroids = data_input[:k]

    # Number of loop to find centroids
    limit_assign = 10

    for _ in range(limit_assign):
        # 2. Compute distance
        distances = np.sqrt((data_input.reshape(-1, 1)))

        # 3. Compute labels
        labels = np.argmin(distances, axis=1)

        # 4. Update centroids
        centroids_news = np.array([data_input[labels == i].mean() for i in range(k)])

        # 5. Check stop point
        if np.all(centroids_news == centroids):
            break
        centroids = centroids_news


