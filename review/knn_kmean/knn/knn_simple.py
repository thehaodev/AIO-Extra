import math


# Compute distance between two point
def compute_distance(point1, point2):
    # Get dimension of point
    ndim = point1.ndim
    result = 0

    for i in range(ndim):
        result += (float(point1[i]) - float(point2[i])) ** 2

    return math.sqrt(result)


# Find nearst neighbor
def compute_k_nearest_neighbor(training_set, item, k):
    distances = []
    for data_point in training_set:
        distances.append(
            {
                "label": data_point[-1],
                "value": compute_distance(item, data_point)
            }
        )
    distances.sort(key=lambda x: x["value"])
    labels = [item["label"] for item in distances]

    return labels[:k]


# In regression return np.mean(array, dtype=np.float64)
def vote_the_distances(array):
    labels = set(array)
    result = ""
    max_occur = 0
    for label in labels:
        num = array.count(label)
        if num > max_occur:
            max_occur = num
            result = label

    return result


# Used when the number of sample equal in array. Return label of closet point
def vote_the_distances_weight(array):
    return array[0]
