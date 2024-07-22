import numpy as np


def create_position_matrix(seq_len, embed_size):
    pos_matrix = np.zeros((seq_len, embed_size))
    for k in range(seq_length):
        for j in np.arange(int(embed_size / 2)):
            rad = np.power(10000, 2*j / embed_size)
            pos_matrix[k, 2*j] = np.sin(k / rad)
            pos_matrix[k, 2*j + 1] = np.cos(k / rad)

    return pos_matrix


seq_length = 10
embed_size = 16
position_matrix = create_position_matrix(seq_length, embed_size)
print(position_matrix)
