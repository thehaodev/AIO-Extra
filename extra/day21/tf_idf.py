import numpy as np
import math


documents = ["Tôi thích học AI",
             "AI là trí tuệ nhân tạo",
             " AGI là siêu trí tuệ nhân tạo"]


def compute_tf(doc: str):
    words = np.array(doc.split())
    numb_of_word = words.size
    _, word_occur = np.unique(words, return_counts=True, return_index=False)

    return word_occur / numb_of_word


def compute_idf(docs: np.array, word):
    numb_of_docs = docs.size
    word_occur = 0
    for doc in docs:
        if word in doc:
            word_occur += 1

    return math.log(numb_of_docs / (1+word_occur))


def compute_tf_idf(tf, idf):
    return tf * idf


def run():

    docs = np.array(documents)

    for doc in docs:
        temp = []
        word_arr = np.array(doc.split())
        for word in word_arr:
            temp.append(compute_idf(docs, word))
        temp_arr = np.array(temp)
        tf_idf_arr = np.multiply(temp_arr, compute_tf(doc))

        dict_tf_idf = {}
        for word, word_tf_idf in zip(word_arr, tf_idf_arr):
            dict_tf_idf[str(word)] = float(word_tf_idf)

        print(dict_tf_idf)


run()
