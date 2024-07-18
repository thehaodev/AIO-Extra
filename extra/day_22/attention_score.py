import numpy as np


def softmax_function(x):
    exps = np.exp(x)
    return exps/np.sum(exps)


def run():
    generator = np.random.default_rng(42)
    vocab = {
        "Tôi": 0,
        "thích": 1,
        "học": 2,
        "AI": 3
    }
    vocab_size = len(vocab)
    embedding_dim = 4
    embedding_matrix = generator.standard_normal(size=(vocab_size, embedding_dim))
    input_seq = np.array([embedding_matrix[vocab[word]] for word in ["Tôi", "thích", "học", "AI"]])
    print("Chuỗi đầu vào (đã mã hóa):\n", input_seq)

    w_q = generator.standard_normal(size=(embedding_dim, embedding_dim))
    w_k = generator.standard_normal(size=(embedding_dim, embedding_dim))
    w_v = generator.standard_normal(size=(embedding_dim, embedding_dim))
    q = np.dot(input_seq, w_q)
    k = np.dot(input_seq, w_k)
    v = np.dot(input_seq, w_v)

    print("Ma trận Query Q:\n", q)
    print("Ma trận Key K:\n", k)
    print("Ma trận Value V:\n", v)

    scores = np.dot(q, k.transpose())
    d_k = embedding_dim
    scores = scores / np.sqrt(d_k)
    print("Điểm số:\n", scores)

    attention_weights = softmax_function(scores)
    print("Trọng số Attention :\n", attention_weights)

    output = np.multiply(attention_weights, v)
    print("Đầu ra :\n", output)


run()
