def accuracy(tp, tn, fp, fn):
    return (tp+tn) / (tp+tn+fp+fn)


def f1_score(p, r):
    return 2 * (p * r) / (p + r)


def precision(tp, fp):
    return tp / (tp+fp)


def recall(tp, fn):
    return tp / (tp+fn)


def specificity(tn, fp):
    return tn / (tn + fp)


def run():
    # input img = 1000, img_cat= 400 predict_result = 100 (cat pic - real 98)
    tp = 98
    fp = 2
    tn = 598
    fn = 302
    precs = precision(tp, fp)
    r_call = recall(tp, fn)
    accur = accuracy(tp, tn, fp, fn)

    print(accur)
    print(precs)
    print(r_call)
    print(f1_score(precs, r_call))


run()
