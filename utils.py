import numpy as np
import time


def rank_based_evaluate(y_true, y_pred, k):
    if not isinstance(k, list):
        ks = [k]
    else:
        ks = k

    precisions = [list() for _ in range(len(ks))]
    recalls = [list() for _ in range(len(ks))]
    apks = list()

    for i in range(len(y_true)):
        pred = np.array(y_pred[i]).argsort()
        for j, _k in enumerate(ks):
            prec, rec = _compute_precision_recall(y_true[i], pred, _k)
            precisions[j].append(prec)
            recalls[j].append(rec)
        apks.append(_compute_apk(y_true[i], pred, np.inf))

    return np.mean(precisions, axis=-1), np.mean(recalls, axis=-1), np.mean(apks, axis=-1)

def _compute_precision_recall(targets, predictions, k):

    pred = predictions[:k]
    num_hit = len(set(pred).intersection(set(targets)))
    precision = float(num_hit) / len(pred)
    recall = float(num_hit) / len(targets)
    return precision, recall

def _compute_apk(targets, predictions, k):

    if len(predictions) > k:
        predictions = predictions[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predictions):
        if p in targets and p not in predictions[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not list(targets):
        return 0.0

    return score / min(len(targets), k)

def one_hot(data, k):
    data = np.reshape(data, [-1])
    return np.eye(len(data), k)[data]

def timestampToWallClock(timestamp):
    #2009-05-04T23:08:57Z
    temptime = time.localtime(timestamp)
    return int (temptime.tm_hour / 2)

def timestampToWallClock_2(timestamp):
    temptime = time.localtime(timestamp)
    wday = temptime.tm_wday
    wday = 0 if wday < 5 else 1
    return wday
