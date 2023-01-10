import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Orange.evaluation.scoring import compute_CD, graph_ranks

# Evaluation
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import precision_score

# Classifier
from sklearn.svm import SVC


class Classifier():
    def __init__(self, model=SVC(C=1.0, kernel='linear')):
        self.model = model
        self.flip_score = False

    def fit(self, X, y):
        self.model.fit(X, y.ravel())

    def score(self, x, y):
        s = int(self.model.predict([x]) == y)
        if self.flip_score == True:
            return 1 - s
        else:
            return s

    def score_set(self, X, y):
        return self.model.score(X, y.ravel())


def evaluate(true_concept_drifts, pred_concept_drifts, time_elapsed, tol=200):
    false_positives = 0
    drift_detected = 0
    drift_not_detected = 0
    delays = []

    # Check for false alarms
    for t in pred_concept_drifts:
        b = False
        for dt in true_concept_drifts:
            if dt <= t and t <= dt + tol:
                b = True
                break
        if b is False:  # False alarm
            false_positives += 1

    # Check for detected and undetected drifts
    for dt in true_concept_drifts:
        b = False
        for t in pred_concept_drifts:
            if dt <= t and t <= dt + tol:
                b = True
                drift_detected += 1
                delays.append(t - dt)
                break
        if b is False:
            drift_not_detected += 1

    return {"false_positives": false_positives, "drift_detected": drift_detected,
            "drift_not_detected": drift_not_detected,
            "delays": delays, "time_elapsed": time_elapsed}


def evaluate_rw(data_desc, method, D, pred_concept_drifts, time_elapsed, batch_size, train_size=0.2,
                train_buffer_size=100):
    # 构造分类器
    model = AdaBoostClassifier()

    # 数据准备
    n_train = (int)(0.2 * D["data"][0].shape[0])
    X, Y = D["data"]
    X0, Y0 = X[0:n_train, :], Y[0:n_train, :]  # Training dataset
    model.fit(X0, Y0.ravel())

    collect_samples = False
    rw_scores = []
    data_indexs = []
    n_x_samples = len(X)
    X_training_buffer = []
    Y_training_buffer = []

    t = 0
    update_num = 0
    while t < n_x_samples:
        end_idx = t + batch_size
        if end_idx >= n_x_samples:
            end_idx = n_x_samples
        x_batch = X[t:end_idx, :]
        y_batch = Y[t:end_idx, :]
        if not collect_samples:

            y_pred = model.predict(x_batch)
            rw_scores.append(precision_score(y_batch, y_pred, average='weighted'))
            data_indexs.append(t)
            for pred_concept_drift in pred_concept_drifts:
                if t <= pred_concept_drift < end_idx:
                    collect_samples = True
                    X_training_buffer = []
                    Y_training_buffer = []
        else:
            y_pred = model.predict(x_batch)
            rw_scores.append(precision_score(y_batch, y_pred, average='weighted'))
            data_indexs.append(t)
            X_training_buffer.append(x_batch)
            Y_training_buffer.append(y_batch)
            if len(X_training_buffer) * len(x_batch) > train_buffer_size:
                X_training = np.concatenate(X_training_buffer, axis=0)
                Y_training = np.concatenate(Y_training_buffer, axis=0)
                model.fit(X_training, Y_training)
                update_num += 1
                collect_samples = False
        t += batch_size

    # 画出结果
    plt.xlabel('index')
    plt.ylabel('accuracy/p-value')
    plt.title(data_desc + "___" + method)
    plt.vlines(x=pred_concept_drifts, ymin=0.0, ymax=1, colors='r', linestyles='-',
               label='drift')
    plt.plot(data_indexs, rw_scores, lw=2, label='accuracy')
    plt.title(data_desc + "___" + method + "__" + str(np.average(rw_scores)))
    plt.savefig('./result/real_word/' + data_desc + "___" + method + "__" + str(update_num) + "__" + str(
        np.average(rw_scores)) + '.png')
    plt.show()
    # return {"data_indexs": data_indexs, "scores": rw_scores, "update_num": update_num,"acc_avg":np.mean(rw_scores)}
    return {"update_num": update_num, "acc_avg": np.mean(rw_scores), "drift_detect": len(pred_concept_drifts)}


def nemenyi_test(indicators, results, reverses, n_itr=1):
    j = 0
    # fig, ax = plt.subplots(len(indicators), 1)

    for indicator in indicators:
        index = []
        column = []
        data = []

        i = 0
        for dataset in results.keys():
            index.append(dataset)
            row = []
            for method in results[dataset].keys():
                if i == 0:
                    column.append(method)
                row.append(results[dataset][method][indicator])
            i += 1
            data.append(row)
        table = pd.DataFrame(data=data, index=index, columns=column)
        table = pd.DataFrame(table.values.T, index=table.columns, columns=table.index)
        # print(table)
        # print(table.shape[1])
        table_rank = table.copy()
        for col in table.columns:
            table = table.sort_values(by=col, ascending=reverses[j])
            flag = 0
            temp = 0
            for index in table.index:
                value = table[col][index]
                df = table.loc[table[col] == value]
                if df.shape[0] == 1:
                    table_rank[col][index] = list(table.index).index(index) + 1
                elif df.shape[0] > 1 and flag == 0:
                    a1 = list(table.index).index(index) + 1
                    n = df.shape[0]
                    an = a1 + n - 1
                    temp = table_rank[col][index] = (n * (a1 + an) / 2) / n
                    flag = n - 1
                else:
                    table_rank[col][index] = temp
                    flag -= 1
        print(table)
        print(table_rank)
        mean = table_rank.mean(axis=1)
        print(mean)
        names = list(mean.index)
        avranks = list(mean.values)
        datasets_num = len(results.keys()) * n_itr
        CD = compute_CD(avranks, datasets_num, alpha='0.05', test='nemenyi')
        # subPlot = ax[i]
        graph_ranks(avranks, names, cd=CD, width=8, textspace=1.5, reverse=not reverses[j])
        j += 1
        plt.savefig('./result/nemenyi/' + str(time.time()) + indicator + '.png')
        plt.show()
