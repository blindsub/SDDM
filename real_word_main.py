#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/10/25 11:54
# @Author  : ZWP
# @Desc    : 
# @File    : real_word_main.py

from detector.DAWIDD import DAWIDD as SWIDD
from detector.HDDDM import HDDDM
from detector.PCACD import *
from detector.SDDM import *
from utils.datasets import *
from utils.drift_detector import *
from utils.experiments import evaluate, evaluate_rw


def test_rw_dataset(all_dataset, methods):
    global run_record
    run_record += 1
    print("第" + str(run_record) + "次迭代")
    # 各个数据集的运行结果
    results = dict()
    for dataset in all_dataset:
        results[dataset] = dict()

    # 数据集数据，使用的实验工具包的数据集生成函数，内容一概是（名称，（x，y））的元组
    dataset_build = [
        ("Weather", create_weather_drift_dataset(n_concept_drifts=4)),
        ("ForestCover", create_forest_cover_drift_dataset(n_concept_drifts=4)),
        ("ElectricityMarket", create_electricity_market_drift_dataset(n_concept_drifts=4)),
        ("Phishing", create_phishing_drift_dataset()),
    ]
    datasets = []
    for dataset in dataset_build:
        if dataset[0] in all_dataset:
            datasets.append(dataset)
    # Test all data sets
    # r_all_datasets = Parallel(n_jobs=2)(delayed(test_on_rw_data_set)(data_desc, D, methods) for data_desc, D in datasets)
    r_all_datasets = [test_on_rw_data_set(data_desc, D, methods) for data_desc, D in datasets]
    for r_data in r_all_datasets:
        for k in r_data.keys():
            results[k] = r_data[k]

    return results


# 对所有方法进行测试的地方
def test_on_rw_data_set(data_desc, D, methods):
    r = {data_desc: dict()}
    for method in methods:
        r.get(data_desc)[method] = []

    training_buffer_size = 100  # Size of training buffer of the drift detector
    n_train = (int)(0.2 * D["data"][0].shape[0])  # Initial training set size

    concept_drifts = D["drifts"]
    X, Y = D["data"]
    data_stream = np.concatenate((X, Y.reshape(-1, 1)), axis=1)

    X0, Y0 = X[0:n_train, :], Y[0:n_train, :]  # Training dataset
    data0 = data_stream[0:n_train, :]

    # 为什么漂移数据点没有减去训练数据？？？？
    # 太傻逼了，设计漂移点位置时候有训练数据，检测点计数的时候他妈的没了，真绝了
    X_next, Y_next = X[n_train:, :], Y[n_train:, :]  # Test set
    data_next = data_stream[n_train:, :]
    # Run unsupervised drift detector

    if "SDDM" in r[data_desc].keys():
        dd = DriftDetectorUnsupervised(SDDM(X0, Y0, 200, 0, alpha_ks), batch_size=batch_size)
        changes_detected, time_elapsed = dd.apply_to_stream(X)

        # Evaluation
        scores = evaluate_rw(data_desc, "SDDM", D, changes_detected, time_elapsed, batch_size=batch_size)
        r[data_desc]["SDDM"].append(scores)

    if "HDDDM" in r[data_desc].keys():
        dd = DriftDetectorUnsupervised(HDDDM(data0, gamma=None, alpha=0.005), batch_size=batch_size)
        changes_detected, time_elapsed = dd.apply_to_stream(data_stream)

        # Evaluation
        scores = evaluate_rw(data_desc, "HDDDM", D, changes_detected, time_elapsed, batch_size=batch_size)
        r[data_desc]["HDDDM"].append(scores)
    if "SWIDD" in r[data_desc].keys():
        dd = DriftDetectorUnsupervised(SWIDD(max_window_size=300, min_window_size=100), batch_size=1)
        changes_detected, time_elapsed = dd.apply_to_stream(data_stream)

        # Evaluation
        scores = evaluate_rw(data_desc, "SWIDD", D, changes_detected, time_elapsed, batch_size=batch_size)
        r[data_desc]["SWIDD"].append(scores)

    if "PCACD" in r[data_desc].keys():
        detector = PcaCD(window_size=50, divergence_metric="intersection")
        dd = DriftDetectorUnsupervised(drift_detector=detector, batch_size=1)
        # changes_detected, time_elapsed = dd.apply_to_stream(X)
        # fake drift test
        changes_detected, time_elapsed = dd.apply_to_stream(X_next)

        # Evaluation

        scores = evaluate_rw(data_desc, "PCACD", D, changes_detected, time_elapsed, batch_size=batch_size)
        r[data_desc]["PCACD"].append(scores)

    return r


if __name__ == '__main__':
    tol = 90
    patience = 0
    alpha_tran = 0.05
    alpha_ks = 0.01
    n_itr = 1
    batch_size = 50
    run_record = 0
    all_datasets = [
        "Weather",
        "ForestCover",
        "ElectricityMarket",
        "Phishing"
    ]
    methods = [
        "SDDM",
        "HDDDM",
        "SWIDD",
        "PCACD"
    ]
    indicators = [
        # 'precision',
        # 'recall',
        # 'drift_detected',
        'false_positives',
        'drift_not_detected'
    ]
    all_results = [test_rw_dataset(all_datasets, methods) for _ in range(n_itr)]
    print(all_results)
