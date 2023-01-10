#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/10/15 12:02
# @Author  : ZWP
# @Desc    :
# @File    : main.py
import time

import numpy as np
import matplotlib.pyplot as plt

from detector.HDDDM import HDDDM
from detector.PCACD import PcaCD
from detector.SDDM import SDDM
from detector.DAWIDD import DAWIDD as SWIDD
from utils.datasets import create_comfort_level_fake_drift_dataset
from utils.drift_detector import *
from utils.experiments import evaluate, nemenyi_test


def test_dataset(all_dataset, methods):
    global run_record
    run_record += 1
    print("第" + str(run_record) + "次迭代")
    # 各个数据集的运行结果
    results = dict()
    for dataset in all_dataset:
        results[dataset] = dict()

    # 数据集数据，使用的实验工具包的数据集生成函数，内容一概是（名称，（x，y））的元组
    dataset_build = [
        ("ideal_level", create_comfort_level_fake_drift_dataset()),
    ]
    datasets = []
    for dataset in dataset_build:
        if dataset[0] in all_dataset:
            datasets.append(dataset)
    # Test all data sets
    # r_all_datasets = Parallel(n_jobs=4)(delayed(test_on_data_set)(data_desc, D, methods) for data_desc, D in datasets)
    r_all_datasets = [test_on_data_set(data_desc, D, methods) for data_desc, D in datasets]
    for r_data in r_all_datasets:
        for k in r_data.keys():
            results[k] = r_data[k]

    return results


# 对所有方法进行测试的地方
def test_on_data_set(data_desc, D, methods):
    r = {data_desc: dict()}
    for method in methods:
        r.get(data_desc)[method] = []

    training_buffer_size = 100  # Size of training buffer of the drift detector
    n_train = (int)(0.2 * D["data"][0].shape[0])  # Initial training set size
    print('n_tyrain:', n_train)

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
        dd = DriftDetectorUnsupervised(SDDM(X0, Y0, batch_size, 0, alpha_ks), batch_size=batch_size)
        # changes_detected, time_elapsed = dd.apply_to_stream(X)
        # fake drift test
        changes_detected, time_elapsed = dd.apply_to_stream(X_next)

        # Evaluation
        scores = evaluate(concept_drifts, changes_detected, time_elapsed, tol)
        r[data_desc]["SDDM"].append(scores)

    if "HDDDM" in r[data_desc].keys():
        dd = DriftDetectorUnsupervised(HDDDM(data0, gamma=None, alpha=0.005), batch_size=batch_size)
        # changes_detected, time_elapsed = dd.apply_to_stream(data_stream)
        # fake drift test
        changes_detected, time_elapsed = dd.apply_to_stream(data_next)

        # Evaluation
        scores = evaluate(concept_drifts, changes_detected, time_elapsed, tol)
        r[data_desc]["HDDDM"].append(scores)
    if "SWIDD" in r[data_desc].keys():
        dd = DriftDetectorUnsupervised(SWIDD(max_window_size=30, min_window_size=10), batch_size=1)
        # changes_detected, time_elapsed = dd.apply_to_stream(data_stream)
        # fake drift test
        changes_detected, time_elapsed = dd.apply_to_stream(data_next)

        # Evaluation
        scores = evaluate(concept_drifts, changes_detected, time_elapsed, tol)
        r[data_desc]["SWIDD"].append(scores)
    if "PCACD" in r[data_desc].keys():
        detector = PcaCD(window_size=batch_size, divergence_metric="intersection")
        dd = DriftDetectorUnsupervised(drift_detector=detector, batch_size=1)
        # changes_detected, time_elapsed = dd.apply_to_stream(X)
        # fake drift test
        changes_detected, time_elapsed = dd.apply_to_stream(X_next)

        # Evaluation
        scores = evaluate(concept_drifts, changes_detected, time_elapsed, tol)
        r[data_desc]["PCACD"].append(scores)

    return r


def evaluate_all_methods(all_results, indicators, n_itr):
    results = all_results[0]
    # 将所有的结果加到一起,暂时保留原结构，其中有一个数组没有用但是不改了

    # 这里计算的消耗时间是将多次迭代的时间加到了一起,都加到第一个数组上了
    for result_number in range(1, len(all_results)):
        for dataset in results.keys():
            for method in results[dataset].keys():
                results[dataset][method][0]['false_positives'] += all_results[result_number][dataset][method][0][
                    'false_positives']

    # 统计出召回率等信息，并将无用的数组删除
    results_stat = dict()
    for dataset_name, dataset in results.items():
        results_stat[dataset_name] = dict()
        for method_name, result in dataset.items():
            results_stat[dataset_name][method_name] = dict()
            results_stat[dataset_name][method_name]['false_positives'] = result[0]['false_positives'] / n_itr
    # 方法的个数方便画图
    dataset_num = len(results.keys())

    fig, ax = plt.subplots(dataset_num, len(indicators),
                           figsize=(5 * len(indicators), 5 * dataset_num)
                           )
    # 对每个数据集做循环每个数据集的图占据一行
    for dataset_index, (dataset_name, dataset) in enumerate(results_stat.items()):
        # 调用函数绘制一行指标
        for indicator_index, indicator in enumerate(indicators):
            # 绘制准确率曲线
            indicator_value = []
            method_names = []
            # 行数为一的情况下是一维的
            if dataset_num == 1 and len(indicators) == 1:
                subPlot = ax
            elif dataset_num == 1:
                subPlot = ax[indicator_index]
            else:
                subPlot = ax[dataset_index, indicator_index]
            subPlot.set_title(dataset_name + "_" + indicator)
            for method_name, result in dataset.items():
                method_names.append(method_name)
                indicator_value.append(result[indicator])
            subPlot.bar(range(len(indicator_value)), indicator_value)
            subPlot.set_xticks(range(len(method_names)), method_names)
            subPlot.set_xlabel("methods")
            subPlot.set_ylabel(indicator)

    # plt.suptitle("tol:" + str(tol)
    #              + " patience:" + str(patience)
    #              + " n_itr:" + str(n_itr)
    #              + " batch_size:" + str(batch_size)
    #              + " alpha_ks:" + str(alpha_ks)
    #              + " alpha_tran:" + str(alpha_tran), fontsize=30)
    plt.savefig("result/result_" + str(time.time()) + ".png")
    plt.show()
    print(results_stat)
    nemenyi_test(indicators, results_stat, reverses=[False, True],
                 n_itr=n_itr)
    return results


if __name__ == '__main__':
    tol = 90
    patience = 0
    alpha_tran = 0.02
    alpha_middle = 0.02
    alpha_ks = 0.008
    n_itr = 20

    # fake drift test
    batch_size = 30
    run_record = 0
    all_datasets = [
        "ideal_level",
    ]
    methods = [
        "SDDM",
        "HDDDM",
        "PCACD",
        "SWIDD"
    ]
    indicators = [
        'false_positives',
    ]

    all_results = [test_dataset(all_datasets, methods) for _ in range(n_itr)]
    evaluate_all_methods(all_results, indicators, n_itr)
