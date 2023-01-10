#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/10/15 12:02
# @Author  : ZWP
# @Desc    : 
# @File    : main.py
from matplotlib import pyplot as plt
from sklearn.naive_bayes import GaussianNB
from skmultiflow.drift_detection import ADWIN, EDDM, DDM

from detector.PCACD import *
from detector.DAWIDD import DAWIDD as SWIDD
from detector.HDDDM import HDDDM
from detector.SDDM import *
from utils.datasets import *
from utils.drift_detector import *
from utils.experiments import nemenyi_test, evaluate, Classifier


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
        ("RandomRBF", create_rbf_drift_dataset(n_samples_per_concept=500, n_concept_drifts=4)),
        ("UG_2C_2D", create_UG_2C_2D_dataset(n_max_length=5000)),
        ("MG_2C_2D", create_MG_2C_2D_dataset(n_max_length=5000)),
        ("2CDT", create_2CDT_dataset(n_max_length=2000)),
        ("2CHT", create_2CHT_dataset(n_max_length=2000)),
        ("4CR", create_4CR_dataset(n_max_length=2000)),
        ("4CRE-V1", create_4CRE_V1_dataset(5000)),
        ("5CVT", create_5CVT_dataset(n_max_length=5000)),
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

    concept_drifts = D["drifts"]
    X, Y = D["data"]
    data_stream = np.concatenate((X, Y.reshape(-1, 1)), axis=1)

    X0, Y0 = X[0:n_train, :], Y[0:n_train, :]  # Training dataset
    data0 = data_stream[0:n_train, :]

    X_next, Y_next = X[n_train:, :], Y[n_train:, :]  # Test set
    data_next = data_stream[n_train:, :]
    # Run unsupervised drift detector

    if "SDDM" in r[data_desc].keys():
        dd = DriftDetectorUnsupervised(SDDM(X0, Y0, 50, 0, alpha_ks), batch_size=batch_size)
        changes_detected, time_elapsed = dd.apply_to_stream(X)

        # Evaluation
        scores = evaluate(concept_drifts, changes_detected, time_elapsed, tol)
        r[data_desc]["SDDM"].append(scores)

    if "HDDDM" in r[data_desc].keys():
        dd = DriftDetectorUnsupervised(HDDDM(data0, gamma=None, alpha=0.005), batch_size=batch_size)
        changes_detected, time_elapsed = dd.apply_to_stream(data_stream)

        # Evaluation
        scores = evaluate(concept_drifts, changes_detected, time_elapsed, tol)
        r[data_desc]["HDDDM"].append(scores)
    if "SWIDD" in r[data_desc].keys():
        dd = DriftDetectorUnsupervised(SWIDD(max_window_size=300, min_window_size=100), batch_size=1)
        changes_detected, time_elapsed = dd.apply_to_stream(data_stream)

        # Evaluation
        scores = evaluate(concept_drifts, changes_detected, time_elapsed, tol)
        r[data_desc]["SWIDD"].append(scores)

    if "PCACD" in r[data_desc].keys():
        detector = PcaCD(window_size=50, divergence_metric="intersection")
        dd = DriftDetectorUnsupervised(drift_detector=detector, batch_size=1)
        changes_detected, time_elapsed = dd.apply_to_stream(data_stream)

        # Evaluation
        scores = evaluate(concept_drifts, changes_detected, time_elapsed, tol)
        r[data_desc]["PCACD"].append(scores)

    # Run supervised drift detector
    model = GaussianNB()

    # EDDM
    drift_detector = EDDM()

    clf = Classifier(model)
    clf.flip_score = True
    clf.fit(X0, Y0.ravel())
    if "EDDM" in r[data_desc].keys():
        dd = DriftDetectorSupervised(clf=clf, drift_detector=drift_detector, training_buffer_size=training_buffer_size)
        changes_detected, time_elapsed = dd.apply_to_stream(X_next, Y_next)

        # Evaluation
        scores = evaluate(concept_drifts, changes_detected, time_elapsed, tol)
        r[data_desc]["EDDM"].append(scores)

    if "DDM" in r[data_desc].keys():
        drift_detector = DDM(min_num_instances=30, warning_level=2.0, out_control_level=3.0)

        clf = Classifier(model)
        clf.flip_score = True
        clf.fit(X0, Y0.ravel())

        dd = DriftDetectorSupervised(clf=clf, drift_detector=drift_detector, training_buffer_size=training_buffer_size)
        changes_detected, time_elapsed = dd.apply_to_stream(X_next, Y_next)

        # Evaluation
        scores = evaluate(concept_drifts, changes_detected, time_elapsed, tol)
        r[data_desc]["DDM"].append(scores)

    if "ADWIN" in r[data_desc].keys():
        drift_detector = ADWIN(delta=2.)

        clf = Classifier(model)
        clf.fit(X0, Y0.ravel())

        dd = DriftDetectorSupervised(clf=clf, drift_detector=drift_detector, training_buffer_size=training_buffer_size)
        changes_detected, time_elapsed = dd.apply_to_stream(X_next, Y_next)

        # Evaluation
        scores = evaluate(concept_drifts, changes_detected, time_elapsed, tol)
        r[data_desc]["ADWIN"].append(scores)
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
                results[dataset][method][0]['drift_detected'] += all_results[result_number][dataset][method][0][
                    'drift_detected']
                results[dataset][method][0]['drift_not_detected'] += all_results[result_number][dataset][method][0][
                    'drift_not_detected']
                results[dataset][method][0]['time_elapsed'] += all_results[result_number][dataset][method][0][
                    'time_elapsed']
                # results[dataset][method][0]['delays'] += all_results[result_number][dataset][method][0]['delays']

    # 统计出召回率等信息，并将无用的数组删除
    results_stat = dict()
    for dataset_name, dataset in results.items():
        results_stat[dataset_name] = dict()
        for method_name, result in dataset.items():
            results_stat[dataset_name][method_name] = dict()
            results_stat[dataset_name][method_name]['false_positives'] = result[0]['false_positives'] / n_itr
            results_stat[dataset_name][method_name]['drift_detected'] = result[0]['drift_detected'] / n_itr
            results_stat[dataset_name][method_name]['drift_not_detected'] = result[0]['drift_not_detected'] / n_itr
            # results_stat[dataset_name][method_name]['delays'] = result[0]['delays']
            results_stat[dataset_name][method_name]['avg_time_elapsed'] = result[0]['time_elapsed'] / len(all_results)
            if result[0]['drift_detected'] + result[0]['drift_not_detected'] == 0:
                results_stat[dataset_name][method_name]['recall'] = 0
            else:
                results_stat[dataset_name][method_name]['recall'] = result[0]['drift_detected'] / (
                        result[0]['drift_detected'] + result[0]['drift_not_detected'])

            if result[0]['drift_detected'] + result[0]['false_positives'] == 0:
                results_stat[dataset_name][method_name]['precision'] = 0
            else:
                results_stat[dataset_name][method_name]['precision'] = result[0]['drift_detected'] / (
                        result[0]['drift_detected'] + result[0]['false_positives'])
    # 方法的个数方便画图
    dataset_num = len(results.keys())
    fig, ax = plt.subplots(dataset_num, len(indicators),
                           figsize=(6 * len(indicators), 5 * dataset_num)
                           )
    # 对每个数据集做循环每个数据集的图占据一行
    for dataset_index, (dataset_name, dataset) in enumerate(results_stat.items()):
        # 调用函数绘制一行指标
        for indicator_index, indicator in enumerate(indicators):
            # 绘制准确率曲线
            indicator_value = []
            method_names = []
            # 行数为一的情况下是一维的
            if dataset_num == 1:
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
    nemenyi_test(indicators, results_stat, reverses=[False, False, True, False, True, True],
                 n_itr=n_itr)
    return results


if __name__ == '__main__':
    tol = 90
    patience = 0
    alpha_tran = 0.02
    alpha_middle = 0.02
    alpha_ks = 0.008
    n_itr = 3
    batch_size = 50
    run_record = 0
    all_datasets = [
        "RandomRBF",
        "2CDT",
        "2CHT",
        "UG_2C_2D",
        "MG_2C_2D",
        "4CRE-V1",
        "5CVT",
        "4CR",
    ]
    methods = [
        "SDDM",
        # "DWDDM",
        "HDDDM",
        # "PCACD",
        # "SWIDD",
        # "EDDM",
        # "DDM",
        # "ADWIN",
    ]
    indicators = [
        'precision',
        'recall',
        'avg_time_elapsed',
        'drift_detected',
        'false_positives',
        # 'drift_not_detected'
    ]

    # all_results = Parallel(n_jobs=-1)(delayed(test_dataset)(all_datasets, methods) for _ in range(n_itr))
    all_results = [test_dataset(all_datasets, methods) for _ in range(n_itr)]
    evaluate_all_methods(all_results, indicators, n_itr)
