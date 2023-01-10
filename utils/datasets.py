import math
import random

import numpy as np
import pandas as pd

from skmultiflow.data import STAGGERGenerator, RandomRBFGenerator, SEAGenerator
from sklearn.datasets import make_blobs

from sklearn.utils import shuffle

from utils import load_rw_data


def read_benchmark_data(filename):
    folder_name = "data/benchmark/"
    df = pd.read_csv(folder_name + filename)
    data = df.values
    X, y = data[:, 0:-1], data[:, -1].astype(int)
    return X, y


# 有些人工数据生成器不剧本生成漂移的能力，或者生成的漂移不可能被无监督检测器检测到
def make_drift(dataset_gen, batch_size, true_drift, dataset_num):
    if true_drift:
        pass
    else:
        X, y = dataset_gen.next_sample(batch_size)
        # 随机选取一个特征
        feature_num = random.randint(0, X.shape[1] - 1)

        # 按照某种规则挑选出一堆数据
        random_num = [0.1, 0.2, 0.3]
        # 漂移数据的混合比率
        drift_rate = int(batch_size * (1 - random_num[random.randint(0, len(random_num) - 1)]))
        # 特殊数据的取样比率
        border_rate = random_num[random.randint(0, len(random_num) - 1)]

        border = np.min(X[:, feature_num]) + border_rate * (np.max(X[:, feature_num]) - np.min(X[:, feature_num]))
        some_X = X[X[:, feature_num] < border]
        some_y = y[X[:, feature_num] < border]

        # 少补
        while some_X.shape[0] <= drift_rate:
            new_X, new_y = dataset_gen.next_sample(batch_size)
            some_X = np.concatenate([some_X, new_X[new_X[:, feature_num] < border]], axis=0)
            some_y = np.concatenate([some_y, new_y[new_X[:, feature_num] < border]], axis=0)
        # 多退
        some_X = some_X[:drift_rate, :]
        some_y = some_y[:drift_rate]
        # 重新生成数据
        X, y = dataset_gen.next_sample(batch_size - some_X.shape[0])
        drift_X = np.concatenate([X, some_X], axis=0)
        drift_y = np.concatenate([y, some_y], axis=0)
        # 打乱顺序
        drift_X, drift_y = shuffle(drift_X, drift_y, )
        return drift_X, drift_y


# Random rbf
def create_rbf_drift_dataset(n_samples_per_concept=500, n_concept_drifts=3):
    X_stream = []
    Y_stream = []
    concept_drifts = []

    t = 0
    for _ in range(n_concept_drifts):
        if t != 0:
            concept_drifts.append(t)
        # 每次漂移点创建一个数据生成器，使每次生成的数据不同
        gen = RandomRBFGenerator(n_features=5, n_centroids=10)
        gen.prepare_for_use()
        X, y = gen.next_sample(batch_size=n_samples_per_concept)
        X_stream.append(X)
        Y_stream.append(y)

        t += n_samples_per_concept

    return {"data": (np.concatenate(X_stream, axis=0), np.concatenate(Y_stream, axis=0).reshape(-1, 1)),
            "drifts": np.array(concept_drifts)}


#  correlation coefficient change
def create_rou_change_dataset(epsilon, n_samples_per_concept=500, n_drifts=10):
    mean1 = mean2 = 0.5
    std1 = std2 = 0.2
    interval1 = [epsilon / 2, epsilon + 1e-100]
    interval2 = [-epsilon, -epsilon / 2 + 1e-100]
    concept_drifts = []
    X_stream = []
    Y_stream = []
    for i in range(n_drifts):
        if i != 0:
            concept_drifts.append(i * n_samples_per_concept)

        rou1 = np.random.uniform(interval1[0], interval1[1], 1)
        rou2 = np.random.uniform(interval2[0], interval2[1], 1)
        rou = np.random.choice([rou1[0], rou2[0]])
        cov = std1 * std2 * rou
        cov_matrix = [[std1 * std1, cov], [cov, std2 * std2]]
        data = np.random.multivariate_normal([mean1, mean2], cov_matrix, n_samples_per_concept)
        data_y = []
        for index in range(data.shape[0]):
            if data[index][0] > data[index][1]:
                data_y.append(1)
            else:
                data_y.append(0)
        data_y = np.array(data_y)
        X_stream.append(data)
        Y_stream.append(data_y)
    return {"data": (np.concatenate(X_stream, axis=0), np.concatenate(Y_stream, axis=0).reshape(-1, 1)),
            "drifts": np.array(concept_drifts)}


def create_2CDT_dataset(n_max_length=16000):
    X, y = read_benchmark_data("2CDT.csv")
    drift_size = 400
    return create_benchmark_dataset(X, y, n_max_length, drift_size=drift_size)


def create_2CHT_dataset(n_max_length=16000):
    X, y = read_benchmark_data("2CHT.csv")
    drift_size = 400
    return create_benchmark_dataset(X, y, n_max_length, drift_size=drift_size)


def pre_benchmark2(dataset, n_max_length, drift_size, step):
    concept_drifts = [drift for drift in range(drift_size, n_max_length, drift_size)]

    data = dataset["data"]
    X = data[0]
    y = data[1]
    for i in range(len(concept_drifts)):
        X[i * drift_size:(i + 1) * drift_size] = X[i * step * drift_size:i * step * drift_size + drift_size]
        y[i * drift_size:(i + 1) * drift_size] = y[i * step * drift_size:i * step * drift_size + drift_size]
    X = X[0:n_max_length]
    y = y[0:n_max_length]
    data = (X, y)
    dataset["data"] = data
    dataset["drifts"] = np.array(concept_drifts)
    return dataset


def create_4CR_dataset(n_max_length=144400):
    X, y = read_benchmark_data("4CR.csv")
    drift_size = 400
    step = 25
    dataset = create_benchmark_dataset(X, y, n_max_length * step, drift_size=drift_size)
    return pre_benchmark2(dataset, n_max_length, drift_size, step)


def create_MG_2C_2D_dataset(n_max_length=200000):
    X, y = read_benchmark_data("MG_2C_2D.csv")
    drift_size = 2000
    step = 20
    dataset = create_benchmark_dataset(X, y, n_max_length * step, drift_size=drift_size)
    return pre_benchmark2(dataset, n_max_length, drift_size, step)


def create_UG_2C_2D_dataset(n_max_length=100000):
    X, y = read_benchmark_data("UG_2C_2D.csv")
    drift_size = 1000
    step = 10
    dataset = create_benchmark_dataset(X, y, n_max_length * step, drift_size=drift_size)
    return pre_benchmark2(dataset, n_max_length, drift_size, step)


def create_4CRE_V1_dataset(n_max_length=4000):
    X, y = read_benchmark_data("4CRE-V1.csv")
    drift_size = 1000
    return create_benchmark_dataset(X, y, n_max_length, drift_size=drift_size)


def create_5CVT_dataset(n_max_length=40000):
    X, y = read_benchmark_data("5CVT.csv")
    drift_size = 1000
    step = 2
    dataset = create_benchmark_dataset(X, y, n_max_length * step, drift_size=drift_size)
    return pre_benchmark2(dataset, n_max_length, drift_size, step)


def create_4CE1CF_dataset(n_max_length=173250):
    X, y = read_benchmark_data("4CE1CF.csv")
    drift_size = 750
    step = 35
    dataset = create_benchmark_dataset(X, y, n_max_length * step, drift_size=drift_size)
    return pre_benchmark2(dataset, n_max_length, drift_size, step)


def create_benchmark_dataset(X, y, n_max_length, drift_size):
    concept_drifts = [drift for drift in range(drift_size, n_max_length, drift_size)]
    if X.shape[0] > n_max_length:
        X = X[0:n_max_length, :]
        y = y[0:n_max_length]
    data = (X, y.reshape(-1, 1))
    dataset = {"data": data, "drifts": np.array(concept_drifts)}

    return {"data": data, "drifts": np.array(concept_drifts)}


# Real world data


def create_weather_drift_dataset(n_max_length=1000, n_concept_drifts=3):
    X, y = load_rw_data.read_data_weather()
    return create_controlled_drift_dataset(X, y, n_max_length, n_concept_drifts)


def create_forest_cover_drift_dataset(n_max_length=1000, n_concept_drifts=3):
    X, y = load_rw_data.read_data_forest_cover_type()
    return create_controlled_drift_dataset(X, y, n_max_length, n_concept_drifts)


def create_electricity_market_drift_dataset(n_max_length=1000, n_concept_drifts=3):
    X, y = load_rw_data.read_data_electricity_market()
    return create_controlled_drift_dataset(X, y, n_max_length, n_concept_drifts)


def create_phishing_drift_dataset(n_max_length=1000, n_concept_drifts=3):
    X, y = load_rw_data.read_data_phishing()
    return create_controlled_drift_dataset(X, y, n_max_length, n_concept_drifts)


# 2011训练，在2012上检测
def create_bike_share_drift_dataset(n_concept_drifts=5):
    X = load_rw_data.read_data_bike_share()
    train_size = 0.5 * X.shape[0]
    flag = train_size
    concept_drifts = []
    for i in range(1, n_concept_drifts):
        if i in [1, 3, 5, 7, 8, 10, 12]:
            flag += 31 * 24
        elif i == 2:
            flag += 28 * 24
        else:
            flag += 30 * 24
        concept_drifts.append(flag)
    Y_stream = X['count'].values
    X_stream = X.drop(columns=['count']).values

    return {"data": (X_stream, Y_stream.reshape(-1, 1)), "drifts": []}


def ideal_func(humidity, temp, windspeed):
    ideal_humidity = 45
    ideal_temp = 24
    ideal_windspeed = 10
    ideal = math.sqrt((humidity - ideal_humidity) ** 2 + (temp - ideal_temp) ** 2 + (windspeed - ideal_windspeed) ** 2)
    ideal_min = 0
    ideal_max = 78
    if ideal > (ideal_max - ideal_min) / 4:
        level = 1
    else:
        level = 0
    return level


def data_gen(data_dict, group):
    df_dict = {}
    for col in data_dict.keys():
        data = data_dict[col]
        data_list = []
        for month in data.keys():
            size = 0
            if month in [1, 3, 5, 7, 8, 10, 12]:
                size = 31
            elif month == 2:
                size = 29
            else:
                size = 30
            month_data = np.random.normal(data[month][0], data[month][1], size)
            data_list += list(month_data)
        df_dict[col] = data_list
    data_df = pd.DataFrame(df_dict)
    data_df[['month', 'dayname', 'season']] = group[['month', 'dayname', 'season']]
    for i in range(data_df.shape[0]):
        data_df.loc[i:i + 1, 'ideal'] = ideal_func(data_df.humidity.loc[i], data_df.temp.loc[i],
                                                   data_df.windspeed.loc[i])
    return data_df


def create_comfort_level_fake_drift_dataset():
    full = load_rw_data.read_data_bike_share()
    to_drop_features = ['weather', 'workingday', 'count']
    full.drop(to_drop_features, axis=1, inplace=True)
    data_2012 = full[full.year == 2012]
    group = data_2012.groupby('date').mean()
    group = group.reset_index(drop=True)
    for i in range(group.shape[0]):
        group.loc[i, 'ideal'] = ideal_func(group.humidity.loc[i], group.temp.loc[i], group.windspeed.loc[i])
    data_dict = {'temp': {}, 'humidity': {}, 'windspeed': {}}
    for month in range(1, 13):

        for col in ['temp', 'humidity', 'windspeed']:
            mean = group.loc[group.month == month, col].mean()
            std = group.loc[group.month == month, col].std()
            data_dict[col][month] = (mean, std)
    group = group.drop(['hour', 'year'], axis=1)
    group[['month', 'dayname']] = group[
        ['month', 'dayname']].astype(
        'object')
    group = group.reset_index(drop=True)

    group1 = data_gen(data_dict, group)
    # group3 = data_gen(data_dict, group)
    test_data = data_gen(data_dict, group)
    # test_data2 = data_gen(data_dict, group3)
    group = pd.concat((group, group1, test_data), axis=0)
    group = group.drop(columns=['month', 'dayname', 'season'], axis=1)
    group_X = group.drop(['ideal'], axis=1)
    group_y = group.loc[:, ['ideal']].astype('int8')
    X_stream = group_X.values
    Y_stream = group_y.values
    return {"data": (X_stream, Y_stream.reshape(-1, 1)), "drifts": []}


def create_controlled_drift_dataset(X, y=None, n_max_length=1000, n_concept_drifts=3):
    return {"data": (X, y.reshape(-1, 1)), "drifts": []}
