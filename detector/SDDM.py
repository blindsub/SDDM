#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/10/14 18:24
# @Author  : ZWP
# @Desc    : 
# @File    : SDDM.py
import numpy as np
import scipy
import pandas as pd
import shap
import tensorflow as tf
import tensorflow.compat.v1.keras.backend as K
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.layers import Input, Dense, Flatten, concatenate, Dropout, Conv2D, \
    MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.python.keras.layers.embeddings import Embedding
from sklearn.tree import DecisionTreeRegressor

tf.compat.v1.disable_eager_execution()


class explain_model:
    def __init__(self, X_train, y_train, layer_name="drift"):
        self.X_train = X_train
        self.y_train = y_train
        self.dtypes = None
        self.model = None
        self.X_middle_train = None
        self.layer_name = layer_name

        # 进行数据集的归一化处理
        X = X_train
        y = y_train
        self.dtypes = list(zip(X.dtypes.index, map(str, X.dtypes)))
        for k, dtype in self.dtypes:
            if dtype == "float32":
                X[k] -= X[k].mean()
                X[k] /= X[k].std()

        # 若整列都可被1整除，则为分类问题
        values = []
        if all(y.iloc[0] % 1 == 0):
            y[0] = y[0].astype(int)
            values = list(set(y[0]))
            if len(values) == 2:
                data_type = "binary-classifier"
                y[0] = y[0].replace({values[0]: 0, values[1]: 1})
            elif len(values) < 20:
                data_type = "multi-classifier"
                enc = OneHotEncoder()
                enc.fit(y[0].values.reshape(-1, 1))
                # one-hot编码的结果是比较奇怪的，最好是先转换成二维数组
                y = pd.DataFrame(enc.transform(y[0].values.reshape(-1, 1)).toarray())
            else:
                data_type = "regression"
        else:
            data_type = "regression"

        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=7)
        input_els = []
        encoded_els = []
        for k, dtype in self.dtypes:
            input_els.append(Input(shape=(1,)))
            # 从刚刚加入的Input上创建一个嵌入层加平铺层，否则不加入嵌入层
            if dtype == "int8":
                e = Flatten()(Embedding(X_train[k].max() + 1, 1)(input_els[-1]))
            else:
                e = input_els[-1]
            encoded_els.append(e)
        encoded_els = concatenate(encoded_els)
        layer1 = Dropout(0.5)(Dense(10, activation="relu", name="input")(encoded_els))
        layer2 = Dense(30, name="drift", activation="relu")(layer1)
        layer3 = Dense(5)(layer2)
        if data_type == "binary-classifier":
            # train model
            out = Dense(1, name="output", activation="sigmoid")(layer3)
            self.model = Model(inputs=input_els, outputs=[out])
            self.model.compile(optimizer="adam", loss='binary_crossentropy',
                               metrics=['accuracy'])

        if data_type == "regression":
            # train model
            out = Dense(1, name="output")(layer3)
            self.model = Model(inputs=input_els, outputs=[out])
            self.model.compile(optimizer="adam", loss='mean_squared_error',
                               metrics=['accuracy'])
        if data_type == "multi-classifier":
            # train model
            out = Dense(len(values), name="output", activation='softmax')(layer3)
            self.model = Model(inputs=input_els, outputs=[out])
            self.model.compile(optimizer="adam", loss='categorical_crossentropy',
                               metrics=['accuracy'])
        self.model.fit(
            [X_train[k].values for k, t in self.dtypes],
            y_train,
            epochs=200,
            batch_size=512,
            shuffle=True,
            validation_data=([X_valid[k].values for k, t in self.dtypes], y_valid),
            verbose=0,
            callbacks=[
                # early_stop
            ]
        )
        self.X_middle_train = self.map2layer(X.copy(), self.layer_name)

        self.explainer = shap.DeepExplainer(
            (self.model.get_layer(self.layer_name).input, self.model.layers[-1].output),
            self.map2layer(X.copy(), self.layer_name))

        shap_values = self.explainer.shap_values(self.map2layer(X, self.layer_name))
        # shap.force_plot(shap_values, X.iloc[299, :])
        self.X_middle_train = self.layer2df(self.X_train)

    def preprocessing(self, x_copy):
        # 同样归一化
        for k, dtype in self.dtypes:
            if dtype == "float32":
                x_copy[k] -= x_copy[k].mean()
                x_copy[k] /= x_copy[k].std()
        return x_copy

    def map2layer(self, x, layer_name):
        x_copy = x.copy()
        self.preprocessing(x_copy)

        feed_dict = dict(zip(self.model.inputs, [np.reshape(x_copy[k].values, (-1, 1)) for k, t in self.dtypes]))
        return K.get_session().run(self.model.get_layer(layer_name).input, feed_dict)

    def layer2df(self, x):
        layer = self.map2layer(x, self.layer_name)
        layer = pd.DataFrame(layer)
        return layer

    def get_shap(self, X):
        return self.explainer.shap_values(self.map2layer(X, self.layer_name))

    def re_train(self, X_retrain):
        X_retrain = self.preprocessing(X_retrain)

        # 尝试去更新检测器，实际上是去改变均值，去获取当前均值
        self.explainer = shap.GradientExplainer(
            (self.model.get_layer(self.layer_name).input, self.model.layers[-1].output),
            self.map2layer(X_retrain.copy(), self.layer_name))


class SDDM():

    def __init__(self, X_train, y_train, window_size, shap_class=0, alpha=0.01, threshold=0.99):

        X_train = pd.DataFrame(X_train)
        y_train = pd.DataFrame(y_train)
        self.alpha = alpha
        self.shap_class = shap_class
        self.windows_size = window_size
        self.current_window = []
        self.threshold = threshold
        self.model = explain_model(X_train, y_train)

        # 获取训练数据的shap值，进行特征的过滤
        shap = self.model.get_shap(X_train)
        self.feature_select = self.feat_selection(shap[shap_class])
        self.shap_predict = DecisionTreeRegressor()
        self.shap_predict.fit(X_train, self.model.get_shap(X_train)[shap_class][:, self.feature_select])
        self.drift_detected = False

    def add_batch(self, x):

        self.drift_detected = False
        # 首先还是先做数据集积累
        self.current_window.append(x)
        # 当数据积累够了之后进行漂移检测

        if len(self.current_window) * self.current_window[0].shape[0] > self.windows_size:
            detected_window = np.concatenate([x for x in self.current_window], axis=0)
            self.current_window = []

            x = pd.DataFrame(detected_window)
            # 预测值
            shap_pred = self.shap_predict.predict(x)
            # 真实值,需要进行过滤
            shap = self.model.get_shap(x)[self.shap_class][:, self.feature_select]
            # t_value, p_value = stats.ttest_ind(shap, shap_pred)
            p_value = [scipy.stats.ks_2samp(shap[:, i], shap_pred[:, i]).pvalue for i in range(shap.shape[1])]
            # print(p_value)
            # 存在至少一个列的p值分布有差异
            if sum(np.array(p_value) <= self.alpha) > 0:
                self.shap_predict.fit(x, shap)
                # 标识当前窗口是否检测到漂移
                self.drift_detected = True

    def detected_change(self):
        return self.drift_detected

    def feat_selection(self, shap_values):
        feature_important = []
        feature_select = []
        for shap_index in range(shap_values.shape[1]):
            feature_shap_values = shap_values[:, shap_index]
            # 先求绝对值之后再求平均值，作为特征重要性
            feature_important.append(np.mean(abs(feature_shap_values)))
        feature_important = np.array(feature_important) / sum(feature_important)
        feature_important_zip = list(zip(range(shap_values.shape[1]), feature_important))
        feature_sorted = sorted(feature_important_zip, key=lambda x: x[1], reverse=True)
        important_sum = 0
        for shap_index, important in feature_sorted:
            important_sum += important
            feature_select.append(shap_index)
            if important_sum >= self.threshold:
                if len(feature_select) > 20:
                    feature_select = feature_select[:20]
                return feature_select
