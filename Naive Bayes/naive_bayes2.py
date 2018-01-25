#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from collections import Counter, defaultdict
import math


# 计算对数概率值，概率值以字典的结构返回
def occurrences(list1):
    no_of_examples = len(list1)
    prob = dict(Counter(list1))
    for key in prob.keys():
        prob[key] = math.log(prob[key] / float(no_of_examples))
    return prob


# 训练模型
def naive_bayes(training, outcome, new_sample):
    classes = np.unique(outcome)
    rows, cols = np.shape(training)
    likelihoods = {}
    # 以 label 为 key 初始化 likelihoods
    for cls in classes:
        likelihoods[cls] = defaultdict(list)

    # 计算先验概率
    class_probabilities = occurrences(outcome)

    # 根据 label 以及 features 统计 sample 中 feature出现的次数 --> 为计算条件概率做准备
    for cls in classes:
        # 抽取每个 label 对应的样本
        row_indices = np.where(outcome == cls)[0]
        subset = training[row_indices, :]
        r, c = np.shape(subset)
        for j in range(0, c):
            likelihoods[cls][j] += list(subset[:, j])

    # 计算条件概率
    for cls in classes:
        for j in range(0, cols):
            likelihoods[cls][j] = occurrences(likelihoods[cls][j])

    # 新样本做测试
    results = {}
    for cls in classes:
        class_probability = class_probabilities[cls]
        for i in range(0, len(new_sample)):
            relative_values = likelihoods[cls][i]
            if new_sample[i] in relative_values.keys():
                class_probability += relative_values[new_sample[i]]
            else:
                class_probability += 0
            results[cls] = class_probability

    result = -100
    clazz = ''
    for cls in classes:
        if result < results[cls]:
            result = results[cls]
            clazz = cls

    print(clazz)


if __name__ == "__main__":
    # 训练集
    training = np.asarray((
        (1, 0, 1, 1),
        (1, 1, 0, 0),
        (1, 0, 2, 1),
        (0, 1, 1, 1),
        (0, 0, 0, 0),
        (0, 1, 2, 1),
        (0, 1, 2, 0),
        (1, 1, 1, 1)
    ))

    # 标签
    outcome = np.asarray((0, 1, 1, 1, 0, 1, 0, 1))

    # 测试数据
    new_sample = np.asarray((1, 0, 1, 0))

    # 返回结果
    naive_bayes(training, outcome, new_sample)