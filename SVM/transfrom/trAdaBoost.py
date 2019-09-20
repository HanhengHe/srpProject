# -*- coding: UTF-8 -*-
from math import log

import numpy as np
from SVM.SVC import svc


#   parameter全是list
#   返回训练后的模型
#   TrainS 原训练样本
#   TrainA 辅助训练样本
#   LabelS 原训练样本标签
#   LabelA 辅助训练样本标签
#   DataT  测试样本
#   LabelT 测试样本标签
#   N 迭代次数
#   param=[C, tol, maxIter, kTup]

class trClassifier:
    def __init__(self, svcs, beta_Ts):
        self.svcs = svcs
        #   构造一个队列减少计算量
        self.core = []
        for i in range(len(svcs)):
            self.core.append(log(1 / beta_Ts[i]))

    def predict(self, x):
        right = 0
        left = 0
        for i in range(len(self.svcs)):
            right += self.core[i] * self.svcs[i].predict(x)
            left += self.core[i] / 2
        if right >= left:
            return 1
        else:
            return -1


def trAdaBoost(trans_S, trans_A, label_S, label_A, param, N=20, errorRate=0.05):

    print("trAdaBoost.")

    trans_data = trans_A + trans_S

    row_A = len(trans_A)
    row_S = len(trans_S)

    # 初始化权重
    # 权重C：C越大系统越重视对应样本
    # ??
    # weights_A = [1 / row_A] * row_A
    # weights_S = [1 / row_S] * row_S
    # weights = np.mat(weights_A + weights_S)
    weights = np.mat([1/(row_A + row_S)] * (row_A + row_S))

    beta = 1 / (1 + np.sqrt(2 * np.log(row_A / N)))

    svcs = []
    beta_Ts = []
    result = np.ones([row_A + row_S, N])

    print('params initial finished.')

    for i in range(N):

        # 归一化权重
        weights = calculate_P(weights)

        # 训练分类器并返回预测结果
        result[:, i], classifier = train_classify(trans_data, label_A+label_S, param, weights.tolist()[0])

        # 计算错误率
        error_rate = calculate_error_rate(np.mat(label_S), result[row_A:row_A + row_S, i], weights[0, row_A:row_A + row_S])
        print('Error rate:', error_rate)
        print('')
        if error_rate > 0.5:
            error_rate = 0.5  # 确保eta大于0.5
        if error_rate <= errorRate:
            break  # 防止过拟合

        beta_T = error_rate / (1 - error_rate)

        # 调整源域样本权重
        for j in range(row_S):
            weights[0, row_A + j] = weights[0, row_A + j] * np.power(beta_T,
                                                               -np.abs(result[row_A + j, i] - label_S[j]) / 2)

        # 调整辅域样本权重
        for j in range(row_A):
            weights[0, j] = weights[0, j] * np.power(beta, np.abs(result[j, i] - label_A[j]) / 2)

        # 记录后N/2个分类器和betaT, 向下取整
        if i > int(N / 2):
            svcs.append(classifier)
            beta_Ts.append(beta_T)

    # 构造训练出来的集成分类器并返回
    classifier = trClassifier(svcs, beta_Ts)

    print("trAdaBoost finished.")

    return classifier


# 归一化权重
def calculate_P(weights):
    total = np.sum(weights)
    return weights / total


#  训练分类器，返回对源数据集的分类结果以及分类器
def train_classify(trans_data, trans_labels, param, P):
    classifier = svc(trans_data, trans_labels, param[0], param[1], param[2], param[3], cWeight=P)

    result = [0]*len(trans_data)
    for i in range(len(trans_data)):
        result[i] = classifier.predict(trans_data[i])
        result[i] = result[i]/abs(result[i])
    return result, classifier


# 计算错误率
def calculate_error_rate(label_R, label_H, weight):
    total = np.sum(weight)

    return (weight * np.abs(label_R - label_H).T / total)/2
