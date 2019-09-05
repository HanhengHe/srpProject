# -*- coding: UTF-8 -*-
from math import log

import numpy as np
from SVM.SVC import svr


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

class Classifier:
    def __init__(self, svcs, beta_Ts):
        self.svcs = svcs
        #   构造一个队列减少计算量
        self.core = []
        for i in range(len(svcs)):
            self.core.append(log(1 / beta_Ts[i]))

    def predict(self, x):
        right = 0
        left = 0
        for i in range(self.svcs):
            right += self.core[i] * self.svcs[i].predict(x)
            left += self.core[i] / 2
        if right >= left:
            return 1
        else:
            return -1


def trAdaBoost(trans_S, trans_A, label_S, label_A, N, errorRate, param):
    trans_data = trans_S + trans_A
    trans_label = label_S + label_A

    row_A = len(trans_A)
    row_S = len(trans_S)

    # 初始化权重
    # 权重C：C越大系统越重视对应样本
    weights_A = [1 / row_A] * row_A
    weights_S = [1 / row_S] * row_S
    weights = np.mat(weights_A + weights_S)

    beta = 1 / (1 + np.sqrt(2 * np.log(row_A / N)))

    svcs = []
    beta_Ts = []
    result = np.ones([row_A + row_S, N])

    print('params initial finished.')

    # s是预测器
    s = None

    for i in range(N):

        # 归一化权重
        P = calculate_P(weights)

        # 训练分类器并返回预测结果
        result[:, i], s = train_classify(trans_data, trans_label, trans_S, param, P)

        # 计算错误率
        error_rate = calculate_error_rate(label_S, result, weights[row_A:row_A + row_S, :])
        print('Error rate:', error_rate)
        if error_rate > 0.5:
            error_rate = 0.5  # 确保eta大于0.5
        if error_rate <= errorRate:
            N = i
            print("Error rate: " + str(error_rate))
            break  # 防止过拟合

        beta_T = error_rate / (1 - error_rate)

        # 调整源域样本权重
        for j in range(row_S):
            weights[row_A + j] = weights[row_A + j] * np.power(beta_T,
                                                               np.abs(result[row_A + j, i] - label_S[j]))

        # 调整辅域样本权重
        for j in range(row_A):
            weights[j] = weights[j] * np.power(beta, (-np.abs(result[j, i] - label_A[j])))

        # 记录后N/2个分类器和betaT, 向下取整
        if i > int(N / 2):
            svcs.append(s)
            beta_Ts.append(beta_T)

    # 构造训练出来的集成分类器并返回
    classifier = Classifier(svcs, beta_Ts)

    return classifier


# 归一化权重
def calculate_P(weights):
    total = np.sum(weights)
    return np.asarray(weights / total, order='C')


#  训练分类器，返回对源数据集的分类结果以及分类器
def train_classify(trans_data, trans_label, trans_S, param, P):
    s = svr(trans_data, trans_label, param[0], param[1], param[2], param[3], cWeight=P)

    result = np.zeros(1, len(trans_S))
    for i in range(len(trans_S)):
        result[0, i] = s.predict(trans_S[i])
    return result, s


# 计算错误率
def calculate_error_rate(label_R, label_H, weight):
    total = np.sum(weight)

    print(weight[:, 0] / total)
    print(np.abs(label_R - label_H))
    return np.sum(weight[:, 0] / total * np.abs(label_R - label_H))
