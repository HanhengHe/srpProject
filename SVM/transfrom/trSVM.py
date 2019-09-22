# -*- coding: UTF-8 -*-
from SVM.transfrom.trAdaBoost import trAdaBoost

#   这是一个不成熟的svc多分类器

#   目前只打算写 dag 部分和 ecoc 部分
#   ovo 和 ovr 不打算做

Type = ('DAG', 'ECOC')


#   List格式：[data,...,data,label]
#   建议label从0开始，不跳过数字
#   当然其实你不这么干也没什么关系

#   返回训练好的分类器，直接调用predict
#   classifier = Classifier(parameters...)
#   classifier.predict(parameters...)

#   这是迁移学习的SVM
#   调用的是trAdaBoost.py

#   暂时来说参数是共享的
#   如果效果不好可以考虑变参处理
#   主要是变核函数的分母
#   但是具体操作不太懂

#   Error我是随便raise的, 因为我不知道raise什么Error比较合适....

class Classifier:
    def __init__(self, dataList_A, dataList_S, C, tol, maxIter, kTup=('lin', 0), trMaxIter=20, trTol=0.05, classifierType=Type[0]):
        #   check type
        if not isinstance(dataList_A, list):
            raise NameError('error: dataList_A should be a list.')

        if not isinstance(dataList_S, list):
            raise NameError('error: dataList_S should be a list.')

        self.C = C
        self.tol = tol
        self.maxIter = maxIter
        self.kTup = kTup
        self.trMaxIter = trMaxIter
        self.trTol = trTol
        self.classifierType = classifierType

        #   需要 num*(num+1)/2 个分类器
        self.svcs = []
        self.svcsName = []

        # ***************这部分代码还可以进一步优化******************
        #   按标签类型重新整理数据集
        #   list嵌套list，这种方式的空间效率可能会很低，期待后续修正

        #   neatDataSet: [[type one set], [type two set]]
        #   neatLabelSet: [typeOneLabel, typeTwoLabel]

        #  初始化 neatDataSet_A
        #  二重嵌套循环
        #  O(h^2)
        self.neatDataSet_A = []
        self.neatLabelSet_A = []

        for data in dataList_A:
            if data[len(data) - 1] not in self.neatLabelSet_A:  # 如果data末尾的标签不在neatLabelSet中
                self.neatLabelSet_A.append((data[len(data) - 1]))

                self.neatDataSet_A.append([])

                for da in dataList_A:  # 再次遍历数据集
                    if da[len(da) - 1] == self.neatLabelSet_A[len(self.neatLabelSet_A) - 1]:  # 如果da末尾标签与当前处理标签相符
                        self.neatDataSet_A[len(self.neatDataSet_A) - 1].append(da[0:len(da) - 1])

        #  初始化 neatDataSet_A
        #  二重嵌套循环
        #  O(h^2)
        self.neatDataSet_S = []
        self.neatLabelSet_S = []

        for data in dataList_S:
            if data[len(data) - 1] not in self.neatLabelSet_S:  # 如果data末尾的标签不在neatLabelSet中
                self.neatLabelSet_S.append((data[len(data) - 1]))

                self.neatDataSet_S.append([])

                for da in dataList_S:  # 再次遍历数据集
                    if da[len(da) - 1] == self.neatLabelSet_S[len(self.neatLabelSet_S) - 1]:  # 如果da末尾标签与当前处理标签相符
                        self.neatDataSet_S[len(self.neatDataSet_S) - 1].append(da[0:len(da) - 1])

        #  计算类别数目
        self.num = len(self.neatLabelSet_A)
        if self.num <= 1:
            raise NameError('error: require two or more types .')

        if self.num != len(self.neatLabelSet_S):
            raise NameError('error: some types in assistant data did not find in source data.')

        #  训练
        if classifierType not in Type:
            raise NameError('error: classifierType error .')

        self.train()

        # ************************到这******************************

    #   End function

    def train(self):

        #  prepare for train

        #   require num*(num+1)/2 classifiers

        #  prepare for svc nameList
        for i in range(self.num):
            for j in range(i + 1, self.num):
                self.svcsName.append(str(self.neatLabelSet_A[i]) + '&' + str(self.neatLabelSet_A[j]))

        #  start train
        #  memory requirement would be huge here
        for i in range(self.num):
            for j in range(i + 1, self.num):
                self.svcs.append(
                    trAdaBoost(self.neatDataSet_A[i] + self.neatDataSet_A[j],  # A svc on type number i and type number j
                               self.neatDataSet_S[i] + self.neatDataSet_S[j],  # S
                               [-1] * len(self.neatDataSet_A[i]) + [1] * len(self.neatDataSet_A[j]),  # A label set
                               [-1] * len(self.neatDataSet_S[i]) + [1] * len(self.neatDataSet_S[j]),  # S label set
                               [self.C, self.tol, self.maxIter, self.kTup],
                               self.trMaxIter, self.trTol  # parameters
                               )
                )

    #   End function

    def predict(self, x):

        #   I have no idea whether this will work

        atList = self.neatLabelSet_A.copy()

        index = 1

        firstStep = self.svcs[0].predict(x)

        if firstStep < 0:
            predict = self.svcsName[0].split('&')[0]
            atList.remove(self.svcsName[0].split('&')[1])
        elif firstStep > 0:
            predict = self.svcsName[0].split('&')[1]
            atList.remove(self.svcsName[0].split('&')[0])
        else:
            raise NameError('error: predict zero .')

        #  输出判断步
        print("\nPredict step:")
        print("Compare between ", end=self.svcsName[0]+';')
        print("step is ", end=predict)

        while True:

            if index == len(self.svcsName):
                return predict

            if predict not in self.svcsName[index].split('&'):
                index += 1
                continue

            tempIndex = (self.svcsName[index].split('&').index(predict) + 1) % 2

            if self.svcsName[index].split('&')[tempIndex] not in atList:
                index += 1
                continue

            takeStep = self.svcs[index].predict(x)

            if takeStep < 0:
                predict = self.svcsName[index].split('&')[0]
                atList.remove(self.svcsName[index].split('&')[1])

            elif takeStep > 0:
                predict = self.svcsName[index].split('&')[1]
                atList.remove(self.svcsName[index].split('&')[0])

            else:
                raise NameError('error: predict zero .')

            #  输出判断步
            print("Compare between ", end=self.svcsName[0] + ';')
            print("step is ", end=predict)

            index += 1

    #   End function
