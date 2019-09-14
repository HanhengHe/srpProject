# -*- coding: UTF-8 -*-
from SVM.SVC import svc

#   这是一个不成熟的svr多分类器

#   目前只打算写 dag 部分和 ecoc 部分
#   ovo 和 ovr 不打算做

Type = ('DAG', 'ECOC')


#   List格式：[data,...,data,label]
#   建议label从0开始，不跳过数字
#   当然其实你不这么干也没什么关系

#   返回训练好的分类器，直接调用predict
#   classifier = Classifier(parameters...)
#   classifier.predict(parameters...)

#   暂时来说参数是共享的
#   如果效果不好可以考虑变参处理
#   主要是变核函数的分母
#   但是具体操作不太懂

#   Error我是随便raise的, 因为我不知道raise什么Error比较合适....

class Classifier:
    def __init__(self, dataList, C, tol, maxIter, kTup=('lin', 0), cWeight=0, classifierType=Type[0]):
        #   check type
        if not isinstance(dataList, list):
            raise NameError('error: dataList should be a list.')

        self.C = C
        self.tol = tol
        self.maxIter = maxIter
        self.kTup = kTup
        self.cWeight = cWeight
        self.classifierType = classifierType

        #   需要 num*(num+1)/2 个分类器
        self.svcs = []
        self.svcsName = []

        # ***************这部分代码还可以进一步优化******************
        #   按标签类型重新整理数据集
        #   list嵌套list，这种方式的空间效率可能会很低，期待后续修正

        #   neatDataSet: [[type one set], [type two set]]
        #   neatLabelSet: [typeOneLabel, typeTwoLabel]

        #  初始化 neatDataSet
        #  二重嵌套循环
        #  O(h^2)
        self.neatDataSet = []
        self.neatLabelSet = []

        # 下面这块二重循环仔细看看
        for data in dataList:
            if data[len(data) - 1] not in self.neatLabelSet:  # 如果data末尾的标签不在neatLabelSet中
                self.neatLabelSet.append((data[len(data) - 1], 1))
                tempDataSet = []  # 一个临时的list

                for da in dataList:  # 再次遍历数据集
                    if da[len(data) - 1] == self.neatLabelSet[len(self.neatLabelSet)]:  # 如果da末尾标签与当前处理标签相符
                        tempDataSet.append(da)

                self.neatDataSet.append(tempDataSet)

        #  计算类别数目
        self.num = len(self.neatLabelSet)
        if self.num <= 1:
            raise NameError('error: require two or more types .')

        #  训练
        if classifierType not in type:
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
                self.svcsName.append(str(self.neatLabelSet[i]) + '&' + str(self.neatLabelSet[j]))

        #  start train
        #  memory requirement would be huge here
        for i in range(self.num):
            for j in range(i + 1, self.num):
                self.svcs.append(
                    svc(self.neatDataSet[i] + self.neatDataSet[j],  # svc on type number i and type number j
                        [-1] * len(self.neatDataSet[i]) + [1] * len(self.neatDataSet[j]),  # label set
                        self.C, self.tol, self.maxIter, self.kTup, self.cWeight  # parameters
                        )
                )

    #   End function

    def predict(self, x):
        #   I have no idea whether this will work

        index = 1

        firstStep = self.svcs[0].predict(x)

        if firstStep < 0:
            predict = self.svcsName[0].split('&')[0]
        elif firstStep > 0:
            predict = self.svcsName[0].split('&')[1]
        else:
            raise NameError('error: predict zero .')

        while True:

            if index == self.num:
                return predict

            if predict not in self.svcsName[index].split('&'):
                index += 1
                continue

            takeStep = self.svcs[index].predict(x)

            if takeStep < 0:
                predict = self.svcsName[index].split('&')[0]
            elif takeStep > 0:
                predict = self.svcsName[index].split('&')[1]
            else:
                raise NameError('error: predict zero .')

            index += 1

    #   End function
