# -*- coding: UTF-8 -*-

#  import multiprocessing
from multiprocessing import Process, cpu_count
from SVM.transfrom.trAdaBoost import trAdaBoost
import numpy as np

###############################################
#               wdnmd GIL                     #
#               wdnmd GIL                     #
#               wdnmd GIL                     #
#               wdnmd GIL                     #
#               wdnmd GIL                     #
#               wdnmd GIL                     #
#               wdnmd GIL                     #
#               wdnmd GIL                     #
###############################################

#   这是一个不成熟的svc多分类器

#   目前只打算写 dag 部分和 ecoc 部分
#   ovo 和 ovr 不打算做

Type = ('DAG', 'ECOC')

#  cores = multiprocessing.cpu_count()
cores = cpu_count()


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
    def __init__(self, dataList_A, dataList_S, C, tol, maxIter, kTup=('lin', 0), trMaxIter=20, trTol=0.05,
                 coreNum=cores, nonTr=False, classifierType=Type[0]):
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
        self.coreNum = coreNum
        self.nonTr = nonTr

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

        #   需要 num*(num+1)/2 个分类器
        self.svcs = [None] * int(self.num * (self.num + 1) / 2)
        self.svcsName = []

        self.train()

        # ************************到这******************************

    #   End function

    def train(self):

        #  prepare for train

        #   require num*(num+1)/2 classifiers

        #  prepare for svc nameList
        for ini in range(self.num):
            for j in range(ini + 1, self.num):
                self.svcsName.append(str(self.neatLabelSet_A[ini]) + '&' + str(self.neatLabelSet_A[j]))

        #  thread mission dispatch

        perCore = int(len(self.svcsName) / self.coreNum) + 1
        threadMission = []
        threadMIndex = []

        for ini in range(self.coreNum):
            threadMission.append([])
            threadMIndex.append([])

        index = 0

        for ini in range(len(self.svcsName)):
            threadMission[index].append(self.svcsName[ini])
            threadMIndex[index].append(ini)
            if len(threadMission[index]) == perCore:
                index += 1

        #  start train

        #  memory requirement would be huge here
        """counter = 0
        for i in range(self.num):
            for j in range(i + 1, self.num):
                counter += 1
                print(counter)
                self.svcs.append(
                    trAdaBoost(self.neatDataSet_A[i] + self.neatDataSet_A[j],
                               # A svc on type number i and type number j
                               self.neatDataSet_S[i] + self.neatDataSet_S[j],  # S
                               [-1] * len(self.neatDataSet_A[i]) + [1] * len(self.neatDataSet_A[j]),  # A label set
                               [-1] * len(self.neatDataSet_S[i]) + [1] * len(self.neatDataSet_S[j]),  # S label set
                               [self.C, self.tol, self.maxIter, self.kTup],
                               self.trMaxIter, self.trTol,  # parameters
                               self.svcsName[len(self.svcs)]  # check trAdaBoost
                               )
                )
        """

        processes = []

        for ini in range(self.coreNum):
            process = Process(target=self.subThread, args=(threadMission[ini], threadMIndex[ini]))
            processes.append(process)
        if __name__ == '__main__':
            for pr in processes:
                pr.start()

            for pr in processes:
                pr.join()

    #   End function

    def subThread(self, missionList, threadMIndex):
        for iterIndex in range(len(missionList)):
            a = int(missionList[iterIndex].split('&')[0])
            b = int(missionList[iterIndex].split('&')[1])
            self.svcs[threadMIndex[iterIndex]] = \
                trAdaBoost(self.neatDataSet_A[a] + self.neatDataSet_A[b],  # A
                           self.neatDataSet_S[a] + self.neatDataSet_S[b],  # S
                           [-1] * len(self.neatDataSet_A[a]) + [1] * len(
                               self.neatDataSet_A[b]),
                           # A label set
                           [-1] * len(self.neatDataSet_S[a]) + [1] * len(
                               self.neatDataSet_S[b]),
                           # S label set
                           [self.C, self.tol, self.maxIter, self.kTup],
                           self.trMaxIter, self.trTol,  # parameters
                           self.svcsName[int(threadMIndex[iterIndex])],  # check trAdaBoost
                           self.nonTr  # with non-tr support
                           )

    #   End function

    def predict(self, x, real=''):  # 方便整合输出

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
        log = open("D:\\WINTER\\Pycharm_project\\srpProject\\SVM\\predictLog", 'a')
        log.write("\nPredict step:\n")
        log.write("Compare between " + self.svcsName[0] + ',')
        log.write("step is " + predict + ';\n')

        while True:

            if index == len(self.svcsName):
                log.write("real label is " + real + "\n")
                log.close()
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
            log.write("Compare between " + self.svcsName[index] + ',')
            log.write("step is " + predict + ';\n')

            index += 1

    #   End function


# work part
# thanks for GIL
# work part was moved to here
# wdnmd GIL

trainFile = open('D:\\WINTER\\Pycharm_project\\data\\Mnist\\train')

ASRate = 0.1

# svc get double size

trainASize = 500
SourceSize = 150
trainSSize = trainASize * ASRate
testSize = SourceSize - trainSSize

C = 0.8
tol = 0.1
maxIter = 20
kTup = ['lin', 0]
trMaxIter = 1
errorRate = 0.05
coreNum = 6
nonTr = True

trainSetA = []
SourceSet = []

trainSetS = []
testSet = []
testLabels = []

trainSCounter = [[1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0], [8, 0], [9, 0], [0, 0]]

SourceCounter = [[1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0], [8, 0], [9, 0], [0, 0]]

trainACounter = [[1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0], [8, 0], [9, 0], [0, 0]]

testCounter = [[1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0], [8, 0], [9, 0], [0, 0]]

#  clear old log
oLog = open("D:\\WINTER\\Pycharm_project\\srpProject\\SVM\\predictLog", 'w')
oLog.close()

#   assistant data
for line in trainFile.readlines():
    #  修改格式
    dataSet = line.split(')')[0]
    label = line.split(')')[1].replace('\n', '')

    # 计数
    #  ****************从这开始*********************
    if trainACounter[int(label)][1] == trainASize:
        continue

    trainACounter[int(label)][1] = trainACounter[int(label)][1] + 1

    #  ******************到这***********************

    dataSets = dataSet.split(',')
    dataSets[0] = dataSets[0].replace('(', '')

    #  data set调整为float类型，label调整为int类型
    temp = []
    for i in range(len(dataSets)):
        t = int(dataSets[i]) / 255  # 归一化
        # 四舍五入
        if t > 0.5:
            temp.append(1)
        else:
            temp.append(0)

    temp.append(label)

    #  置入数据结构中
    trainSetA.append(temp)

trainFile.close()

testFile = open('D:\\WINTER\\Pycharm_project\\data\\Mnist\\test')

#   data source and test data
for line in testFile.readlines():
    #  修改格式
    dataSet = line.split(')')[0]
    label = line.split(')')[1].replace('\n', '')

    # 计数
    #  ****************从这开始*********************
    if SourceCounter[int(label)][1] == SourceSize:
        continue

    SourceCounter[int(label)][1] = SourceCounter[int(label)][1] + 1

    #  ******************到这***********************

    dataSets = dataSet.split(',')
    dataSets[0] = dataSets[0].replace('(', '')

    #  data set调整为float类型，label调整为int类型
    temp = []
    for i in range(len(dataSets)):

        t = int(dataSets[i]) / 255  # 归一化
        # 四舍五入
        if t > 0.5:
            temp.append(1)
        else:
            temp.append(0)

    temp.append(label)

    #  置入数据结构中
    SourceSet.append(temp)

testFile.close()

for i in range(len(SourceSet)):
    if trainSSize > trainSCounter[int(SourceSet[i][len(SourceSet[i]) - 1])][1]:
        trainSetS.append(SourceSet[i])
        trainSCounter[int(SourceSet[i][len(SourceSet[i]) - 1])][1] = \
            trainSCounter[int(SourceSet[i][len(SourceSet[i]) - 1])][1] + 1
    else:
        testSet.append(SourceSet[i][:len(SourceSet[i]) - 1])
        testLabels.append(SourceSet[i][len(SourceSet[i]) - 1])

if not isinstance(trainSetA, list):
    raise NameError('error: dataList_A should be a list.')

if not isinstance(trainSetS, list):
    raise NameError('error: dataList_S should be a list.')

classifier = Classifier(trainSetS, trainSetA, C, tol, maxIter, kTup, trMaxIter, errorRate, coreNum, nonTr)

correct = 0

error = np.zeros((10, 10), int)

# test
for ind in range(0, len(testSet)):
    prediction = classifier.predict(testSet[ind], str(testLabels[ind]))

    print(ind, end='')
    print(" predict is ", end='')
    print(prediction, end=';')
    print(" real label is ", end=str(testLabels[ind]) + ";")

    if prediction == testLabels[ind]:
        correct += 1
        print('predict is right.')
    else:
        error[int(testLabels[ind]), int(prediction)] = error[int(testLabels[ind]), int(prediction)] + 1
        print('predict is wrong.')

print("Source train data set situation: ")
print(trainSCounter)

print("test data set situation: ")
print(testCounter)

print("Correct present: ")
print(correct / len(testLabels))
print(error)
