# -*- coding: UTF-8 -*-

from multiprocessing import Pool, cpu_count, freeze_support
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
#      (SVM决策树, ECOC)

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

# work part
# thanks for GIL
# work part was moved to here
# wdnmd GIL

ASRate = 0.1

# svc get double size

trainASize = 50
SourceSize = 15
trainSSize = trainASize * ASRate
testSize = SourceSize - trainSSize

C = 0.8
tol = 0.1
maxIter = 20
kTup = ['lin', 0]
trMaxIter = 10
trTol = 0.05
errorRate = 0.05
# coreNum = cpu_count()
coreNum = 1
nonTr = False

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

neatDataSet_A = []
neatLabelSet_A = []

neatDataSet_S = []
neatLabelSet_S = []

num = 0

svcs = []
svcsName = []

perCore = 0
threadMission = []
threadMIndex = []

processes = []


def init():
    ###########################################################################################
    #                                      init data                                          #
    ###########################################################################################

    trainFile = open('D:\\WINTER\\Pycharm_project\\data\\Mnist\\train')

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
        tempIn = []
        for i in range(len(dataSets)):
            t = int(dataSets[i]) / 255  # 归一化
            # 四舍五入
            if t > 0.5:
                tempIn.append(1)
            else:
                tempIn.append(0)

        tempIn.append(label)

        #  置入数据结构中
        trainSetA.append(tempIn)

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
        tempIn = []
        for i in range(len(dataSets)):

            t = int(dataSets[i]) / 255  # 归一化
            # 四舍五入
            if t > 0.5:
                tempIn.append(1)
            else:
                tempIn.append(0)

        tempIn.append(label)

        #  置入数据结构中
        SourceSet.append(tempIn)

    testFile.close()

    for i in range(len(SourceSet)):
        if trainSSize > trainSCounter[int(SourceSet[i][len(SourceSet[i]) - 1])][1]:
            trainSetS.append(SourceSet[i])
            trainSCounter[int(SourceSet[i][len(SourceSet[i]) - 1])][1] = \
                trainSCounter[int(SourceSet[i][len(SourceSet[i]) - 1])][1] + 1
        else:
            testSet.append(SourceSet[i][:len(SourceSet[i]) - 1])
            testLabels.append(SourceSet[i][len(SourceSet[i]) - 1])
            testCounter[int(SourceSet[i][len(SourceSet[i]) - 1])][1] = \
                testCounter[int(SourceSet[i][len(SourceSet[i]) - 1])][1] + 1

    if not isinstance(trainSetA, list):
        raise NameError('error: dataList_A should be a list.')

    if not isinstance(trainSetS, list):
        raise NameError('error: dataList_S should be a list.')


def listData():
    ###########################################################################################
    #                                      list data                                          #
    ###########################################################################################

    # ***************这部分代码还可以进一步优化******************
    #   按标签类型重新整理数据集
    #   list嵌套list，这种方式的空间效率可能会很低，期待后续修正

    #   neatDataSet: [[type one set], [type two set]]
    #   neatLabelSet: [typeOneLabel, typeTwoLabel]

    #  初始化 neatDataSet_A
    #  二重嵌套循环
    #  O(h^2)

    for data in trainSetA:
        if data[len(data) - 1] not in neatLabelSet_A:  # 如果data末尾的标签不在neatLabelSet中
            neatLabelSet_A.append((data[len(data) - 1]))  # 注意neatLabelSet_A乱序

            neatDataSet_A.append([])

            for da in trainSetA:  # 再次遍历数据集
                if da[len(da) - 1] == neatLabelSet_A[len(neatLabelSet_A) - 1]:  # 如果da末尾标签与当前处理标签相符
                    neatDataSet_A[len(neatDataSet_A) - 1].append(da[0:len(da) - 1])

    #  初始化 neatDataSet_A
    #  二重嵌套循环
    #  O(h^2)

    for data in trainSetS:
        if data[len(data) - 1] not in neatLabelSet_S:  # 如果data末尾的标签不在neatLabelSet中
            neatLabelSet_S.append((data[len(data) - 1]))

            neatDataSet_S.append([])

            for da in trainSetS:  # 再次遍历数据集
                if da[len(da) - 1] == neatLabelSet_S[len(neatLabelSet_S) - 1]:  # 如果da末尾标签与当前处理标签相符
                    neatDataSet_S[len(neatDataSet_S) - 1].append(da[0:len(da) - 1])

    # ************************到这******************************

    #  计算类别数目
    global num, svcs
    num = len(neatLabelSet_A)
    if num <= 1:
        raise NameError('error: require two or more types .')

    if num != len(neatLabelSet_S):
        raise NameError('error: some types in assistant data did not find in source data.')

    #  训练

    #   需要 num*(num+1)/2 个分类器
    svcs = [None] * int(num * (num - 1) / 2)


#  ***************************************************************************************
#                                  assist function                                       *
#  ***************************************************************************************

def subProcess(missionList, neatDataSet_Assist, neatDataSet_Source, neatLabelSet_Assist, neatLabelSet_Source, proNum):
    svms = [None] * len(missionList)
    for iterIndex in range(len(missionList)):
        print("Core %s processing %s" % (str(proNum), str((iterIndex+1)/len(missionList))))
        left = int(missionList[iterIndex].split('&')[0])
        right = int(missionList[iterIndex].split('&')[1])
        a = neatLabelSet_Assist.index(left)
        b = neatLabelSet_Source.index(right)
        svms[iterIndex] = \
            trAdaBoost(neatDataSet_Assist[a] + neatDataSet_Assist[b],  # A
                       neatDataSet_Source[a] + neatDataSet_Source[b],  # S
                       [-1] * len(neatDataSet_Assist[a]) + [1] * len(
                           neatDataSet_Assist[b]),
                       # A label set
                       [-1] * len(neatDataSet_Source[a]) + [1] * len(
                           neatDataSet_Source[b]),
                       # S label set
                       [C, tol, maxIter, kTup],
                       trMaxIter, trTol,  # parameters
                       missionList[iterIndex],  # check trAdaBoost
                       proNum,
                       nonTr  # with non-tr support
                       )
    return svms


def predict(x, real=''):  # 方便整合输出

    #   I have no idea whether this will work

    atList = neatLabelSet_A.copy()

    indexIn = 1

    firstStep = svcs[0].predict(x)

    if firstStep < 0:
        predictIn = svcsName[0].split('&')[0]
        atList.remove(svcsName[0].split('&')[1])
    elif firstStep > 0:
        predictIn = svcsName[0].split('&')[1]
        atList.remove(svcsName[0].split('&')[0])
    else:
        raise NameError('error: predict zero .')

    #  输出判断步
    log = open("D:\\WINTER\\Pycharm_project\\srpProject\\SVM\\predictLog", 'a')
    log.write("\nPredict step:\n")
    log.write("Compare between " + svcsName[0] + ',')
    log.write("step is " + predictIn + ';\n')

    while True:

        if indexIn == len(svcsName):
            log.write("real label is " + real + "\n")
            log.close()
            return predictIn

        if predictIn not in svcsName[indexIn].split('&'):
            indexIn += 1
            continue

        tempIndex = (svcsName[indexIn].split('&').index(predictIn) + 1) % 2

        if svcsName[indexIn].split('&')[tempIndex] not in atList:
            indexIn += 1
            continue

        takeStep = svcs[indexIn].predict(x)

        if takeStep < 0:
            predictIn = svcsName[indexIn].split('&')[0]
            atList.remove(svcsName[indexIn].split('&')[1])

        elif takeStep > 0:
            predictIn = svcsName[indexIn].split('&')[1]
            atList.remove(svcsName[indexIn].split('&')[0])

        else:
            raise NameError('error: predict zero .')

        #  输出判断步
        log.write("Compare between " + svcsName[indexIn] + ',')
        log.write("step is " + predictIn + ';\n')

        indexIn += 1


#  ***************************************************************************s*

def prepare4train():
    ###############################################################################
    #                         dispatch task for train                             #
    ###############################################################################

    #   require num*(num-1)/2 classifiers

    #  prepare for svc nameList
    for index2In in range(num):
        for j in range(index2In + 1, num):
            svcsName.append(str(neatLabelSet_A[index2In]) + '&' + str(neatLabelSet_A[j]))

    #  thread mission dispatch

    global perCore

    perCore = int(len(svcsName) / coreNum) + 1

    for index2In in range(coreNum):
        threadMission.append([])
        threadMIndex.append([])

    indexIn = 0

    for index2In in range(len(svcsName)):
        threadMission[indexIn].append(svcsName[index2In])
        threadMIndex[indexIn].append(index2In)
        if len(threadMission[indexIn]) == perCore:
            indexIn += 1

###############################################################################
#                                 start train                                 #
###############################################################################

#  memory requirement would be huge here


if __name__ == '__main__':

    init()

    listData()

    prepare4train()

    pool = Pool(processes=coreNum)
    temp = []
    freeze_support()

    for ini in range(coreNum):
        temp.append(pool.apply_async(subProcess, (threadMission[ini], neatDataSet_A, neatDataSet_S,
                                                  neatLabelSet_A, neatLabelSet_S, ini, )))

    pool.close()
    pool.join()

    for index in range(coreNum):
        svcs[threadMIndex[index][0]:threadMIndex[index][len(threadMIndex[index]) - 1] + 1] = temp[index].get()

    ############################################################################
    #                                 test                                     #
    ############################################################################

    correct = 0

    error = np.zeros((10, 10), int)

    # test
    for ind in range(0, len(testSet)):
        prediction = predict(testSet[ind], str(testLabels[ind]))

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
