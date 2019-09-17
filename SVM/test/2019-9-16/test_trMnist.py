# -*- coding: UTF-8 -*-
from SVM.transfrom.trSVM import *
import numpy as np

trainFile = open('D:\\WINTER\\Pycharm_project\\data\\Mnist\\train')

trainASize = 50
SourceSize = 15
trainSSize = 5
testSize = 10

trainSetA = []
SourceSet = []

trainSetS = []
testSet = []
testLabels = []

trainSCounter = [
    [1, 0],
    [2, 0],
    [3, 0],
    [4, 0],
    [5, 0],
    [6, 0],
    [7, 0],
    [8, 0],
    [9, 0],
    [0, 0],
]

SourceCounter = [
    [1, 0],
    [2, 0],
    [3, 0],
    [4, 0],
    [5, 0],
    [6, 0],
    [7, 0],
    [8, 0],
    [9, 0],
    [0, 0],
]

trainACounter = [
    [1, 0],
    [2, 0],
    [3, 0],
    [4, 0],
    [5, 0],
    [6, 0],
    [7, 0],
    [8, 0],
    [9, 0],
    [0, 0],
]

testCounter = [
    [1, 0],
    [2, 0],
    [3, 0],
    [4, 0],
    [5, 0],
    [6, 0],
    [7, 0],
    [8, 0],
    [9, 0],
    [0, 0],
]

#   assistant data
for line in trainFile.readlines():
    #  修改格式
    dataSet = line.split(')')[0]
    label = line.split(')')[1].replace('\n', '')

    # 计数
    #  ****************从这开始*********************
    if trainACounter[int(label)][1] == trainASize:
        break

    trainACounter[int(label)][1] = trainACounter[int(label)][1] + 1

    #  ******************到这***********************

    dataSets = dataSet.split(',')
    dataSets[0] = dataSets[0].replace('(', '')

    #  data set调整为float类型，label调整为int类型
    temp = []
    for i in range(len(dataSets)):
        temp.append(int(dataSets[i]))

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
        break

    SourceCounter[int(label)][1] = SourceCounter[int(label)][1] + 1

    #  ******************到这***********************

    dataSets = dataSet.split(',')
    dataSets[0] = dataSets[0].replace('(', '')

    #  data set调整为float类型，label调整为int类型
    temp = []
    for i in range(len(dataSets)):
        temp.append(int(dataSets[i]))

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
        testSet.append(SourceSet[:len(SourceSet)-1])
        testLabels.append(SourceSet[len(SourceSet)-1])

print(len(trainSetA))
print(len(trainSetS))
print(len(testSet))
print(len(SourceSet))


classifier = Classifier(trainSetA, SourceSet, 0.7, 0.01, 20, ['lin', 0.8])

correct = 0

error = np.zeros((10, 10), int)

# test
for index in range(0, len(testSet)):
    predict = classifier.predict(testSet[index])

    print(index, end='')
    print(" predict is ", end='')
    print(predict, end=';')
    print(" real label is ", end=testLabels[index]+";")

    if predict == testLabels[index]:
        correct += 1
        print('predict is right.')
    else:
        error[int(testLabels[index]), int(predict)] = error[int(testLabels[index]), int(predict)]+1
        print('predict is wrong.')

print("train data set situation: ")
print(trainSCounter)

print("test data set situation: ")
print(testCounter)

print("Correct present: ")
print(correct/len(testLabels))
print(error)
