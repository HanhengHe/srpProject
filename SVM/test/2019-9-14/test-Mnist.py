# -*- coding: UTF-8 -*-
from SVM.SVM import *
import numpy as np

trainFile = open('D:\\WINTER\\Pycharm_project\\data\\Mnist\\train')

trainSize = 50
testSize = 10

trainSet = []

trainCounter = [
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

for line in trainFile.readlines():
    #  修改格式
    dataSet = line.split(')')[0]
    label = line.split(')')[1].replace('\n', '')

    # 计数
    #  ****************从这开始*********************
    if trainCounter[int(label)][1] == trainSize:
        continue

    trainCounter[int(label)][1] = trainCounter[int(label)][1]+1

    #  ******************到这***********************

    dataSets = dataSet.split(',')
    dataSets[0] = dataSets[0].replace('(', '')

    #  data set调整为float类型，label调整为int类型
    temp = []
    for i in range(len(dataSets)):
        temp.append(int(dataSets[i]))

    temp.append(label)

    #  置入数据结构中
    trainSet.append(temp)

classifier = Classifier(trainSet, 0.7, 0.01, 20, ['lin', 0.8])

testFile = open('D:\\WINTER\\Pycharm_project\\data\\Mnist\\test')

testSet = []
testLabels = []

for line in testFile.readlines():
    #  修改格式
    dataSet = line.split(')')[0]
    label = line.split(')')[1].replace('\n', '')

    # 计数
    #  ****************从这开始*********************
    if testCounter[int(label)][1] == trainSize:
        continue

    testCounter[int(label)][1] = testCounter[int(label)][1] + 1

    #  ******************到这***********************

    dataSets = dataSet.split(',')
    dataSets[0] = dataSets[0].replace('(', '')

    #  data set调整为float类型，label调整为int类型
    temp = []
    for i in range(len(dataSets)):
        temp.append(int(dataSets[i]))

    #  置入数据结构中
    testSet.append(temp)
    testLabels.append(label)

correct = 0

error = np.zeros((10, 10), int)

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
print(trainCounter)

print("test data set situation: ")
print(testCounter)

print("Correct present: ")
print(correct/len(testLabels))
print(error)