# -*- coding: UTF-8 -*-
from SVM.SVM import *

trainFile = open('D:\\WINTER\\Pycharm_project\\data\\Iris\\iris')

trainList = []

for line in trainFile.readlines():
    data = line.strip().split(' ')
    trainList.append([float(data[1]), float(data[2]), float(data[3]), float(data[4]), str(data[5])])

classifier = Classifier(trainList, 0.8, 0.01, 20, ['rbf', 0.5])

testFile = open('D:\\WINTER\\Pycharm_project\\data\\Iris\\iris_test')

testList = []
testLabels = []

for line in testFile.readlines():
    data = line.strip().split(' ')
    testList.append([float(data[1]), float(data[2]), float(data[3]), float(data[4])])
    testLabels.append(str(data[5]))

correct = 0

for index in range(0, len(testList)):
    predict = classifier.predict(testList[index])

    print(index, end='')
    print(" predict is ", end='')
    print(predict, end=';')

    if predict == testLabels[index]:
        correct += 1
        print('predict is right.')
    else:
        print('predict is wrong.')

print("Correct present: ")
print(correct/len(testLabels))
