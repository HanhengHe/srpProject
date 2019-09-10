#  test on Mnist data set
from SVM.SVC import svr
import numpy as np

#  this is a one v one classifier test
#  my SVM.py is not finished up to now

#  读取训练集
fTrain = open('D:\\PycharmProjects\\DataSet\\Mnist\\train')
trainSet = []
trainLabels = []

num0 = 0
num1 = 1

for line in fTrain.readlines():

    #  修改格式
    dataSet = line.split(')')[0]
    label = line.split(')')[1].replace('\n', '')

    #  只分类0和1.这是一个2分类工具
    # ********************************
    if label != '0' and label != '1':
        continue

    if num0 + num1 == 600:
        break

    if label == '0':
        num0 += 1
    elif label == '1':
        num1 += 1
    # ********************************

    dataSets = dataSet.split(',')
    dataSets[0] = dataSets[0].replace('(', '')

    #  data set调整为float类型，label调整为int类型
    temp = []
    for i in range(len(dataSets)):
        temp.append(int(dataSets[i]))

    #  置入数据结构中
    trainSet.append(temp)
    if label == '1':
        trainLabels.append(1)
    elif label == '0':
        trainLabels.append(-1)

fTrain.close()

print('Train set loaded.')

print('Start training')

#  训练分类器
s = svr(np.mat(trainSet), trainLabels, 0.8, 0.01, 30, kTup=['lin', 0.8])

print('Trained.')

print('Start testing')

#  读取测试集
fTest = open('D:\\PycharmProjects\\DataSet\\Mnist\\test')
testSet = []
testLabels = []

num0 = 0
num1 = 1

for line in fTest.readlines():

    #  修改格式
    dataSet = line.split(')')[0]
    label = line.split(')')[1].replace('\n', '')

    #  只分类0和1.这是一个2分类工具
    # ********************************
    if label != '0' and label != '1':
        continue

    if num0 + num1 == 200:
        break

    if label == '0':
        num0 += 1
    elif label == '1':
        num1 += 1
    # ********************************

    dataSets = dataSet.split(',')
    dataSets[0] = dataSets[0].replace('(', '')

    #  data set调整为float类型，label调整为int类型
    temp = []
    for i in range(len(dataSets)):
        temp.append(int(dataSets[i]))

    #  置入数据结构中
    testSet.append(temp)

    if label == '1':
        testLabels.append(1)
    elif label == '0':
        testLabels.append(-1)

fTest.close()

correct = 0
i = 0

for testArr in testSet:

    var = s.predict(testArr)

    print('predict is ', end='')
    print(var, end='; ')
    print('reality is ', end='')
    print(testLabels[i])

    if testLabels[i] * var > 0:
        correct = correct + 1

    i += 1

print(correct / len(testLabels))
