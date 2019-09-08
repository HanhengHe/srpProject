#  test on Mnist data set
from SVM.SVC import svr

#  this is a one v one classifier test
#  my SVM.py is not finished up to now

#  读取训练集
fTrain = open('D:\\WINTER\\Pycharm_project\\data\\Mnist\\train')
trainSet = []
trainLabels = []
for line in fTrain.readlines():

    #  修改格式
    dataSet = line.split(')')[0]
    label = line.split(')')[1].replace('\n', '')

    #  只分类0和1.这是一个2分类工具
    if label != '0' and label != '1':
        continue

    dataSets = dataSet.split(',')
    dataSets[0] = dataSets[0].replace('(', '')

    #  data set调整为float类型，label调整为int类型
    temp = []
    for i in range(len(dataSets)):
        temp.append(int(dataSets[i]))

    #  置入数据结构中
    trainSet.append(temp)
    trainLabels.append(int(label))

fTrain.close()

print('Train set loaded.')

print('Start training')

#  训练分类器
s = svr(trainSet, trainLabels, 0.8, 0.01, 1000, kTup=['rbf', 0.8])

print('Trained.')

print('Start testing')

#  读取测试集
fTest = open('D:\\WINTER\\Pycharm_project\\data\\Mnist\\test')
testSet = []
testLabels = []
for line in fTest.readlines():

    #  修改格式
    dataSet = line.split(')')[0]
    label = line.split(')')[1].replace('\n', '')

    #  只分类0和1.这是一个2分类工具
    if label != '0' and label != '1':
        continue

    dataSets = dataSet.split(',')
    dataSets[0] = dataSets[0].replace('(', '')

    #  data set调整为float类型，label调整为int类型
    temp = []
    for i in range(len(dataSets)):
        temp.append(int(dataSets[i]))

    #  置入数据结构中
    testSet.append(temp)
    testLabels.append(int(label))

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

print(correct/len(testLabels))
