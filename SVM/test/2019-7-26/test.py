# 三分类SVM分类器
# voting model
# Using Iris.data(40 training data and 10 testing data each kind, 3 kinds totally)

from SVM.Platt_SMO import smoP
from numpy import mat

# train part
# read data
setosas = [];
versicolors = [];
virginicas = []
fr = open('/home/emh/dataset/iris/train.data')
index = 1
for line in fr.readlines():
    lineArr = line.strip().split(',')
    if 1 <= index <= 40:
        setosas.append([float(lineArr[0]), float(lineArr[1]), float(lineArr[2]), float(lineArr[3])])
    elif 41 <= index <= 80:
        versicolors.append([float(lineArr[0]), float(lineArr[1]), float(lineArr[2]), float(lineArr[3])])
    else:
        virginicas.append([float(lineArr[0]), float(lineArr[1]), float(lineArr[2]), float(lineArr[3])])
    index += 1

C = 0.0001
toler = 0.1
maxIter = 40

labels = 40 * [-1] + 40 * [1]
# setosa and versicolor
dataMatIn = setosas.copy() + versicolors.copy()
w_sve, b_sve = smoP(dataMatIn, labels, C, toler, maxIter)

# versicolor and virginica
dataMatIn = versicolors.copy() + virginicas.copy()
w_vevi, b_vevi = smoP(dataMatIn, labels, C, toler, maxIter)

# setosa and virginica
dataMatIn = setosas.copy() + virginicas.copy()
w_svi, b_svi = smoP(dataMatIn, labels, C, toler, maxIter)


def test(dataArr):
    setosa = 0
    versicolor = 0
    virginica = 0

    # vote
    if (mat(dataArr) * w_sve + b_sve) < 0:
        setosa += 1;versicolor -= 1
    else:
        versicolor += 1;setosa -= 1

    if (mat(dataArr) * w_vevi + b_vevi) < 0:
        versicolor += 1;virginica -= 1
    else:
        virginica += 1;versicolor -= 1

    if (mat(dataArr) * w_svi + b_svi) < 0:
        setosa += 1;virginica -= 1
    else:
        virginica += 1;setosa -= 1

    """print('Result of vote: ')
    print('setosa: ',end='');print(setosa)
    print('versicolor: ', end='');print(versicolor)
    print('virginica: ', end='');print(virginica)
    print()"""

    if setosa > versicolor and setosa > virginica:
        # print('Result of vote: setosa')
        return 1
    elif versicolor > setosa and versicolor > virginica:
        # print('Result of vote: versicolor')
        return 2
    elif virginica > setosa and virginica > versicolor:
        # print('Result of vote: virginica')
        return 3
    else:
        print('Deny vote')
        return 0


result = 0
index = 1;
fr = open('/home/emh/dataset/iris/test.data')
for line in fr.readlines():
    lineArr = line.strip().split(',')
    dataArr = [float(lineArr[0]), float(lineArr[1]), float(lineArr[2]), float(lineArr[3])]
    # print('Name: ',end='');print(lineArr[4].split('-')[1])
    name = 0
    if lineArr[4].split('-')[1] == 'setosa':
        name = 1
    elif lineArr[4].split('-')[1] == 'versicolor':
        name = 2
    else:
        name = 3

    if test(dataArr) == name:
        print(index)
        result += 1
    index += 1

print(result / 30)
