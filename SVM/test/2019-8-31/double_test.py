from SVM.SVC import svr, ReadProblem
from numpy import mat

index = 1

dataMat, labelMat = ReadProblem('D:\\WINTER\\Pycharm_project\\MachineLearning\\data\\train')

s = svr(dataMat, labelMat, 0.8, 0.01, 20, kTup=['rbf', 0.8])

testSet = []
testLabels = 10 * [1] + 10 * [-1]

fre = open('D:\\WINTER\\Pycharm_project\\MachineLearning\\data\\test')

for line in fre.readlines():
    data = line.strip().split(' ')
    testSet.append([float(data[1]), float(data[2]), float(data[3]), float(data[4])])
    index += 1

result = 0
i = 0

for dataArr in testSet:
    print(i, end=": ")
    var = s.predict(dataArr)
    print(var)
    if testLabels[i] * var > 0:
        result = result + 1
    i += 1

print()
print(result / 20)
