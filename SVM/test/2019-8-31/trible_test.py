from SVM.SVC import svr
from numpy import mat

#  三分类
#  分类方法：有向无环图（DirectedAcyclic Graph）

dataMat = []  # 训练集
testMat = []  # 测试集
file = open('D:\\PycharmProjects\\srpProject\\data\\iris')
counter = 0

# 统一选取前40个作为训练集，后10个作为测试集

# 读取训练集
for line in file.readlines():
    lineArr = line.strip().split(' ')
    dataMat.append([float(lineArr[1]), float(lineArr[2]), float(lineArr[3]), float(lineArr[4])])

# 读取测试集
fre = open('D:\\PycharmProjects\\srpProject\\data\\iris_test')

for line in fre.readlines():
    data = line.strip().split(' ')
    testMat.append([float(data[1]), float(data[2]), float(data[3]), float(data[4])])

#  训练 (n-1)*n/2 = 3个分类器

#  配置参数
C = 0.8
tol = 0.01
maxIter = 30
kTup = ['rbf', 0.5]

#  setosa与versicolor
svm_sve = svr(dataMat[0:80], [-1] * 40 + [1] * 40, C, tol, maxIter, kTup)

#  setosa与virginica
svm_svi = svr(dataMat[0:40] + dataMat[81:120], [-1] * 40 + [1] * 40, C, tol, maxIter, kTup)

#  versicolor与virginica
svm_vevi = svr(dataMat[41:120], [-1] * 40 + [1] * 40, C, tol, maxIter, kTup)

result = 0
i = 0

for dataArr in testMat:

    #  回答 setosa还是versicolor

    pri1 = svm_sve.predict(dataArr)

    if pri1 < 0:  # 认为不是versicolor

        print("predict not versicolor ", end=' ')

        #  回答setosa还是virginica
        print(i, end=': ')
        print("predict:", end="")

        pri2 = svm_svi.predict(dataArr)

        if pri2 < 0:  # 认为是setosa
            print("setosa", end='')
            if 0 <= i < 10:
                print(",prediction RIGHT")
                result += 1
            else:
                print(",prediction WRONG")
        elif pri2 > 0:  # 认为是virginica
            print("virginica", end="")
            if 20 <= i < 30:
                print(",prediction RIGHT")
                result += 1
            else:
                print(",prediction WRONG")
        else:
            print("Error!")

    elif pri1 > 0:  # 认为不是setosa

        print("predict not setosa ", end=' ')

        #  回答versicolor还是virginica

        pri2 = svm_vevi.predict(dataArr)

        print(i, end=': ')
        print("predict:", end="")

        if pri2 < 0:  # 认为是versicolor
            print("versicolor", end="")
            if 10 <= i < 20:
                print(",prediction RIGHT")
                result += 1
            else:
                print(",prediction WRONG")
        elif pri2 > 0:  # 认为是virginica
            print("virginica", end="")
            if 20 <= i < 30:
                print(",prediction RIGHT")
                result += 1
            else:
                print(",prediction WRONG")
        else:
            print("Error!")

    else:
        print("Error!")
    i += 1

print()
print(result / 30)
