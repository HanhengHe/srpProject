# -*- coding: UTF-8 -*-
from SVM.SVC import svr

#   这是一个不成熟的svr多分类器

#   目前只打算写 dag 部分和 ecoc 部分
#   ovo 和 ovr 不打算做

Type = ('DAG', 'ECOC')


#   List格式：[data,...,data,label]
#   label从0开始，请不要跳过数字

def Classifier(dataList, labelList, C, tol, maxIter, kTup=('lin', 0), cWeight=0, classifierType=Type[0]):
    #   check type
    if not isinstance(dataList, list):
        raise NameError('error: dataList should be a list.')

    if not isinstance(labelList, list):
        raise NameError('error: labelList should be a list.')

    # ***************这部分代码还可以进一步优化******************
    #   计算类别数
    counter = []
    for data in dataList:
        if data[len(data) - 1] not in counter:
            counter.append((data[len(data) - 1], 1))

    num = len(counter)
    #   按标签类型重新整理数据集
    #   list嵌套list，这种方式的空间效率可能会很低，期待后续修正
    neatDataSet = []


    #   按多分类类型派发任务
    if classifierType == 'DAG':
        return dagClassifier(dataList, labelList, num, C, tol, maxIter, kTup)
    elif classifierType == 'ECOC':
        return ecocClassifier(dataList, labelList, num, C, tol, maxIter, kTup)
    else:
        raise NameError('error: classifierType error .')

    # ************************到这******************************


#   有向无环图分类器
def dagClassifier(dataList, labelList, num, C, tol, maxIter, kTup):
    #   建立 num*(num+1)/2 个分类器
    svrs = list()
    svrsName = list()

    #  prepare for data

    #  counter on label number
    counter = 0

    index = 0
    #  prepare for nameList
    for i in range(num):
        for j in range(i + 1, num):
            svrsName.append(str(i) + '&' + str(j))
            index += 1

    pass


#   ECOC 分类器
def ecocClassifier(dataList, labelList, num, C, tol, maxIter, kTup):
    #   建立 num*(num+1)/2 个分类器
    svrs = list(num * (num + 1) / 2)
    pass
