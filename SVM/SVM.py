from SVM.SVR import svr

#   这是一个不成熟的svr多分类器

Type = ('DAG', 'OVR', 'OVO')

#   List格式：[data,...,data,label]
#   label从0开始，请不要跳过数字

def Classifier(dataList, labelList, C, tol, maxIter, kTup=('lin', 0), classifierType=Type[0]):

    #   check type
    if not isinstance(dataList, list):
        raise NameError('error: dataList should be a list.')

    if not isinstance(labelList, list):
        raise NameError('error: testList should be a list.')

    #   计算类别数
    counter = []
    for data in dataList:
        if data[len(data) - 1] not in counter:
            counter.append(data[len(data) - 1])

    num = len(counter)

    #   按多分类类型派发任务
    if classifierType == 'DAG':
        return dagClassifier(dataList, labelList, num, C, tol, maxIter, kTup)
    elif classifierType == 'OVR':
        return ovrClassifier(dataList, labelList, num, C, tol, maxIter, kTup)
    elif classifierType == 'OVO':
        return ovoClassifier(dataList, labelList, num, C, tol, maxIter, kTup)

#   有向无环图分类器
def dagClassifier(dataList, labelList, num, C, tol, maxIter, kTup):
    #   建立 num*(num+1)/2 个分类器
    svrs = list()
    svrsName = list()

    index = 0
    for i in range(num):
        for j in range(i+1, num):
            svrsName.append(str(i)+'&'+str(j))
            index += 1

    pass


#   1 v Rest 分类器
def ovrClassifier(dataList, labelList, num, C, tol, maxIter, kTup):
    #   建立 num 个分类器
    svrs = list(num)
    pass


#   1 v 1分类器
def ovoClassifier(dataList, labelList, num, C, tol, maxIter, kTup):
    #   建立 num*(num+1)/2 个分类器
    svrs = list(num * (num + 1) / 2)
    pass
