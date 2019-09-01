#三分类SVM分类器,1 vs rest
#voting model
#Using Iris.data(40 training data and 10 testing data each kind, 3 kinds totally)

from SVM.Platt_SMO import smoP,kernelTrans
from numpy import multiply,mat, nonzero

#train part
#read data
setosas=[];versicolors=[];virginicas=[]
fr=open('/home/emh/dataset/iris/train.data')
index=1
for line in fr.readlines():
    lineArr=line.strip().split(',')
    if index>=1 and index<=40:
        setosas.append([float(lineArr[0]), float(lineArr[1]), float(lineArr[2]), float(lineArr[3])])
    elif index>=41 and index<=80:
        versicolors.append([float(lineArr[0]), float(lineArr[1]), float(lineArr[2]), float(lineArr[3])])
    else:
        virginicas.append([float(lineArr[0]), float(lineArr[1]), float(lineArr[2]), float(lineArr[3])])
    index+=1

kTup=('rbf',2.6)
C=0.8
toler=0.001
maxIter=40

kTup1=('rbf',0.5)
C1=0.9
toler1=0.001
maxIter1=15

#setosa
labels=40*[-1]+80*[1]
dataMatIn_sve=setosas.copy()+versicolors.copy()+virginicas.copy()
alphas_sve,b_sve=smoP(dataMatIn_sve,labels,C,toler,maxIter,kTup)

svInd_sve=nonzero(alphas_sve.A>0)[0]
sVs_sve=mat(dataMatIn_sve)[svInd_sve]
labelSV_sve=mat(labels.copy()).transpose()[svInd_sve]

#versicolor and virginicas
labels=40*[-1]+40*[1]
dataMatIn_vevi=versicolors.copy()+virginicas.copy()
alphas_vevi,b_vevi=smoP(dataMatIn_vevi,labels,C1,toler1,maxIter1,kTup1)

svInd_vevi=nonzero(alphas_vevi.A>0)[0]
sVs_vevi=mat(dataMatIn_vevi)[svInd_vevi]
labelSV_vevi=mat(labels.copy()).transpose()[svInd_vevi]


def test(dataArr):
    setosa=0;versicolor=0;virginica=0

    kernelEval = kernelTrans(sVs_sve, dataArr.copy(), kTup)
    predict = kernelEval.T * multiply(labelSV_sve, alphas_sve[svInd_sve]) + b_sve
    if predict<0:print('one');return 1
    else:
        kernelEval = kernelTrans(sVs_vevi, dataArr.copy(), kTup1)
        predict = kernelEval.T * multiply(labelSV_vevi, alphas_vevi[svInd_vevi]) + b_vevi
        print(predict)
        if predict < 0:
            print('two')
            return 2
        else:
            print('three')
            return 3

result=0
f=0
index=1
fr=open('/home/emh/dataset/iris/test.data')
for line in fr.readlines():
    lineArr=line.strip().split(',')
    dataArr=[float(lineArr[0]), float(lineArr[1]), float(lineArr[2]), float(lineArr[3])]
    if lineArr[4].split('-')[1]=='setosa':
        name=1
    elif lineArr[4].split('-')[1]=='versicolor':
        name=2
    else:
        name=3

    predict=test(dataArr)
    if predict==1 and name==1:f+=1
    elif predict!=1 and name!=1:f+=1

    if predict==name:
        print(index)
        result += 1
    index+=1

print()
print(f/30)
print()
print(result/30)