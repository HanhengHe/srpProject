#三分类SVM分类器
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

kTup=('rbf',1.5)
C=0.8
toler=0.001
maxIter=40

labels=40*[-1]+40*[1]
#setosa and versicolor
dataMatIn_sve=setosas.copy()+versicolors.copy()
alphas_sve,b_sve=smoP(dataMatIn_sve,labels,C,toler,maxIter,kTup)

svInd_sve=nonzero(alphas_sve.A>0)[0]
sVs_sve=mat(dataMatIn_sve)[svInd_sve]
labelSV_sve=mat(labels.copy()).transpose()[svInd_sve]

#versicolor and virginica
dataMatIn_vevi=versicolors.copy()+virginicas.copy()
alphas_vevi,b_vevi=smoP(dataMatIn_vevi,labels,C,toler,maxIter,kTup)

svInd_vevi=nonzero(alphas_vevi.A>0)[0]
sVs_vevi=mat(dataMatIn_vevi)[svInd_vevi]
labelSV_vevi=mat(labels.copy()).transpose()[svInd_vevi]

#setosa and virginica
dataMatIn_svi=setosas.copy()+virginicas.copy()
alphas_svi,b_svi=smoP(dataMatIn_svi,labels,C,toler,maxIter,kTup)

svInd_svi=nonzero(alphas_svi.A>0)[0]
sVs_svi=mat(dataMatIn_svi)[svInd_svi]
labelSV_svi=mat(labels.copy()).transpose()[svInd_svi]

def test(dataArr):
    setosa=0;versicolor=0;virginica=0

    #vote
    kernelEval = kernelTrans(sVs_sve, dataArr, kTup)
    predict = kernelEval.T * multiply(labelSV_sve, alphas_sve[svInd_sve]) + b_sve
    if predict<0:setosa-=predict
    else:versicolor+=predict

    kernelEval = kernelTrans(sVs_vevi, dataArr, kTup)
    predict = kernelEval.T * multiply(labelSV_vevi, alphas_vevi[svInd_vevi]) + b_vevi
    if predict < 0:versicolor-=predict
    else:virginica+=predict

    kernelEval = kernelTrans(sVs_svi, dataArr, kTup)
    predict = kernelEval.T * multiply(labelSV_svi, alphas_svi[svInd_svi]) + b_svi
    if predict < 0:setosa-=predict
    else:virginica+=predict

    if setosa>versicolor and setosa>virginica:
        return 1
    elif versicolor>setosa and versicolor>virginica:
        return 2
    elif virginica>setosa and virginica>versicolor:
        return 3
    else:
        print('Deny vote')
        return 0

result=0
index=1
fr=open('/home/emh/dataset/iris/test.data')
for line in fr.readlines():
    lineArr=line.strip().split(',')
    dataArr=[float(lineArr[0]), float(lineArr[1]), float(lineArr[2]), float(lineArr[3])]
    #print('Name: ',end='');print(lineArr[4].split('-')[1])
    name=0
    if lineArr[4].split('-')[1]=='setosa':
        name=1
    elif lineArr[4].split('-')[1]=='versicolor':
        name=2
    else:
        name=3

    if test(dataArr)==name:
        print(index)
        result += 1
    index+=1

print(result/30)