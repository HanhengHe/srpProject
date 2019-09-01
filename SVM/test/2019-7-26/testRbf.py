from SVM.Platt_SMO import smoP,kernelTrans
from numpy import mat,nonzero,multiply

dataset=[];labels=45*[-1]+45*[1]
fr=open('/home/emh/dataset/iris/train.data')
index=1
for line in fr.readlines():
    data=line.strip().split(',')
    if 0<=index<46:index+=1;continue
    dataset.append([float(data[0]),float(data[1]),float(data[2]),float(data[3])])
    index+=1

kTup=('rbf',0.9)
C=0.9
toler=0.001
maxIter=40

alphas,b=smoP(dataset,labels,C,toler,maxIter,kTup)

svInd=nonzero(alphas.A>0)[0]
sVs=mat(dataset)[svInd]
labelSV=mat(labels).transpose()[svInd]

testset=[];testlabels=5*[-1]+5*[1]
fre=open('/home/emh/dataset/iris/test.data')
index=1
for line in fre.readlines():
    data=line.strip().split(',')
    if 0<=index<11:index+=1;continue
    testset.append([float(data[0]),float(data[1]),float(data[2]),float(data[3])])
    index+=1

result=0
i=0
for dataArr in testset:
    kernelEval=kernelTrans(sVs,testset[i],kTup)
    predict=kernelEval.T*multiply(labelSV,alphas[svInd])+b
    if testlabels[i]*(predict/abs(predict))==1:
        result = result + 1
        print(i+1)
    i+=1

print(result/10)