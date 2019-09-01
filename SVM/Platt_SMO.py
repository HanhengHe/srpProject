import numpy as np
import random


# kernel function
def kernelTrans(X, A, kTup):
    m, n = np.shape(X)
    K = np.mat(np.zeros((m, 1)))
    if kTup[0] == 'lin':
        K = X * np.mat(A).T
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow * deltaRow.T
        K = np.exp(K / (-1 * kTup[1] ** 2))
    else:
        raise NameError('error: check ur kernel order.')
    return K


def ReadData(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat


def selectJRand(i, m):
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j


def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


# support function
class optStruct:
    def __init__(self, dataMatIn, classLabels, C, tol, kTup):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = tol
        self.m = np.shape(dataMatIn)[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m, 2)))
        self.K = np.mat(np.zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.X, self.X[i, :], kTup)

    def calcEk(self, k):
        fXk = float(np.multiply(self.alphas, self.labelMat).T * (self.K[:, k])) + self.b
        Ek = fXk - float(self.labelMat[k])
        return Ek

    """def calcEk(self, k):
        fXk = float(np.multiply(self.alphas, self.labelMat).T * (self.X*self.X[k,:].T)) + self.b
        Ek = fXk - float(self.labelMat[k])
        return Ek"""

    def selectJ(self, i, Ei):
        maxK = -1
        maxDeltaE = 0
        Ej = 0
        self.eCache[i] = [1, Ei]
        validECacheList = np.nonzero(self.eCache[:, 0].A)[0]
        if (len(validECacheList)) > 1:
            for k in validECacheList:
                if k == i: continue
                Ek = self.calcEk(k)
                deltaE = abs(Ei - Ek)
                if (deltaE > maxDeltaE):
                    maxK = k
                    maxDeltaE = deltaE
                    Ej = Ek
            return maxK, Ej
        else:
            j = selectJRand(i, self.m)
            Ej = self.calcEk(j)
        return j, Ej

    def updateEk(self, k):
        Ek = self.calcEk(k)
        self.eCache[k] = [1, Ek]

    def innerL(self, i):
        Ei = self.calcEk(i)
        if ((self.labelMat[i] * Ei < -self.tol) and (self.alphas[i] < self.C)) or \
                ((self.labelMat[i] * Ei > self.tol) and (self.alphas[i] > 0)):
            j, Ej = self.selectJ(i, Ei)
            alphaIold = self.alphas[i].copy()
            alphaJold = self.alphas[j].copy()
            if (self.labelMat[i] != self.labelMat[j]):
                L = max(0, self.alphas[j] - self.alphas[i])
                H = min(self.C, self.C + self.alphas[j] - self.alphas[i])
            else:
                L = max(0, self.alphas[j] + self.alphas[i] - self.C)
                H = min(self.C, self.alphas[j] + self.alphas[i])
            # if L==H:print("L==H");return 0
            if L == H: return 0
            """eta=2.0*self.K[i,j]-self.K[i,i]-self.K[j,j]"""
            eta = 2.0 * self.X[i, :] * self.X[j, :].T - self.X[i, :] * self.X[i, :].T - self.X[j, :] * self.X[j, :].T
            if eta >= 0: print('eta>=0');return 0
            self.alphas[j] -= self.labelMat[j] * (Ei - Ej) / eta
            self.alphas[j] = clipAlpha(self.labelMat[j], H, L)
            self.updateEk(j)
            b1 = self.b - Ei - self.labelMat[i] * (self.alphas[i] - alphaIold) * self.K[i, i] - \
                 self.labelMat[j] * (self.alphas[j] - alphaJold) * self.K[i, j]
            b2 = self.b - Ej - self.labelMat[i] * (self.alphas[i] - alphaIold) * self.K[i, j] - \
                 self.labelMat[j] * (self.alphas[j] - alphaJold) * self.K[j, j]
            """b1=self.b-Ei-self.labelMat[i]*(self.alphas[i]-alphaIold)*self.X[i,:]*self.X[i,:].T-self.labelMat[j]*\
               (self.alphas[j]-alphaJold)*self.X[i,:]*self.X[j,:].T
            b2=self.b-Ej-self.labelMat[i]*(self.alphas[i]-alphaIold)*self.X[i,:]*self.X[j,:].T-self.labelMat[j]*\
               (self.alphas[j]-alphaJold)*self.X[j,:]*self.X[j,:].T"""
            if (0 < self.alphas[i]) and (self.C > self.alphas[i]):
                self.b = b1
            elif (0 < self.alphas[j]) and (self.C > self.alphas[j]):
                self.b = b2
            else:
                self.b = (b1 + b2) / 2.0
            return 1
        else:
            return 0


# outer loop
def smoP(dataMatIn, classLabels, C, tol, maxIter, kTup=('lin', 0)):
    oS = optStruct(np.mat(dataMatIn), np.mat(classLabels).transpose(), C, tol, kTup)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:
            for i in range(oS.m):
                alphaPairsChanged += oS.innerL(i)
                # print('full set,iter: %d i: %d, pairs changed %d'%(iter,i,alphaPairsChanged))
                iter += 1
        else:
            nonBoundIs = np.nonzeros((oS.alphas.A > 0) * oS.alphas.A < C)[0]
            for i in nonBoundIs:
                alphaPairsChanged += oS.innerL(i)
                # print('non-bound,iter: %d i:%d, pairs changed %d' %(iter,i,alphaPairsChanged))
                iter += 1
            if entireSet:
                entireSet = False
            elif alphaPairsChanged == 0:
                entireSet = True
            # print('iteration number: %d' %iter)
        return calcWs(oS.alphas, dataMatIn, classLabels), oS.b
        # return oS.alphas, oS.b


def calcWs(alphas, dataArr, classLabels):
    X = np.mat(dataArr)
    labelMat = np.mat(classLabels).transpose()
    m, n = np.shape(X)
    w = np.zeros((n, 1))
    for i in range(m):
        w += np.multiply(alphas[i] * labelMat[i], X[i, :].T)
    return w
