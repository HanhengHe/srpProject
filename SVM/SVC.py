# -*- coding: UTF-8 -*-
import random

import numpy as np


class SVC:
    def __init__(self, dataMat, labelMat, kTup, tol, C, cWeight):
        self.dataMat = dataMat
        self.labelMat = labelMat
        self.C = C
        self.tol = tol
        self.kTup = kTup
        self.b = 0
        self.m, self.n = np.shape(dataMat)
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.ay = np.mat(np.zeros((self.m, 1)))

        if cWeight is None:
            self.cWeight = [1] * self.m
        elif len(cWeight) != self.m:
            raise NameError("Error on C weight")
        else:
            self.cWeight = cWeight * self.m

        self.K = np.mat(np.zeros((self.m, self.m), dtype=int))
        print('Init dataMat')
        for i in range(self.m):
            print(i/self.m)
            self.K[:, i] = self.kernelTrans(self.dataMat[i, :])

    def calcEk(self, k):
        ay = np.mat(np.zeros((self.m, 1)))
        for i in range(self.m):
            ay[i, 0] = self.alphas[i, 0] * self.labelMat[i, 0]
        fXk = float(ay.T * (self.K[:, k]))
        Ek = fXk - float(self.labelMat[k])
        return Ek

    def calcB(self):
        ay = np.mat(np.zeros((self.m, 1)))
        for i in range(self.m):
            ay[i, 0] = self.alphas[i, 0] * self.labelMat[i, 0]

        #  save ay for further calculation
        self.ay = ay

        bs = 0.0
        counter = 0
        for k in range(self.m):
            if 0 < self.alphas[k] < self.C * self.cWeight[k]:
                bs += (self.labelMat[k, 0] - ay.T * self.K[:, k])
                counter += 1

        return bs / counter

    def kernelTrans(self, A):
        K = np.mat(np.zeros((self.m, 1)))
        if self.kTup[0] == 'lin':
            K = self.dataMat * np.mat(A).T
        elif self.kTup[0] == 'rbf':
            for j in range(self.m):
                deltaRow = self.dataMat[j, :] - A
                K[j] = deltaRow * deltaRow.T
            K = np.exp(K / (-2 * self.kTup[1] ** 2))
        else:
            raise NameError('error: check ur kernel order.')
        return K

    def takeStep(self, i, Ei, j, Ej):
        if i == j:
            return 0

        alphaI = self.alphas[i].copy()
        alphaJ = self.alphas[j].copy()

        #  alphaIOld = self.alphas[i].copy()
        alphaJOld = self.alphas[j].copy()

        yI = self.labelMat[i]
        yJ = self.labelMat[j]

        # get L and H
        if yI != yJ:
            L = max(0, alphaJ - alphaI)
            H = min(self.C * self.cWeight[j], self.C * self.cWeight[j] + alphaJ - alphaI)
        else:
            L = max(0, alphaJ + alphaI - self.C * self.cWeight[j])
            H = min(self.C * self.cWeight[j], alphaJ + alphaI)
        if L == H:
            print("H==L!")
            return False

        eta = self.K[i, i] + self.K[j, j] - 2 * self.K[i, j]

        if eta <= 0:
            print('eta<=0')
            return False

        alphaJ = alphaJ + yJ * (Ei - Ej) / eta
        if alphaJ < L:
            alphaJ = L
        elif alphaJ > H:
            alphaJ = H

        alphaI += yJ.T * yI * (alphaJOld - alphaJ)

        self.alphas[i] = alphaI
        self.alphas[j] = alphaJ

        return True

    def selectJ(self, i, Ei):
        maxK = -1
        maxDeltaE = 0
        Ej = 0
        varList = []
        for i in range(self.m):
            if 0 < self.alphas[i] < self.C * self.cWeight[i]:
                varList.append(i)

        if (len(varList)) > 1:
            for k in varList:
                if k == i:
                    continue
                Ek = self.calcEk(k)
                deltaE = abs(Ei - Ek)
                if deltaE > maxDeltaE:
                    maxK = k
                    maxDeltaE = deltaE
                    Ej = Ek
            return maxK, Ej
        else:
            j = selectJRand(i, self.m)
            Ej = self.calcEk(j)
        return j, Ej

    def examineExample(self, i):
        y = self.labelMat[i]
        Ei = self.calcEk(i)
        r = Ei * y
        if (r < -self.tol and self.alphas[i] < self.C * self.cWeight[i]) or (r > self.tol and self.alphas[i] > 0):
            j, Ej = self.selectJ(i, Ei)
            if self.takeStep(i, Ei, j, Ej):
                return 1
        return 0

    def predict(self, x):  # x should be a list
        if not isinstance(x, list):
            raise NameError('error: x should be a list.')
        if len(x) != self.n:
            raise NameError('error: bad len(x)')

        wx = float(self.ay.T * self.kernelTrans(x))

        return wx + self.b


def selectJRand(i, m):
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j


# main loop
def svc(dataMat, labelMat, C, tol, maxIter, kTup, cWeight=None):
    s = SVC(np.mat(dataMat), np.mat(labelMat).T, kTup, tol, C, cWeight)
    numChange = 0
    examineAll = True
    m = len(dataMat)
    iter = 0
    while (numChange > 0 or examineAll) and (iter < maxIter):
        numChange = 0

        print("Loop %s" % str(iter))

        iter += 1

        if examineAll:  # loop I over all training examples
            for i in range(m):
                numChange += s.examineExample(i)
        else:
            for i in range(m):
                if 0 < s.alphas[i] < C:
                    numChange += s.examineExample(i)

        if examineAll:
            examineAll = False
        elif numChange == 0:
            examineAll = True

    #  get b
    s.b = s.calcB()

    return s


def ReadProblem(filePath):
    dataMat = []
    labelMat = []
    file = open(filePath)
    counter = 0
    for line in file.readlines():
        lineArr = line.strip().split(' ')
        dataMat.append([float(lineArr[1]), float(lineArr[2]), float(lineArr[3]), float(lineArr[4])])
        if counter <= 39:
            labelMat.append(1)
        else:
            labelMat.append(-1)
        counter += 1
    return dataMat, labelMat
