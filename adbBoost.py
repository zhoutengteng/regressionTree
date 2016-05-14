#coding=utf-8
import numpy as np
import loadData
import copy
import operator
import  plotMap
import matplotlib.pyplot as plt

def getMapTable():
    table = {0:"x", 1:"y"}
    return table

def getMapTableReverse():
    table = {"x":0, "y":1}
    return table

def getClassLabel(dataSet):
    classLabel = []
    for i in range(len(dataSet)):
        classLabel.append(dataSet[i, -1])
    classLabel = list(set(classLabel))
    return classLabel



def stumpClassify(dataSet, feature, ineq,  value):
    retArray = np.ones((np.shape(dataSet)[0], 1))
    if ineq == 'lt':
        retArray[dataSet[:, feature] <= value] = -1
    else:
        retArray[dataSet[:, feature] > value] = -1
    return retArray

def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def chooseBestSplitRetTree(dataSet, D):
    splitLabel = None
    splitValue = None
    table = getMapTable()
    numFeatures = len(dataSet[0]) - 1
    #print numFeatures
    alpha = None
    minARR = None
    bestLeftLabel = None
    minErr = 100000
    for i in range(numFeatures):
        rangeMin = dataSet[:, i].min()
        rangeMax = dataSet[:, i].max()
        stepNum = 2
        stepSize = (rangeMax - rangeMin) / float(rangeMax - rangeMin)
        for j in range(rangeMin, rangeMax+1):
            featureValue = j
            # featureValue = rangeMin + (j) * stepSize
            for ineq in ["lt", 'gt']:
                predictValue = None
                predictValue = stumpClassify(dataSet,i, ineq, featureValue)
                errArr = np.ones((len(dataSet), 1))
                yLabel = np.array([example[-1] for example in dataSet])
                yLabel = yLabel.reshape((len(yLabel), 1))
                errArr[yLabel == predictValue] = 0
                err = np.dot(D.T, errArr)
                if  err < minErr:
                    splitValue = featureValue
                    splitLabel = table[i]
                    minErr = err[0][0]
                    minARR = errArr
                    bestLeftLabel = ineq


    alpha = float(0.5 * np.log((1.0 - minErr) / max(minErr, 1e-16)))
    minARR[minARR == np.ones((len(minARR), 1))] = -1
    minARR[minARR == np.ones((len(minARR), 0))] = 1
    expon = -1 * alpha * minARR
    D = np.multiply(D, np.exp(expon))
    D = D / (D.sum() + 0.0)
    tree = {}
    #print alpha, minErr
    tree[splitLabel] = {}
    if bestLeftLabel == 'lt':
        tree[splitLabel]['left<=' + str(splitValue)] = -1
        tree[splitLabel]['right>' + str(splitValue)] = 1
    else:
        tree[splitLabel]['left<=' + str(splitValue)] = 1
        tree[splitLabel]['right>' + str(splitValue)] = -1
    tree[splitLabel]['alpha'] = alpha
    return tree, D



def plotAddboost(tree, dataSet, D, count):
    plt.figure("count==>"+str(count))
    dataSet = np.array(dataSet)
    colors = {-1:"red", 1:"green"}
    for i in range(0,len(dataSet)):
        yLabel = dataSet[i, -1]
        yTest = predictByOneTree(dataSet[i], tree)
        plt.scatter([dataSet[i,0]], [dataSet[i,1]], color=colors[yLabel], s=int(D[i] * 10000))
        if yLabel != yTest:
            plt.plot([dataSet[i,0], dataSet[i,0], dataSet[i,0]], [dataSet[i,1] + 1, dataSet[i,1], dataSet[i,1] - 1])
            plt.plot([dataSet[i,0]+1, dataSet[i,0], dataSet[i,0] - 1], [dataSet[i, 1], dataSet[i, 1], dataSet[i,1]])
    print tree
    plt.show()


def predictByOneTree(X, tre):
    reverseTable = getMapTableReverse()
    tree = copy.deepcopy(tre)
    while (isinstance(tree, dict)):
        #print tree
        key = tree.keys()[0]
        index = reverseTable[key]
        indexValue = X[index]
        subTree = tree[key]
        subTreeLeftKey = None
        subTreeRightKey = None
        for key in subTree.keys():
            if key.find("<=") != -1:
                subTreeLeftKey = key
            if key.find(">") != -1:
                subTreeRightKey = key
        if float(indexValue) <= float(subTreeLeftKey.split("<=")[1]):
            tree = subTree[subTreeLeftKey]
        else:
            tree = subTree[subTreeRightKey]
    return tree


def predictFin(X, weakClassArr):
    reverseTable = getMapTableReverse()
    resu = 0
    for weakClass in weakClassArr:
        key = weakClass.keys()[0]
        alpha = weakClass[key]['alpha']
        index = reverseTable[key]
        indexValue = X[index]
        subTree = weakClass[key]
        subTreeLeftKey = None
        subTreeRightKey = None
        for key in subTree.keys():
            if key.find("<=") != -1:
                subTreeLeftKey = key
            if key.find(">") != -1:
                subTreeRightKey = key
        if float(indexValue) <= float(subTreeLeftKey.split("<=")[1]):
            tree = subTree[subTreeLeftKey]
        else:
            tree = subTree[subTreeRightKey]
        resu += alpha * tree
    return np.sign(resu)

def countErr(X, weakClassArr):
    m,n = np.shape(X)
    error = 0
    for x in X:
        y = x[-1]
        yTest = predictFin(x, weakClassArr)
        if y != yTest:
            error += 1
    print "error rate=>", error / (m + 0.0) * 100, "%"

def adbBoost():
    weakClassArr = []
    dataSet = loadData.produceDataBin()
    m,n = np.shape(dataSet)
    D = np.ones((m, 1)) / float(m)
    T = 10
    for treei in range(0,T):
        tree, D = chooseBestSplitRetTree(dataSet, copy.deepcopy(D))
        #plotAddboost(tree, dataSet, D, treei)
        weakClassArr.append(tree)
    countErr(dataSet, weakClassArr)
    #plotMap.plotMap(dataSet)
    #print weakClassArr