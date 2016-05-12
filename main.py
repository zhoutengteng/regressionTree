#coding=utf-8
import copy
import loadData
import matplotlib.pyplot as plt
import numpy as np
import random
import plotMap
import math
import operator

#根据特征值分割成数据两部分
def binSplitDataSet(dataSet, feature, value):
    #print feature
    mat0 = dataSet[np.nonzero(dataSet[:, feature] <= value)]
    mat1 = dataSet[np.nonzero(dataSet[:, feature] > value)]
    #plotMap.plotPart2(mat0, mat1)
    # print np.shape(mat0),"--", np.shape(mat1)
    return mat0, mat1

#计算一个节点中各个类别的数量
def countEveryTyepNum(dataSet, classLabel):
    list =[]
    for label in classLabel:
        list.append(len(filter(lambda value: value[np.shape(dataSet)[1] - 1] == label, dataSet)))
    return list


def countEntropyByclassify(dataSet):
    numTotal = len(dataSet)
    classLabel = getClassLabel(dataSet)
    list = countEveryTyepNum(dataSet, classLabel)
    E = 0.0
    #print classLabel, list
    for lable in range(len(classLabel)):
        if list[lable - 1] == 0:
            continue
        p = list[lable - 1] / float(numTotal)
        E += -1 * p * np.log2(p)
    # print E
    return E

def chooseBestSplit(dataSet):
    oE = countEntropyByclassify(dataSet)
    gain = 0
    splitLabel = None
    splitValue = None
    table = getMapTable()
    classLabel = getClassLabel(dataSet)
    numFeatures = len(dataSet[0]) - 1
    for i in range(numFeatures):
        # featList = [value[i] for value in dataSet]
        featListPair = [(value[i], value[-1]) for value in dataSet]
        sorted(featListPair)
        sign = None
        featList = []
        for pair in featListPair:
            if sign == None:
                sign = pair[1]
                featList.append(pair[0])
            elif sign != pair[1]:
                sign = pair[1]
                featList.append(pair[0])
        for featureValue in featList:
                left, right = binSplitDataSet(dataSet, i, featureValue)
                E = countEntropyByclassify(left) + countEntropyByclassify(right)
                if oE - E > gain:
                    splitValue = featureValue
                    splitLabel = table[i]
                    gain = oE - E
    return splitLabel,splitValue

def getGainThreshold():
    return 1

def getDeepThreshold():
    return 5


def getDeep(tree):
    if not isinstance(tree, dict):
        return 0
    tree = copy.deepcopy(tree)
    maxDeep = 0
    for key in tree.keys():
        thisDeep = 1 +  getDeep(tree[key])
        if thisDeep > maxDeep : maxDeep = thisDeep
    return maxDeep


def writeData(dataSet, tree):
    fr = open("result.csv", 'w')
    for data in dataSet:
        fr.write(str(data) + "\n")
    fr.write(str(tree))
    fr.close()

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

#投票机制处理
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet):
    maxDeep = 0
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0], 0
    #处理当限制深度和没有特征可分的时候,还没有完成分类,采用分类的方法实现(回归树不会用到这里)
    if len(dataSet[0]) == 1:
        return majorityCnt(classList), 0
    reverseTable = getMapTableReverse()
    bestLabel, bestValue = chooseBestSplit(dataSet)
    if bestLabel == None:
        return majorityCnt(classList), 0
    #print额 bestLabel
    #print dataSet
    left, right = binSplitDataSet(dataSet, reverseTable[bestLabel], bestValue)
    myTree = {bestLabel:{}}
    thisDeep = 2
    myTree[bestLabel]["left<=" + str(bestValue)], deep = createTree(left)
    thisDeep += deep
    if maxDeep < thisDeep: maxDeep = thisDeep

    thisDeep = 2
    myTree[bestLabel]["right>" + str(bestValue)], deep = createTree(right)
    thisDeep += deep
    if maxDeep < thisDeep: maxDeep = thisDeep
    print myTree, maxDeep
    return myTree, maxDeep


def predict(X, tre):
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

def costError(dataSet, tre):
    dataSet = np.array(dataSet)
    tree = copy.deepcopy(tre)
    error = 0.0
    for i in range(len(dataSet)):
        label = dataSet[i, -1].reshape(1,1)
        list = dataSet[i, 0:-1].reshape(len(dataSet[i])-1, 1)
        yTest = predict(list, copy.deepcopy(tree))
        if abs(label[0][0] - yTest) > 0.000001:
            error += 1
            print list.T, label[0][0], yTest
    # if error != 0:
    #     print tre
    print "分类错误率=>",error /  len(dataSet) * 100,"%"

if  __name__ == "__main__":
    dataSet = loadData.produceData()
    myTree, deep = createTree(dataSet)
    print deep
    costError(dataSet, myTree)
    plotMap.plotMap(dataSet)
    writeData(dataSet, myTree)
    print getDeep(myTree)