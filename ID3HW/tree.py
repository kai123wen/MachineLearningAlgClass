# @Time    : 2017/12/26 19:15
# @Author  : Leafage
# @File    : seriesTree.py
# @Software: PyCharm
import collections
import random
import string
from math import log
import pandas as pd
import operator
from ID3HW import treePlotter


def calcShannonEnt(dataSet):
    """
    计算给定数据集的信息熵(香农熵)
    :param dataSet:
    :return:
    """
    # 计算出数据集的总数
    numEntries = len(dataSet)

    # 用来统计标签
    labelCounts = collections.defaultdict(int)

    # 循环整个数据集，得到数据的分类标签
    for featVec in dataSet:
        # 得到当前的标签
        currentLabel = featVec[-1]

        # 将对应的标签值加一
        labelCounts[currentLabel] += 1

    # 默认的信息熵
    shannonEnt = 0.0

    for key in labelCounts:
        # 计算出当前分类标签占总标签的比例数
        prob = float(labelCounts[key]) / numEntries

        # 以2为底求对数
        shannonEnt -= prob * log(prob, 2)

    return shannonEnt


def splitDataSetForSeries(dataSet, axis, value):
    """
    按照给定的数值，将数据集分为不大于和大于两部分
    :param dataSet: 要划分的数据集
    :param i: 特征值所在的下标
    :param value: 划分值
    :return:
    """
    # 用来保存不大于划分值的集合
    eltDataSet = []
    # 用来保存大于划分值的集合
    gtDataSet = []
    # 进行划分，保留该特征值
    for feat in dataSet:
        if feat[axis] <= value:
            eltDataSet.append(feat)
        else:
            gtDataSet.append(feat)

    return eltDataSet, gtDataSet


def splitDataSet(dataSet, axis, value):
    """
    按照给定的特征值，将数据集划分
    :param dataSet: 数据集
    :param axis: 给定特征值的坐标
    :param value: 给定特征值满足的条件，只有给定特征值等于这个value的时候才会返回
    :return:
    """
    # 创建一个新的列表，防止对原来的列表进行修改
    retDataSet = []

    # 遍历整个数据集
    for featVec in dataSet:
        # 如果给定特征值等于想要的特征值
        if featVec[axis] == value:
            # 将该特征值前面的内容保存起来
            reducedFeatVec = featVec[:axis]
            # 将该特征值后面的内容保存起来，所以将给定特征值给去掉了
            reducedFeatVec.extend(featVec[axis + 1:])
            # 添加到返回列表中
            retDataSet.append(reducedFeatVec)

    return retDataSet


def calcInfoGainForSeries(dataSet, i, baseEntropy):
    """
    计算连续值的信息增益
    :param dataSet:整个数据集
    :param i: 对应的特征值下标
    :param baseEntropy: 基础信息熵
    :return: 返回一个信息增益值，和当前的划分点
    """

    # 记录最大的信息增益
    maxInfoGain = 0.0

    # 最好的划分点
    bestMid = -1

    # 得到数据集中所有的当前特征值列表
    featList = [example[i] for example in dataSet]

    # 得到分类列表
    classList = [example[-1] for example in dataSet]

    dictList = dict(zip(featList, classList))

    # 将其从小到大排序，按照连续值的大小排列
    sortedFeatList = sorted(dictList.items(), key=operator.itemgetter(0))

    # 计算连续值有多少个
    numberForFeatList = len(sortedFeatList)

    # 计算划分点，保留三位小数
    midFeatList = [round((sortedFeatList[i][0] + sortedFeatList[i + 1][0]) / 2.0, 3) for i in
                   range(numberForFeatList - 1)]

    # 计算出各个划分点信息增益
    for mid in midFeatList:
        # 将连续值划分为不大于当前划分点和大于当前划分点两部分
        eltDataSet, gtDataSet = splitDataSetForSeries(dataSet, i, mid)

        # 计算两部分的特征值熵和权重的乘积之和

        newEntropy = len(eltDataSet) / len(dataSet) * calcShannonEnt(eltDataSet) + len(gtDataSet) / len(
            dataSet) * calcShannonEnt(gtDataSet)

        # 计算出信息增益
        infoGain = baseEntropy - newEntropy

        # print('当前划分值为：' + str(mid) + '，此时的信息增益为：' + str(infoGain))
        if infoGain > maxInfoGain:
            bestMid = mid
            maxInfoGain = infoGain

    return maxInfoGain, bestMid


def calcInfoGain(dataSet, featList, i, baseEntropy):
    """
    计算信息增益
    :param dataSet: 数据集
    :param featList: 当前特征列表
    :param i: 当前特征值下标
    :param baseEntropy: 基础信息熵
    :return:
    """
    # 将当前特征唯一化，也就是说当前特征值中共有多少种
    uniqueVals = set(featList)

    # 新的熵，代表当前特征值的熵
    newEntropy = 0.0

    # 遍历现在有的特征的可能性
    for value in uniqueVals:
        # 在全部数据集的当前特征位置上，找到该特征值等于当前值的集合
        subDataSet = splitDataSet(dataSet=dataSet, axis=i, value=value)
        # 计算出权重
        prob = len(subDataSet) / float(len(dataSet))
        # 计算出当前特征值的熵
        newEntropy += prob * calcShannonEnt(subDataSet)

    # 计算出“信息增益”
    infoGain = baseEntropy - newEntropy

    return infoGain


def chooseBestFeatureToSplit(dataSet, labels):
    """
    选择最好的数据集划分特征，根据信息增益值来计算，可处理连续值
    :param dataSet:
    :return:
    """
    # 得到数据的特征值总数
    numFeatures = len(dataSet[0]) - 1

    # 计算出基础信息熵
    baseEntropy = calcShannonEnt(dataSet)

    # 基础信息增益为0.0
    bestInfoGain = 0.0

    # 最好的特征值
    bestFeature = -1

    # 标记当前最好的特征值是不是连续值
    flagSeries = 0

    # 如果是连续值的话，用来记录连续值的划分点
    bestSeriesMid = 0.0

    # 对每个特征值进行求信息熵
    for i in range(numFeatures):

        # 得到数据集中所有的当前特征值列表
        featList = [example[i] for example in dataSet]

        if isinstance(featList[0], str):
            infoGain = calcInfoGain(dataSet, featList, i, baseEntropy)
        else:
            # print('当前划分属性为：' + str(labels[i]))
            infoGain, bestMid = calcInfoGainForSeries(dataSet, i, baseEntropy)

        # print('当前特征值为：' + labels[i] + '，对应的信息增益值为：' + str(infoGain))

        # 如果当前的信息增益比原来的大
        if infoGain > bestInfoGain:
            # 最好的信息增益
            bestInfoGain = infoGain
            # 新的最好的用来划分的特征值
            bestFeature = i

            flagSeries = 0
            if not isinstance(dataSet[0][bestFeature], str):
                flagSeries = 1
                bestSeriesMid = bestMid

    # print('信息增益最大的特征为：' + labels[bestFeature])
    if flagSeries:
        return bestFeature, bestSeriesMid
    else:
        return bestFeature


def getDataSet(test_size):
    """
    创建测试的数据集，里面的数值中具有连续值
    :return:
    """
    dataSet = pd.read_csv("iris.data").values.tolist()
    random.shuffle(dataSet)
    train_dataset = dataSet[:int(len(dataSet) * (1 - test_size))]  # 训练数据
    test_dataset = dataSet[int(len(dataSet) * (1 - test_size)):]  # 测试数据
    # 特征值列表
    labels = ['sepal length', 'sepal width', 'petal length', 'petal width']

    return train_dataset, test_dataset, labels


def majorityCnt(classList):
    """
    找到次数最多的类别标签
    :param classList:
    :return:
    """
    # 用来统计标签的票数
    classCount = collections.defaultdict(int)

    # 遍历所有的标签类别
    for vote in classList:
        classCount[vote] += 1

    # 从大到小排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)

    # 返回次数最多的标签
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    """
    创建决策树
    :param dataSet: 数据集
    :param labels: 特征标签
    :return:
    """
    # 拿到所有数据集的分类标签
    classList = [example[-1] for example in dataSet]

    # 统计第一个标签出现的次数，与总标签个数比较，如果相等则说明当前列表中全部都是一种标签，此时停止划分
    if classList.count(classList[0]) == len(classList):
        return classList[0]

    # 计算第一行有多少个数据，如果只有一个的话说明所有的特征属性都遍历完了，剩下的一个就是类别标签
    if len(dataSet[0]) == 1:
        # 返回剩下标签中出现次数较多的那个
        return majorityCnt(classList)

    # 选择最好的划分特征，得到该特征的下标
    bestFeat = chooseBestFeatureToSplit(dataSet=dataSet, labels=labels)

    # 得到最好特征的名称
    bestFeatLabel = ''

    # 记录此刻是连续值还是离散值,1连续，2离散
    flagSeries = 0

    # 如果是连续值，记录连续值的划分点
    midSeries = 0.0

    # 如果是元组的话，说明此时是连续值
    if isinstance(bestFeat, tuple):
        # 重新修改分叉点信息
        bestFeatLabel = str(labels[bestFeat[0]]) + '=' + str(bestFeat[1])
        # 得到当前的划分点
        midSeries = bestFeat[1]
        # 得到下标值
        bestFeat = bestFeat[0]
        # 连续值标志
        flagSeries = 1
    else:
        # 得到分叉点信息
        bestFeatLabel = labels[bestFeat]
        # 离散值标志
        flagSeries = 0

    # 使用一个字典来存储树结构，分叉处为划分的特征名称
    myTree = {bestFeatLabel: {}}

    # 得到当前特征标签的所有可能值
    featValues = [example[bestFeat] for example in dataSet]

    # 连续值处理
    if flagSeries:

        # 将连续值划分为不大于当前划分点和大于当前划分点两部分
        eltDataSet, gtDataSet = splitDataSetForSeries(dataSet, bestFeat, midSeries)
        # 得到剩下的特征标签
        subLabels = labels[:]
        # 递归处理小于划分点的子树
        subTree = createTree(eltDataSet, subLabels)
        myTree[bestFeatLabel]['小于'] = subTree

        # 递归处理大于当前划分点的子树
        subTree = createTree(gtDataSet, subLabels)
        myTree[bestFeatLabel]['大于'] = subTree

        return myTree

    # 离散值处理
    else:

        # 将本次划分的特征值从列表中删除掉
        del (labels[bestFeat])
        # 唯一化，去掉重复的特征值
        uniqueVals = set(featValues)
        # 遍历所有的特征值
        for value in uniqueVals:
            # 得到剩下的特征标签
            subLabels = labels[:]
            # 递归调用，将数据集中该特征等于当前特征值的所有数据划分到当前节点下，递归调用时需要先将当前的特征去除掉
            subTree = createTree(splitDataSet(dataSet=dataSet, axis=bestFeat, value=value), subLabels)
            # 将子树归到分叉处下
            myTree[bestFeatLabel][value] = subTree
        return myTree


# 输入三个变量（决策树，属性特征标签，测试的数据）
def classify(inputTree, featLables, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]  # 树的分支，子集合Dict
    featIndex = featLables.index(firstStr[:firstStr.index('=')])  # 获取决策树第一层在featLables中的位置
    for key in secondDict.keys():
        if testVec[featIndex] > float(firstStr[firstStr.index('=') + 1:]):
            if type(secondDict['大于']).__name__ == 'dict':
                classLabel = classify(secondDict['大于'], featLables, testVec)
            else:
                classLabel = secondDict['大于']
            return classLabel
        else:
            if type(secondDict['小于']).__name__ == 'dict':
                classLabel = classify(secondDict['小于'], featLables, testVec)
            else:
                classLabel = secondDict['小于']
            return classLabel


# 保存决策树
def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'wb')
    pickle.dump(inputTree, fw)
    fw.close()


# 读取决策树
def grabTree(filename):
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)


def calConfuMatrix(myTree, label, test_dataSet):
    matrix = {'Iris-setosa': {'Iris-setosa': 0, 'Iris-versicolor': 0, 'Iris-virginica': 0},
              'Iris-versicolor': {'Iris-setosa': 0, 'Iris-versicolor': 0, 'Iris-virginica': 0},
              'Iris-virginica': {'Iris-setosa': 0, 'Iris-versicolor': 0, 'Iris-virginica': 0}}
    for example in test_dataSet:
        predict = classify(myTree, label, example)  # 对测试数据进行测试，predict为预测的分类
        actual = example[-1]  # actual 为实际的分类
        matrix[actual][predict] += 1  # 填充confusion matrix
    return matrix


# 准确度
def precision(classes, matrix):
    precisionDict = {'Iris-setosa': 0, 'Iris-versicolor': 0, 'Iris-virginica': 0}
    all_predict_num = 0  # 例如存储当预测值为Iris-setosa时的所有情况数量
    for classItem in classes:
        true_predict_num = matrix[classItem][classItem]  # 准确预测数量
        all_predict_num = 0
        for temp_class_item in classes:
            all_predict_num += matrix[temp_class_item][classItem]
        precisionDict[classItem] = round(true_predict_num / all_predict_num, 2)
    return precisionDict


# 召回率
def recall(classes, matrix):
    recallDict = {'Iris-setosa': 0, 'Iris-versicolor': 0, 'Iris-virginica': 0}
    for classItem in classes:
        true_predict_num = matrix[classItem][classItem]  # 准确预测数量
        all_predict_num = 0  # 例如存储当预测值为Iris-setosa时的所有情况数量
        for temp_class_item in classes:
            all_predict_num += matrix[classItem][temp_class_item]
        recallDict[classItem] = round(true_predict_num / all_predict_num, 2)
    return recallDict


# 展示结果
def showResult(precisionValue, recallValue, classes):
    print('\t\t\t\t', 'precision', '\t', 'recall')
    for classItem in classes:
        print(classItem, '\t', precisionValue[classItem], '\t', recallValue[classItem])


if __name__ == '__main__':
    """ 
    处理连续值时候的决策树
    """
    train_dataSet, test_dataSet, labels = getDataSet(0.2)
    temp_label = labels[:]  # 深拷贝
    myTree = createTree(train_dataSet, labels)
    storeTree(myTree, 'tree.txt')
    myTree = grabTree('tree.txt')
    treePlotter.createPlot(myTree)
    matrix = calConfuMatrix(myTree, temp_label, test_dataSet)  # confusion matrix
    classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']  # 所有分类
    precisionValue = precision(classes, matrix)
    recallValue = recall(classes, matrix)
    showResult(precisionValue, recallValue, classes)
