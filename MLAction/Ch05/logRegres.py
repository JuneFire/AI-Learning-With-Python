'''
Created on Oct 27, 2010
Logistic Regression Working Module
@author: Peter
'''
from numpy import *


def loadDataSet():
    dataMat = [];
    labelMat = []
    fr = open('D:\PycharmProjects\AI-Learning-With-Python\MLAction\Ch05\/testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))


def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)  # 转为Numpy矩阵
    labelMat = mat(classLabels).transpose()  # 转置
    m, n = shape(dataMatrix)
    alpha = 0.001
    maxCycle = 500
    weight = ones((n, 1))
    for k in range(maxCycle):
        h = sigmoid(dataMatrix * weight)  # (m , 1)
        error = (labelMat - h)
        weight = weight + alpha * dataMatrix.transpose() * error
    return weight


def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)  # add_subplot()函数在一张figure里面生成多张子图参数111，表示1行1列第1个位置
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    # 直线 x 的范围
    x = arange(-3.0, 3.0, 0.1)
    # 画出直线，weights[0]*1.0+weights[1]*x+weights[2]*y=0
    # 之前计算时对原始数据做了拓展，将两维拓展为三维，第一维全部设置为1.0，实际他是一个 y=ax+b, b常量
    y = (-weights[0] - weights[1] * x) / weights[2]  # 最佳拟合线
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


'''
随机梯度上升算法
'''


def stocGradAscent0(dataMatrix, classLabels):
    dataMatrix = array(dataMatrix)  # 传进来的dataMatrix是一个列表，要先转化成np的数组，才能与error相乘。
    m, n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)  # initialize to all ones
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights


'''
优化后的随机梯度上升算法
'''


def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    dataMatrix = array(dataMatrix)
    m, n = shape(dataMatrix)
    weights = ones(n)  # initialize to all ones
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.00001  # apha decreases with iteration, does not
            randIndex = int(random.uniform(0, len(dataIndex)))  # go to 0 because of the constant
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del (dataIndex[randIndex])
    return weights


def classifyVector(inX, weights):
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def colicTest():
    frTrain = open('horseColicTraining.txt');
    frTest = open('horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 1000)
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount) / numTestVec)
    print("the error rate of this test is: %f" % errorRate)
    return errorRate


def multiTest():
    numTests = 10;
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the average error rate is: %f" % (numTests, errorSum / float(numTests)))


if __name__ == '__main__':
    # dataArr, lableMat = loadDataSet()
    # weights = gradAscent(dataArr, lableMat)
    # print(weights)
    # plotBestFit(weights.getA())
    multiTest()
