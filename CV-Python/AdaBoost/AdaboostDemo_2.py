#《机器学习实战》7.1利用AdaBoost元算法提高分类性能
import numpy as np
import matplotlib.pyplot as plt

'''
@description: 创建单层决策树的数据集
@param: None
@return: dataMat - 数据矩阵
        classLabels - 数据标签
'''
def loadSimpData():
    dataMat = np.matrix([
        [ 1. ,  2.1],
        [ 1.5,  1.6],
        [ 1.3,  1. ],
        [ 1. ,  1. ],
        [ 2. ,  1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataMat,classLabels

'''
@description: 数据可视化
@param: dataMat - 数据矩阵
        labelMat - 数据标签 
@return: None
'''
def showDataSet(dataMat,labelMat):
    data_plus = []                                  #正样本
    data_minus = []                                 #负样本
    for i in range(len(dataMat)):
        if labelMat[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    data_plus_np = np.array(data_plus)                                             #转换为numpy矩阵
    data_minus_np = np.array(data_minus)                                         #转换为numpy矩阵
    plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1])        #正样本散点图
    plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1])     #负样本散点图
    plt.show()

if __name__ == "__main__":
    dataArr,classLabels = loadSimpData()
    showDataSet(dataArr,classLabels)
