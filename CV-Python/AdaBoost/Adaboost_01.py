

# 将最小的错误率minError设置为正无穷大
# 对数据集的每一个特征（第一层循环）：
#     对每个步长（第二层循环）：
#         对每个不等号（第三个循环）：
#             建立一颗单层决策树并利用加权数据及对它进行测试
#             如果错误率低语minError，则将当前层决策树设为最佳单层决策树
# 返回最佳单层决策树

#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
import matplotlib

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
@description: 单层决策树分类函数
@param: dataMatrix - 数据矩阵
		dimen - 第dimen列，也就是第几个特征
		threshVal - 阈值
		threshIneq - 标志 "lt" "gt"
        这里lt表示less than，表示分类方式，对于小于阈值的样本点赋值为-1，
        gt表示greater than，也是表示分类方式，对于大于阈值的样本点赋值为-1。
@return: 
'''
def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
    retArray = np.ones((np.shape(dataMatrix)[0],1))				#初始化retArray为1
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0	 	#如果小于阈值,则赋值为-1
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0 		#如果大于阈值,则赋值为-1
    return retArray

'''
@description: 找到数据集上最佳的单层决策树
@param: dataArr - 数据矩阵
        classLabels - 数据标签
        D - 样本权重 
@return: bestStump - 最佳单层决策树信息
        minError - 最小误差
        bestClasEst - 最佳的分类结果
'''
def buildStump(dataArr,classLabels,D):
    dataMatrix = np.mat(dataArr); labelMat = np.mat(classLabels).T
    m,n = np.shape(dataMatrix)
    numSteps = 10.0; bestStump = {}; bestClasEst = np.mat(np.zeros((m,1)))
    #最小误差初始化为正无穷大
    minError = float('inf')
    #遍历所有特征
    for i in range(n):
        #找到特征中最小的值和最大值
        rangeMin = dataMatrix[:,i].min(); rangeMax = dataMatrix[:,i].max()
        #计算步长
        stepSize = (rangeMax - rangeMin) / numSteps
        for j in range(-1, int(numSteps) + 1):
            for inequal in ['lt', 'gt']:
                #计算阈值
                threshVal = (rangeMin + float(j) * stepSize)
                #计算分类结果
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                #初始化误差矩阵
                errArr = np.mat(np.ones((m,1)))
                #分类正确的,赋值为0
                errArr[predictedVals == labelMat] = 0
                #计算误差
                weightedError = D.T * errArr
                #找到误差最小的分类方式-
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    #第一行的特征
                    bestStump['dim'] = i
                    #阈值
                    bestStump['thresh'] = threshVal
                    #标志 "lt" "gt"
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst

# if __name__ == "__main__":
#     dataArr,classLabels = loadSimpData()
#     D = np.mat(np.ones((5, 1)) / 5)
#     bestStump,minError,bestClasEst = buildStump(dataArr,classLabels,D)
#     print('bestStump:\n', bestStump)
#     print('minError:\n', minError)
#     print('bestClasEst:\n', bestClasEst)

'''
@description: 使用AdaBoost算法提升弱分类器性能
@param: dataArr - 数据矩阵
		classLabels - 数据标签
		numIt - 最大迭代次数
@return: weakClassArr - 训练好的分类器
		aggClassEst - 类别估计累计值
'''
def adaBoostTrainDS(dataArr, classLabels, numIt = 40):
    weakClassArr = []
    #初始化权重
    m = np.shape(dataArr)[0]
    D = np.mat(np.ones((m, 1)) / m)
    aggClassEst = np.mat(np.zeros((m,1)))
    for i in range(numIt):
        #构建单层决策树
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        #计算弱学习算法权重alpha,使error不等于0,因为分母不能为0
        alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-16)))
        #存储弱学习算法权重和单层决策树
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        #计算e的指数项
        expon = np.multiply(-1 * alpha * np.mat(classLabels).T, classEst)
        D = np.multiply(D, np.exp(expon))
        #根据样本权重公式，更新样本权重
        D = D / D.sum()
        #计算AdaBoost误差，当误差为0的时候，退出循环
        #计算类别估计累计值
        aggClassEst += alpha * classEst
        #计算误差
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((m,1)))
        errorRate = aggErrors.sum() / m
        print("total error: ", errorRate)
        if errorRate == 0.0: break
    return weakClassArr, aggClassEst

# if __name__ == "__main__":
#     dataArr,classLabels = loadSimpData()
#     weakClassArr, aggClassEst = adaBoostTrainDS(dataArr, classLabels)
#     print(weakClassArr)
#     print(aggClassEst)

'''
@description: 分类函数
@param: datToClass - 待分类样例
		classifierArr - 训练好的分类器
@return: 分类结果
'''
def adaClassify(datToClass,classifierArr):
    dataMatrix = np.mat(datToClass)
    m = np.shape(dataMatrix)[0]
    aggClassEst = np.mat(np.zeros((m,1)))
    #遍历所有分类器，进行分类
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'], classifierArr[i]['thresh'], classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
        print(aggClassEst)
    return np.sign(aggClassEst)

'''
@description: 画图函数
'''
def draw_figure(dataMat, labelList, weakClassArr):  # 画图
    # myfont = FontProperties(fname='/usr/share/fonts/simhei.ttf')    # 显示中文
    matplotlib.rcParams['axes.unicode_minus'] = False  # 防止坐标轴的‘-’变为方块
    matplotlib.rcParams["font.sans-serif"] = ["simhei"]  # 第二种显示中文的方法
    fig = plt.figure()  # 创建画布
    ax = fig.add_subplot(111)  # 添加子图

    red_points_x = []  # 红点的x坐标
    red_points_y = []  # 红点的y坐标
    blue_points_x = []  # 蓝点的x坐标
    blue_points_y = []  # 蓝点的y坐标
    m, n = np.shape(dataMat)  # 训练集的维度是 m×n ，m就是样本个数，n就是每个样本的特征数
    dataSet_list = np.array(dataMat)  # 训练集转化为array数组

    for i in range(m):  # 遍历训练集，把红点，蓝点分开存入
        if labelList[i] == 1:
            red_points_x.append(dataSet_list[i][0])  # 红点x坐标
            red_points_y.append(dataSet_list[i][1])
        else:
            blue_points_x.append(dataSet_list[i][0])
            blue_points_y.append(dataSet_list[i][1])

    line_thresh = 0.025  # 画线阈值，就是不要把线画在点上，而是把线稍微偏移一下，目的就是为了让图更加美观直接
    annotagte_thresh = 0.03  # 箭头间隔，也是为了美观
    x_min = y_min = 0.50  # 自设的坐标显示的最大最小值，这里固定死了，应该是根据训练集的具体情况设定
    x_max = y_max = 2.50

    v_line_list = []  # 把竖线阈值的信息存起来，包括阈值大小，分类方式，alpha大小都存起来
    h_line_list = []  # 横线阈值也是如此，因为填充每个区域时，竖阈值和横阈值是填充边界，是不一样的，需各自分开存贮
    for baseClassifier in weakClassArr:  # 画阈值
        if baseClassifier['dim'] == 0:  # 画竖线阈值
            if baseClassifier['ineq'] == 'lt':  # 根据分类方式,lt时
                ax1 = ax.vlines(baseClassifier['thresh'] + line_thresh, y_min, y_max, colors='green', label='阈值')  # 画直线
                ax.arrow(baseClassifier['thresh'] + line_thresh, 1.5, 0.08, 0, head_width=0.05,
                         head_length=0.02)  # 显示箭头
                ax.text(baseClassifier['thresh'] + annotagte_thresh, 1.5 + line_thresh,
                        str(round(baseClassifier['alpha'], 2)))  # 画alpha值
                v_line_list.append(
                    [baseClassifier['thresh'], 1, baseClassifier['alpha']])  # 把竖线信息存入，注意分类方式，lt就存1,gt就存-1

            else:  # gt时，分类方式不同，箭头指向也不同
                ax.vlines(baseClassifier['thresh'] + line_thresh, y_min, y_max, colors='green', label="阈值")
                ax.arrow(baseClassifier['thresh'] + line_thresh, 1., -0.08, 0, head_width=0.05, head_length=0.02)
                ax.text(baseClassifier['thresh'] + annotagte_thresh, 1. + line_thresh,
                        str(round(baseClassifier['alpha'], 2)))
                v_line_list.append([baseClassifier['thresh'], -1, baseClassifier['alpha']])
        else:  # 画横线阈值
            if baseClassifier['ineq'] == 'lt':  # 根据分类方式，lt时
                ax.hlines(baseClassifier['thresh'] + line_thresh, x_min, x_max, colors='black', label="阈值")
                ax.arrow(1.5 + line_thresh, baseClassifier['thresh'] + line_thresh, 0., 0.08, head_width=0.05,
                         head_length=0.05)
                ax.text(1.5 + annotagte_thresh, baseClassifier['thresh'] + 0.04, str(round(baseClassifier['alpha'], 2)))
                h_line_list.append([baseClassifier['thresh'], 1, baseClassifier['alpha']])
            else:  # gt时
                ax.hlines(baseClassifier['thresh'] + line_thresh, x_min, x_max, colors='black', label="阈值")
                ax.arrow(1.0 + line_thresh, baseClassifier['thresh'], 0., 0.08, head_width=-0.05, head_length=0.05)
                ax.text(1.0 + annotagte_thresh, baseClassifier['thresh'] + 0.04, str(round(baseClassifier['alpha'], 2)))
                h_line_list.append([baseClassifier['thresh'], -1, baseClassifier['alpha']])
    v_line_list.sort(key=lambda x: x[0])  # 我们把存好的竖线信息按照阈值大小从小到大排序，因为我们填充颜色是从左上角开始，所以竖线从小到大排
    h_line_list.sort(key=lambda x: x[0], reverse=True)  # 横线从大到小排序
    v_line_list_size = len(v_line_list)  # 排好之后，得到竖线有多少条
    h_line_list_size = len(h_line_list)  # 得到横线有多少条
    alpha_value = [x[2] for x in v_line_list] + [y[2] for y in h_line_list]  # 把属性横线的所有alpha值取出来，这里也证实了上面的排序不是无用功
    print('alpha_value', alpha_value)

    for i in range(h_line_list_size + 1):  # 开始填充颜色，(横线的条数+1) × (竖线的条数+1) = 分割的区域数,然后开始往这几个区域填颜色
        for j in range(v_line_list_size + 1):  # 我们是左上角开始填充直到右下角，所以采用这种遍历方式
            list_test = list(multiply([1] * j + [-1] * (v_line_list_size - j), [x[1] for x in v_line_list])) + list(
                multiply([-1] * i + [1] * (h_line_list_size - i), [x[1] for x in h_line_list]))
            # 上面是一个规律公式，后面会用文字解释它
            # print('list_test',list_test)
            temp_value = multiply(alpha_value,
                                  list_test)  # list_test其实就是加减号，我们知道了所有alpha值，可是每个alpha是相加还是相加，这就是list_test作用了
            reslut_test = sign(sum(temp_value))  # 计算完后，sign一下，然后根据结果进行分类
            if reslut_test == 1:  # 如果是1,就是正类红点
                color_select = 'orange'  # 填充的颜色是橘红色
                hatch_select = '.'  # 填充图案是。
                # print("是正类，红点")
            else:  # 如果是-1,那么是负类蓝点
                color_select = 'green'  # 填充的颜色是绿色
                hatch_select = '*'  # 填充图案是*
                # print("是负类，蓝点")
            if i == 0:  # 上边界     现在开始填充了，用fill_between函数，我们需要得到填充的x坐标范围，和y的坐标范围，x范围就是多条竖线阈值夹着的区域，y范围是横线阈值夹着的范围
                if j == 0:  # 左上角
                    ax.fill_between(x=[x for x in arange(x_min, v_line_list[j][0] + line_thresh, 0.001)], y1=y_max,
                                    y2=h_line_list[i][0] + line_thresh, color=color_select, alpha=0.3,
                                    hatch=hatch_select)
                elif j == v_line_list_size:  # 右上角
                    ax.fill_between(x=[x for x in arange(v_line_list[-1][0] + line_thresh, x_max, 0.001)], y1=y_max,
                                    y2=h_line_list[i][0] + line_thresh, color=color_select, alpha=0.3,
                                    hatch=hatch_select)
                else:  # 中间部分
                    ax.fill_between(x=[x for x in
                                       arange(v_line_list[j - 1][0] + line_thresh, v_line_list[j][0] + line_thresh,
                                              0.001)], y1=y_max, y2=h_line_list[i][0] + line_thresh, color=color_select,
                                    alpha=0.3, hatch=hatch_select)
            elif i == h_line_list_size:  # 下边界
                if j == 0:  # 左下角
                    ax.fill_between(x=[x for x in arange(x_min, v_line_list[j][0] + line_thresh, 0.001)],
                                    y1=h_line_list[-1][0] + line_thresh, y2=y_min, color=color_select, alpha=0.3,
                                    hatch=hatch_select)
                elif j == v_line_list_size:  # 右下角
                    ax.fill_between(x=[x for x in arange(v_line_list[-1][0] + line_thresh, x_max, 0.001)],
                                    y1=h_line_list[-1][0] + line_thresh, y2=y_min, color=color_select, alpha=0.3,
                                    hatch=hatch_select)
                else:  # 中间部分
                    ax.fill_between(x=[x for x in
                                       arange(v_line_list[j - 1][0] + line_thresh, v_line_list[j][0] + line_thresh,
                                              0.001)], y1=h_line_list[-1][0] + line_thresh, y2=y_min,
                                    color=color_select, alpha=0.3, hatch=hatch_select)
            else:
                if j == 0:  # 中左角
                    ax.fill_between(x=[x for x in arange(x_min, v_line_list[j][0] + line_thresh, 0.001)],
                                    y1=h_line_list[i - 1][0] + line_thresh, y2=h_line_list[i][0] + line_thresh,
                                    color=color_select, alpha=0.3, hatch=hatch_select)
                elif j == v_line_list_size:  # 中右角
                    ax.fill_between(x=[x for x in arange(v_line_list[-1][0] + line_thresh, x_max, 0.001)],
                                    y1=h_line_list[i - 1][0] + line_thresh, y2=h_line_list[i][0] + line_thresh,
                                    color=color_select, alpha=0.3, hatch=hatch_select)
                else:  # 中间部分
                    ax.fill_between(x=[x for x in
                                       arange(v_line_list[j - 1][0] + line_thresh, v_line_list[j][0] + line_thresh,
                                              0.001)], y1=h_line_list[i - 1][0] + line_thresh,
                                    y2=h_line_list[i][0] + line_thresh, color=color_select, alpha=0.3,
                                    hatch=hatch_select)

    ax.scatter(red_points_x, red_points_y, s=30, c='red', marker='s', label="red points")  # 画红点
    ax.scatter(blue_points_x, blue_points_y, s=40, label="blue points")  # 画蓝点
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()  # 显示图例    如果你想用legend设置中文字体，参数设置为 prop=myfont
    ax.set_title("AdaBoost分类", position=(0.5, -0.175))  # 设置标题，改变位置，可以放在图下面，这个position是相对于图片的位置
    plt.show()



if __name__ == "__main__":
    dataArr,classLabels = loadSimpData()
    weakClassArr, aggClassEst = adaBoostTrainDS(dataArr, classLabels)
    draw_figure(dataArr, classLabels, weakClassArr)  # 画图
    print(adaClassify([[0,0],[5,5]], weakClassArr))

