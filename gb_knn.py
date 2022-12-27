import ast
import csv
import math
import time
import warnings
from sklearn.cluster import k_means

import pandas as pd
import random
import numpy
import scipy.io
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold, train_test_split

import numpy as np

# 判断粒球的标签和纯度
def get_label_and_purity(gb):
    # 分离不同标签数据
    len_label = numpy.unique(gb[:, 0], axis=0)
    # print(len_label)
    # print(gb)

    if len(len_label) == 1:
        purity = 1.0
        label = len_label[0]
    else:
        # 矩阵的行数
        num = gb.shape[0]
        gb_label_temp = {}
        for label in len_label.tolist():
            # 分离不同标签数据
            gb_label_temp[sum(gb[:, 0] == label)] = label
        # 粒球中最多的一类数据占整个的比例
        max_label = max(gb_label_temp.keys())
        # print(gb_label_temp.keys())
        purity = max_label / num if num else 1.0
        label = gb_label_temp[max_label]
    # print(label)
    # 标签、纯度
    return label, purity


# 返回粒球中心和半径
def calculate_center_and_radius(gb):
    data_no_label = gb[:, 1:]
    # print(data_no_label)
    center = data_no_label.mean(axis=0)
    radius_mean = numpy.mean((((data_no_label - center) ** 2).sum(axis=1) ** 0.5))
    radius_max = numpy.max((((data_no_label - center) ** 2).sum(axis=1) ** 0.5))
    return center, radius_mean, radius_max


# 计算距离
def calculate_distances(data, p):
    return ((data - p) ** 2).sum(axis=0) ** 0.5


# 绘制粒球
def gb_plot(gb_list, plt_type=0):
    color = {0: 'r', 1: 'k'}
    # 图像宽高与XY轴范围成比例，绘制粒球才是正圆
    plt.figure(figsize=(5, 4))  # 图像宽高
    plt.axis([-1.2, 1.2, -1, 1])  # 设置x轴的范围为[-1.2, 1.2]，y轴的范围为[-1, 1]
    for gb in gb_list:
        # print(key)
        label, p = get_label_and_purity(gb)
        center, radius, radius_max = calculate_center_and_radius(gb)
        if plt_type == 0:  # 绘制所有点
            data0 = gb[gb[:, 0] == 0]
            data1 = gb[gb[:, 0] == 1]
            plt.plot(data0[:, 1], data0[:, 2], '.', color=color[0], markersize=5)
            plt.plot(data1[:, 1], data1[:, 2], '.', color=color[1], markersize=5)

        if plt_type == 0 or plt_type == 1:  # 绘制粒球
            theta = numpy.arange(0, 2 * numpy.pi, 0.01)
            x = center[0] + radius * numpy.cos(theta)
            y = center[1] + radius * numpy.sin(theta)
            plt.plot(x, y, color[label], linewidth=0.8)

        plt.plot(center[0], center[1], 'x' if plt_type == 0 else '.', color=color[label])  # 绘制粒球中心
    plt.show()


def splits(gb_list, purity):
    gb_list_new = []
    for gb in gb_list:
        label, p = get_label_and_purity(gb)
        if p >= purity:
            gb_list_new.append(gb)
        else:
            gb_list_new.extend(splits_ball(gb))
    return gb_list_new


# 去重叠
def isOverlap(gb_dict):
    Flag = True
    # 输入粒球字典
    later_dict = gb_dict.copy()
    while True:
        ball_number_1 = len(gb_dict)
        centers = []  # 中心list
        keys = []  # 去'_'之前的键list
        dict_overlap = {}  # 重叠的球
        center_radius = {}
        for key in later_dict.keys():
            center, radius_mean, radius_max = calculate_center_and_radius(later_dict[key][0])
            # {center:[center, gb, max_distances, radius]}
            center_radius[key] = [center, later_dict[key][0], later_dict[key][1], radius_mean]
            center_temp = []
            keys.append(key)
            for center_split in key.split('_'):
                center_temp.append(float(center_split))
            # 取出所有中心
            centers.append(center_temp)
        centers = np.array(centers)

        # 第一次划分使用传进来的粒球，接下来使用只存在重叠的粒球
        if Flag:
            later_dict = {}
            Flag = False
        # 遍历任意两个粒球中心
        for i, center01 in enumerate(centers):
            for j, center02 in enumerate(centers):
                if i < j and center01[0] != center02[0]:
                    # 如果两个球的标签不同并且两球心之间的距离小于两球的半径和（边界重叠）
                    if calculate_distances(center_radius[keys[i]][0], center_radius[keys[j]][0]) < \
                            center_radius[keys[i]][3] + center_radius[keys[j]][3]:
                        # 只再次划分存在重叠的小球，提高效率
                        if center_radius[keys[i]][3] > center_radius[keys[j]][3]:
                            dict_overlap[keys[i]] = center_radius[keys[i]][1:3]
                        else:
                            dict_overlap[keys[j]] = center_radius[keys[j]][1:3]

        # gb_plot(gb_dict)
        # print('dict_overlap:', dict_overlap.keys())
        # 当重叠粒球数为0，则返回
        if len(dict_overlap) == 0:
            gb_dict.update(later_dict)
            # 记录一次去重叠后的粒球数
            ball_number_2 = len(gb_dict)
            # 如果去重叠前后粒球数不等，则认为还有重叠，再次对所有粒球进行遍历去重叠
            if ball_number_1 != ball_number_2:
                Flag = True
                later_dict = gb_dict.copy()
            else:
                return gb_dict

        # 遍历存在重叠的粒球
        gb_dict_single = dict_overlap.copy()  # 复制一个临时list，接下来再遍历取值
        for i in range(len(gb_dict_single)):
            gb_single = {}
            # 取字典数据，包括键值
            dict_temp = gb_dict_single.popitem()
            gb_single[dict_temp[0]] = dict_temp[1]
            # 如果粒球为单个点，则不继续划分
            if len(dict_temp[1][0]) == 1:
                later_dict.update(gb_single)
                continue
            # print('gb_single:', gb_single.keys())
            gb_dict_new = splits_ball(gb_single).copy()
            # print('gb_dict_new:', gb_dict_new.keys())
            later_dict.update(gb_dict_new)
        # print('later_dict:', later_dict.keys())


def splits_ball(gb):
    splits_k = len(numpy.unique(gb[:, 0], axis=0))
    data_no_label = gb[:, 1:]
    ball_list = []
    label = []
    # X: 数据; n_clusters: K的值; random_state: 随机状态（为了保证程序每次运行都分割一样的训练集和测试集）
    label = k_means(X=data_no_label, n_clusters=splits_k, random_state=5)[1]  # 返回标签
    for single_label in range(0, splits_k):
        ball_list.append(gb[label == single_label, :])
    # print(ball_list)
    return ball_list


def nearest_knn(ball_list, test):
    # 测试最近邻精度
    num = []
    for i in range(len(test)):
        z_dis = float("inf")
        z_label = None
        for ball in ball_list:
            # print(ball_list[k][0])
            center, radius, radius_max = calculate_center_and_radius(ball)  # 平均距离、最大距离作半径
            # print(center,radius)
            label, p = get_label_and_purity(ball)
            DIS = np.sum((test[i, 1:] - center) ** 2) ** 0.5 - radius
            if z_dis > DIS:
                z_dis = DIS
                z_label = label
        if z_label == test[i, 0]:
            num.append(1)
        else:
            num.append(0)
    return num


def mean_std(a):
    # 计算一维数组的均值和标准差
    a = np.array(a)
    std = np.sqrt(((a - np.mean(a)) ** 2).sum() / (a.size - 1))
    return a.mean(), std


def main():
    warnings.filterwarnings("ignore")  # 忽略警告
    # keys = ['fourclass', 'mushroom', 'breastcancer', 'votes', 'svmguide1', 'svmguide3']
    keys = ['fourclass', 'svmguide1', 'diabetes', 'breastcancer', 'creditApproval',
            'votes', 'svmguide3', 'sonar', 'splice', 'mushrooms']
    # keys = ['electrical', 'letter']
    noises = [0]
    for noise in noises:
        print(noise)
        for d in range(len(keys)):
            print(keys[d])
            # print(result[d])
            # df = pd.read_csv(r"UCI\\" + keys[d] + ".csv", header=None)  # 加载数据集
            # data = df.values
            data_mat = scipy.io.loadmat(r'dataset16.mat')
            data = data_mat[keys[d]]
            data[data[:, 0] == -1, 0] = 0
            # data = data[:, 0:3]

            # 数组去重
            data = numpy.unique(data, axis=0)
            data_temp = []
            data_list = data.tolist()
            data = []
            for data_single in data_list:
                if data_single[1:] not in data_temp:
                    data_temp.append(data_single[1:])
                    data.append(data_single)
            data = np.array(data)
            # print(data)
            times = 0

            finish_max = []
            finish_mean = []
            Acc = 0
            acc = []
            for i in range(10):
                # # 加噪声
                # smapleNumber = data.shape[0]
                # data_temp = np.random.rand(smapleNumber, 1)
                # data_temp = data_temp * 10 - np.ones((smapleNumber, 1)) * noise * 10
                # data_temp[np.nonzero(data_temp[:, 0] >= 0), :] = 1
                # data_temp[np.nonzero(data_temp[:, 0] < 0), :] = 0
                # new_train_label = np.multiply(data_temp, data[:, 0].reshape(smapleNumber, 1))
                # data = np.hstack((new_train_label, data[:, 1:]))

                # 记录开始时间
                start = time.time()
                # 划分训练集、测试集
                train, test = train_test_split(data, test_size=0.2)
                # 数组去重
                train = numpy.unique(train, axis=0)
                # 纯度阈值
                purity = 0.95

                # 直接绘制输入数据
                gb_list = [train]
                # print(len(data))
                # gb_plot(gb_list)

                while True:
                    ball_number_1 = len(gb_list)
                    gb_list = splits(gb_list, purity=purity)
                    ball_number_2 = len(gb_list)
                    if ball_number_1 == ball_number_2:  # 粒球数和上一次划分的粒球数一样，即不再变化
                        break

                # 绘制粒球
                # gb_plot(gb_list)
                count_num = nearest_knn(gb_list, test)  # 测试最近邻精度
                # avg, std = mean_std(count_num)
                avg = np.array(count_num).mean()
                # print(avg)
                acc.append(avg)
                # print('acc:', acc)

                # 记录结束时间
                end = time.time()
                times = (end - start) + times

            print('总平均耗时：%s' % (round(times / 10 * 1000, 0)))
            max_acc = np.max(acc)  # 最大精度
            mean_acc = np.mean(acc)  # 平均精度
            print("最终：", round(max_acc, 3), round(mean_acc, 3))
            # filename_0 = 'final\\omean_KNN.csv'
            # csv_fp_0 = open(filename_0, "a+", encoding="utf-8-sig", newline='')
            # writer_0 = csv.writer(csv_fp_0)
            # writer_0.writerow([noise, keys[d], max_acc, mean_acc])
            # csv_fp_0.close()


if __name__ == '__main__':
    main()