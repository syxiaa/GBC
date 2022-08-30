import ast
import math
import time
import warnings

import pandas as pd
import random
import numpy
import scipy.io
import matplotlib.pyplot as plt
from sklearn.cluster import k_means

import numpy as np


# 1.输入数据data
# 2.打印绘制原始数据
# 3.判断粒球的纯度
# 4.纯度不满足要求，k-means划分粒球
# 5.绘制每个粒球的数据点
# 6.计算粒球均值，得到粒球中心和半径，绘制粒球

# Calculate label and purity of granular-balls
def get_label_and_purity(gb):
    # Calculate the number of data categories
    len_label = numpy.unique(gb[:, 0], axis=0)
    # print(len_label)

    if len(len_label) == 1:
        purity = 1.0
        label = len_label[0]
    else:
        num = gb.shape[0]
        gb_label_temp = {}
        for label in len_label.tolist():
            # Separate data with different labels
            gb_label_temp[sum(gb[:, 0] == label)] = label
        # print(gb_label_temp)
        # The proportion of the largest category of data in all data
        max_label = max(gb_label_temp.keys())
        purity = max_label / num if num else 1.0
        label = gb_label_temp[max_label]
    # print(label)
    # label, purity
    return label, purity


# Calculate granular-balls center and radius
def calculate_center_and_radius(gb):
    data_no_label = gb[:, 1:]
    # print(data_no_label)
    center = data_no_label.mean(axis=0)
    radius_mean = numpy.mean((((data_no_label - center) ** 2).sum(axis=1) ** 0.5))
    # radius_max = numpy.max((((data_no_label - center) ** 2).sum(axis=1) ** 0.5))
    return center, radius_mean


# Calculate distance
def calculate_distances(data, p):
    # print(data, p)
    return ((data - p) ** 2).sum(axis=0) ** 0.5


# draw granular-balls
def gb_plot(gb_dict, plt_type=0):
    color = {-1: 'r', 1: 'k', 0: 'b', 3: 'y', 4: 'g', 5: 'c', 6: 'm', 7: 'peru', 8: 'pink', 9: 'gold'}
    plt.figure(figsize=(5, 4))  # width and height of the image
    plt.axis([-1.2, 1.2, -1, 1])
    for key in gb_dict.keys():
        gb = gb_dict[key][0][:, 0:3]
        label, p = get_label_and_purity(gb)
        k = np.unique(gb[:, 0], axis=0)
        center, radius_mean = calculate_center_and_radius(gb)

        if plt_type == 0:
            # plot all points
            for i in k.tolist():
                data0 = gb[gb[:, 0] == i]
                plt.plot(data0[:, 1], data0[:, 2], '.', color=color[i], markersize=5)

        if plt_type == 0 or plt_type == 1:  # draw balls
            theta = numpy.arange(0, 2 * numpy.pi, 0.01)
            x = center[0] + radius_mean * numpy.cos(theta)
            y = center[1] + radius_mean * numpy.sin(theta)
            plt.plot(x, y, color[label], linewidth=0.8)

        plt.plot(center[0], center[1], 'x' if plt_type == 0 else '.', color=color[label])  # draw centers
    plt.show()


# draw granular-balls
def plot_gb(granular_ball_list, plt_type=0):
    color = {-1: 'r', 1: 'k', 0: 'b'}
    plt.figure(figsize=(5, 4))
    plt.axis([-1.2, 1.2, -1, 1])
    ball_num_str = str(len(granular_ball_list))
    for granular_ball in granular_ball_list:
        label, p = get_label_and_purity(granular_ball)
        center, radius= calculate_center_and_radius(granular_ball)

        if plt_type == 0:
            data0 = granular_ball[granular_ball[:, 0] == 0]
            data1 = granular_ball[granular_ball[:, 0] == 1]
            data2 = granular_ball[granular_ball[:, 0] == -1]
            plt.plot(data0[:, 1], data0[:, 2], '.', color=color[0], markersize=5)
            plt.plot(data1[:, 1], data1[:, 2], '.', color=color[1], markersize=5)
            plt.plot(data2[:, 1], data2[:, 2], '.', color=color[-1], markersize=5)

        if plt_type == 0 or plt_type == 1:
            theta = numpy.arange(0, 2 * numpy.pi, 0.01)
            x = center[0] + radius * numpy.cos(theta)
            y = center[1] + radius * numpy.sin(theta)
            plt.plot(x, y, color[label], linewidth=0.8)

        plt.plot(center[0], center[1], 'x' if plt_type == 0 else '.', color=color[label])

    plt.show()


def splits(purity, gb_dict):
    gb_dict_new = {}
    while True:
        # Copy a temporary list, and then traverse the value
        if len(gb_dict_new) == 0:
            # initial assignment
            gb_dict_temp = gb_dict.copy()
        else:
            # Subsequent traversal assignment
            gb_dict_temp = gb_dict_new.copy()
        gb_dict_new = {}
        # record the number before a division
        ball_number_1 = len(gb_dict_temp)
        # print("ball_number_1:", ball_number_1)
        for key in gb_dict_temp.keys():
            gb_single = {}
            # Fetch dictionary data, including keys and values
            gb_single[key] = gb_dict_temp[key]
            # print("gb_single:", gb_single)
            gb = gb_single[key][0]
            # print(len(gb))

            # purity
            p = get_label_and_purity(gb)[1]
            # print(p)
            # Determine whether the purity meets the requirements, if not, continue to divide
            if p < purity and len(gb) != 1:
                # print(gb_single)
                gb_dict_re = splits_ball(gb_single).copy()
                gb_dict_new.update(gb_dict_re)
            else:
                gb_dict_new.update(gb_single)
                continue
        # gb_dict_new = adjust_center(gb_dict_new)
        # record the number after a division
        ball_number_2 = len(gb_dict_new)
        # print("ball_number_2:", len(gb_dict_new))
        # The number of granular-balls is the same as the number of granular-balls last divided, that is, it will not change
        if ball_number_1 == ball_number_2:
            break

        # draw granular-balls
        # gb_plot(gb_dict_new)

    # draw granular-balls
    # gb_plot(gb_dict_new)
    return gb_dict_new

# split granular-balls
def splits_ball(gb_dict):
    # {center: [gb, distances]}
    center = []  # old center
    distances_other_class = []  # distance to heterogeneous data points
    balls = []  # The result after clustering
    center_other_class = []
    ball_list = {}  # Returned dictionary result, key: center point, value (ball , distance from data to center)
    distances_other_temp = []

    centers_dict = []  # centers
    gbs_dict = []  # data
    distances_dict = []  # distances

    # Fetch dictionary data, including keys and values
    gb_dict_temp = gb_dict.popitem()
    for center_split in gb_dict_temp[0].split('_'):
        center.append(float(center_split))
    center = np.array(center)
    gb = gb_dict_temp[1][0]  # Get granular-ball data
    distances = gb_dict_temp[1][1]  # Take out the distance to the old center
    # print('center:', center)
    # print('gb:', gb)
    # print('distances:', distances)
    centers_dict.append(center)  # old center join the centers


    # Take a new center
    len_label = numpy.unique(gb[:, 0], axis=0)
    # When the input has only one type of data, select a point different from the original center
    if len(len_label) > 1:
        gb_class = len(len_label)
    else:
        gb_class = 2
    # Take multiple centers for multiple types of data
    len_label = len_label.tolist()
    for i in range(0, gb_class - 1):
        # print(len_label)
        if len(len_label) < 2:
            # When de-overlapping, there is no heterogeneous point situation
            gb_temp = np.delete(gb, distances.index(0), axis=0)  # Remove the old center
            ran = random.randint(0, len(gb_temp) - 1)
            center_other_temp = gb_temp[ran]  # Take a new center
            center_other_class.append(center_other_temp)
        else:
            if center[0] in len_label:
                len_label.remove(center[0])
            gb_temp = gb[gb[:, 0] == len_label[i], :]  # Extract heterogeneous data

            # random center of heterogeneity
            ran = random.randint(0, len(gb_temp) - 1)
            center_other_temp = gb_temp[ran]

            # center_other_temp = select_center(gb_temp)
            # print(center_other_temp)
            center_other_class.append(center_other_temp)
            # print(distances.index(max(distances)))
    # print('center_other_class:', center_other_class)
    # join the centers
    centers_dict.extend(center_other_class)
    # print('centers_dict:', centers_dict)

    # Store all data distance to old center
    distances_other_class.append(distances)
    # Calculate the distance to each new center
    for center_other in center_other_class:
        balls = []  # The result after clustering
        distances_other = []
        for feature in gb:
            distances_other.append(calculate_distances(feature[1:], center_other[1:]))
        # new centers
        # distances_dict.append(distances_other)
        distances_other_temp.append(distances_other)  # Temporary storage distance to each new center
        # Store all data distance to new center
        distances_other_class.append(distances_other)
    # print('distances_other_class:', len(distances_other_class))

    # The distance from a certain data to the original center and the new center, take the smallest for classification
    for i in range(len(distances)):
        distances_temp = []
        distances_temp.append(distances[i])
        for distances_other in distances_other_temp:
            distances_temp.append(distances_other[i])
        # print('distances_temp:', distances_temp)
        classification = distances_temp.index(min(distances_temp))  # 0:old center；1,2...：new centers
        balls.append(classification)
    # Clustering situation
    balls_array = np.array(balls)
    # print("Clustering situation：", balls_array)

    # Assign data based on clustering
    for i in range(0, len(centers_dict)):
        gbs_dict.append(gb[balls_array == i, :])
    # print('gbs_dict:', gbs_dict)

    # assign new distance
    i = 0
    for j in range(len(centers_dict)):
        distances_dict.append([])
    # print('distances_dict:', distances_dict)
    for label in balls:
        distances_dict[label].append(distances_other_class[label][i])
        i += 1
    # print('distances_dict:', distances_dict)

    # packed into a dictionary
    for i in range(len(centers_dict)):
        gb_dict_key = str(float(centers_dict[i][0]))
        for j in range(1, len(centers_dict[i])):
            gb_dict_key += '_' + str(float(centers_dict[i][j]))
        gb_dict_value = [gbs_dict[i], distances_dict[i]]  # Pellets + distance to centers
        ball_list[gb_dict_key] = gb_dict_value

    # print('ball_list:', ball_list)
    return ball_list


def main():
    warnings.filterwarnings("ignore")  # ignore warning
    # keys = ['fourclass', 'mushrooms', 'breastcancer', 'votes', 'svmguide1', 'svmguide3']
    # keys = ['diabetes', 'creditApproval', 'sonar', 'splice']
    keys = ['fourclass', 'svmguide1', 'diabetes', 'breastcancer', 'creditApproval',
            'votes', 'svmguide3', 'sonar', 'splice', 'mushrooms']
    # splits_k = 2,para = 10,运行时间（粒球数）
    # result = ['fourclass: 302(31),287(115,112)', 'mushroom: 2916(17),1008(18,18)', 'breastcancer: 1429(198),880(311,306)', 'votes: 1013(9),555(9,9)', 'svmguide1: 2958(366),2247(742,855)', 'svmguide3: 4468(673),2350(943,943)']
    result = ['fourclass: 315(31),54,55', 'mushrooms: 2695(17),14,14', 'breastcancer: 1498(196),103,168',
              'votes: 1574(9),2,2', 'svmguide1: 2608(315),968,1122', 'svmguide3: 4506(669),330,696']
    for d in range(len(keys)):
        print(keys[d])
        # print(result[d])
        # keys[d] = 'mushroom'
        times = 0
        num_gb = 0
        # print('开始时间', start)
        # df = pd.read_csv(r"UCI\\" + keys[d] + ".csv", header=None)  # 加载数据集
        # data = df.values
        data_mat = scipy.io.loadmat(r'dataset16.mat')
        data = data_mat[keys[d]]
        # data[data[:, 0] == -1, 0] = 0
        # data = data[:, 0:3]  # 为方便二维可视化，这里只取两维数据

        # print(len(data))

        # 数组去重;Remove duplicate data and data with different labels but the same attributes
        data = numpy.unique(data, axis=0)
        data_temp = []
        data_list = data.tolist()
        data = []
        for data_single in data_list:
            if data_single[1:] not in data_temp:
                data_temp.append(data_single[1:])
                data.append(data_single)
        data = np.array(data)

        # print(len(data))
        for i in range(0, 10):  # Repeat 10 times to get the average precision
            ball_list = []
            # record start time
            start = time.time()
            purity = 0.95

            # initial random center
            center_init = data[random.randint(0, len(data) - 1), :]
            # center_init = data[:, 1:3].mean(axis=0)
            # print(center_init)

            # The distance from the initial center to the data
            distance_init = []
            for feature in data:
                distance_init.append(calculate_distances(feature[1:], center_init[1:]))
            # print('distance_init:', len(distance_init))

            # packed into a dictionary:{center: [gb, distances]}
            gb_dict = {}
            gb_dict_key = str(center_init.tolist()[0])
            for j in range(1, len(center_init)):
                gb_dict_key += '_' + str(center_init.tolist()[j])
            gb_dict_value = [data, distance_init]
            gb_dict[gb_dict_key] = gb_dict_value

            # first time drawing a sphere
            # gb_plot2(gb_dict)

            # Classification
            gb_dict = splits(purity=purity, gb_dict=gb_dict)

            k_centers = []
            splits_k = len(gb_dict)
            for key in gb_dict.keys():
                k_center = []
                for k in key.split('_'):
                    k_center.append(float(k))
                k_centers.append(k_center[1:])
            # print(np.array(k_centers))
            # Perform a global division
            label_cluster = k_means(X=data[:, 1:], n_clusters=splits_k, n_init=2, init=np.array(k_centers), random_state=5)[1]
            for single_label in range(splits_k):
                ball_list.append(data[label_cluster == single_label, :])
            # gb_plot2(gb_dict)
            # draw granular-balls
            plot_gb(ball_list)
            # plot_gb(ball_list, 1)

            # record number
            num_gb += len(ball_list)
            # record end time
            end = time.time()
            times = (end - start) + times

            # print('Number of granular-balls：', len(gb_dict))
            # print('time', round((end - start) * 1000, 0))
            # break
        print('Number of granular-balls：', round(num_gb / 10, 0))
        print('total average time：%s' % (round(times / 10 * 1000, 0)))
        break


if __name__ == '__main__':
    main()
