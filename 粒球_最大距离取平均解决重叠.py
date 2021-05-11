# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/4/2 16:41
# @Author  : xiah
# @File    : 粒球_最大距离平均4.2.2.py


import numpy as np

np.set_printoptions(threshold=np.inf)
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings("ignore")  # Ignore the warning
np.set_printoptions(suppress=True)
from collections import Counter
from sklearn.cluster import k_means
import random
from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import k_means
import numpy
import matplotlib.pyplot as plt


class GranularBall:
    """粒球类"""

    def __init__(self, data):  # data is labeled data, the penultimate column is the label, and the last column is the index
        self.data = data
        self.data_no_label = data[:, 1:-1]
        self.num, self.dim = self.data_no_label.shape  # Data dimension
        self.center = self.data_no_label.mean(0)  # Ball center
        self.label, self.purity = self.__get_label_and_purity()  # granular balls label and purity
        self.radius = 0
        self.init_center = self.random_center()  # Initialize the centroid
        self.label_num = len(set(data[:, 0]))  # Number of different labels

    def random_center(self):
        # Generate random centroid
        center_array = np.empty(shape=[0, len(self.data_no_label[0, :])])
        for i in set(self.data[:, 0]):
            data_set = self.data_no_label[self.data[:, 0] == i, :]
            random_data = data_set[random.randrange(len(data_set)), :]
            center_array = np.append(center_array, [random_data], axis=0)
        return center_array

    def __get_label_and_purity(self):
        # Calculate granular balls label and purity
        count = Counter(self.data[:, 0])
        label = max(count, key=count.get)
        purity = count[label] / self.num
        return label, purity

    def split_clustering(self):
        # k_means divides the cluster
        Clusterings = []
        ClusterLists = k_means(X=self.data_no_label, init=self.init_center, n_clusters=self.label_num)
        data_label = ClusterLists[1]
        for i in set(data_label):
            Cluster_data = self.data[data_label == i, :]
            if len(Cluster_data) > 1:
                Cluster = GranularBall(Cluster_data)
                Clusterings.append(Cluster)
        return Clusterings

    def split_clustering_2(self):
        # Use k_means to divide the cluster again
        Clusterings = []
        center_array = np.empty(shape=[0, len(self.data_no_label[0, :])])
        for i in range(2):
            random_data = self.data_no_label[random.randrange(len(self.data)), :]
            center_array = np.append(center_array, [random_data], axis=0)
        ClusterLists = k_means(X=self.data_no_label, n_clusters=2)
        data_label = ClusterLists[1]
        for i in range(2):
            Cluster_data = self.data[data_label == i, :]
            if len(Cluster_data) > 1:
                Cluster = GranularBall(Cluster_data)
                if len(Cluster.data) > 1:
                    Clusterings.append(Cluster)
        return Clusterings


class GBList:
    """粒球列表"""

    def __init__(self, data, alldata):
        self.data = data
        self.alldata = alldata
        self.granular_balls = [GranularBall(self.data)]

    def init_granular_balls(self, purity=1.0):
        ll = len(self.granular_balls)
        i = 0
        while True:
            # If the purity of the ball is less than 1, then continue to divide the granular balls
            if self.granular_balls[i].purity < purity:
                split_clusters = self.granular_balls[i].split_clustering()
                if len(split_clusters) <= 1:
                    self.granular_balls.pop(i)
                    ll -= 1
                else:
                    self.granular_balls[i] = split_clusters[0]
                    self.granular_balls.extend(split_clusters[1:])
                    ll += len(split_clusters) - 1
            elif self.granular_balls[i].purity == purity:  # When the ball purity reaches the set value
                dis = np.sum((self.granular_balls[i].data[:, 1:-1] - self.granular_balls[i].center) ** 2, axis=1) ** 0.5
                radie = np.max(dis)  # Calculate the average distance within the ball from the center of the ball
                Mul_Array = self.alldata[np.where(
                    np.sum((self.alldata[:, 1:-1] - self.granular_balls[i].center) ** 2, axis=1) ** 0.5 <= radie)[0], :]
                # Find all the points whose distance from the center of the sphere is less than the average distance radius
                if len(np.unique(self.granular_balls[i].data[:, 1:-1])) != 1 and len(set(Mul_Array[:, 0])) != 1:
                    # If the labels of these points are not unique, continue to divide the granular balls
                    split_clusters = self.granular_balls[i].split_clustering_2()
                    if len(split_clusters) == 1:
                        i += 1
                    elif len(split_clusters) < 1:
                        self.granular_balls.pop(i)
                        ll -= 1
                    else:
                        self.granular_balls[i] = split_clusters[0]
                        self.granular_balls.extend(split_clusters[1:])
                        ll += len(split_clusters) - 1
                else:
                    i += 1
            if i >= ll:
                break


def funtion(ball_list):
    Ball_list = ball_list
    for i in range(len(Ball_list)):
        ball_dis = np.sum((Ball_list[i].data[:, 1:-1] - Ball_list[i].center) ** 2, axis=1) ** 0.5  # Calculate the distance from the point to the center of the cluster
        Ball_list[i].radius = np.max(ball_dis)  # Take the average distance as the radius of the sphere
    Ball_list = sorted(Ball_list, key=lambda x: -x.radius, reverse=True)
    ll = len(Ball_list)  # Number of granular balls
    j = 0
    ball = []
    while True:
        if len(ball) == 0:
            ball.append([Ball_list[j].center, Ball_list[j].radius, Ball_list[j].label])
            j += 1
        else:
            flag = False
            for index, values in enumerate(ball):
                # If the labels of the two balls are different and the distance between the centers of the two balls is less than the sum of the radii of the two balls (the boundary overlaps)
                if values[2] != Ball_list[j].label and (np.sum((values[0] - Ball_list[j].center) ** 2) ** 0.5) <= (
                        values[1] + Ball_list[j].radius):
                    flag = True
                    split_clusters = Ball_list[j].split_clustering_2()  # Continue to divide the overlapping balls
                    if len(split_clusters) == 1:
                        j += 1
                    elif len(split_clusters) < 1:
                        Ball_list.pop(j)
                        ll -= 1
                    else:  # Save the divided granular balls and take the radius of the granular balls as the average distance
                        Ball_list[j] = split_clusters[0]
                        ball_dis = np.sum((Ball_list[j].data[:, 1:-1] - Ball_list[j].center) ** 2, axis=1) ** 0.5
                        Ball_list[j].radius = np.max(ball_dis)
                        Ball_list.extend(split_clusters[1:])
                        ll += len(split_clusters[1:])
                        for new_ball in split_clusters[1:]:
                            ball_dis = np.sum((new_ball.data[:, 1:-1] - new_ball.center) ** 2, axis=1) ** 0.5
                            new_ball.radius = np.max(ball_dis)

                    break
            if flag == False:
                ball.append([Ball_list[j].center, Ball_list[j].radius, Ball_list[j].label])
                j += 1
        if j >= ll:
            break
    return Ball_list


def fun(train, test):
    granular_balls = GBList(train, train)  # Build granular balls
    granular_balls.init_granular_balls()  # Initialize granular balls, divide granular balls according to purity
    ball_list = granular_balls.granular_balls
    Ball_list = funtion(ball_list)  # Continue to divide the granular balls with overlapping boundaries
    while True:
        init_center = []
        Ball_num1 = len(Ball_list)  # Count the number of granular balls
        for i in range(len(Ball_list)):
            init_center.append(Ball_list[i].center)

        ClusterLists = k_means(X=train[:, 1:-1], init=np.array(init_center), n_clusters=len(Ball_list))
        data_label = ClusterLists[1]
        ball_list = []
        for i in set(data_label):
            Cluster_data = train[data_label == i, :]
            ball_list.append(GranularBall(Cluster_data))
        Ball_list = funtion(ball_list)
        Ball_num2 = len(Ball_list)  # Number of granular balls after statistical division
        if Ball_num1 == Ball_num2:  # Stop if the number of granular balls no longer changes
            break
    for i in range(len(Ball_list)):  # Re-assign the radius of all granular balls from the maximum distance to the average distance
        ball_dis = np.sum((Ball_list[i].data[:, 1:-1] - Ball_list[i].center) ** 2, axis=1) ** 0.5
        Ball_list[i].radius = np.mean(ball_dis)

    # plot_gb(Ball_list)  # Visualize two-dimensional granular balls (using data.csv data set)
    count_num = nearest_knn(Ball_list, test)  # Test nearest neighbor accuracy

    return count_num


def nearest_knn(ball_list, test):
    # 测试最近邻精度
    num = []
    for i in range(len(test)):
        z_dis = float("inf")
        z_label = None
        for j in range(len(ball_list)):
            DIS = np.sum((test[i, 1:] - ball_list[j].center) ** 2) ** 0.5 - ball_list[j].radius
            if z_dis > DIS:
                z_dis = DIS
                z_label = ball_list[j].label
        if z_label == test[i, 0]:
            num.append(1)
        else:
            num.append(0)
    return num


def mean_std(a):
    # Calculate the mean and standard deviation of a one-dimensional array
    a = np.array(a)
    std = np.sqrt(((a - np.mean(a)) ** 2).sum() / (a.size - 1))
    return a.mean(), std


def plot_gb(granular_ball_list):
    # Visualizing two-dimensional data granular balls
    color = {0: 'r', 1: 'k', 2: 'g'}
    plt.figure(figsize=(5, 4))
    plt.axis([0, 1, 0, 0.7])
    for granular_ball in granular_ball_list:
        label, p = granular_ball.label, granular_ball.purity
        center, radius = granular_ball.center, granular_ball.radius
        data0 = granular_ball.data[granular_ball.data[:, 0] == 0]
        data1 = granular_ball.data[granular_ball.data[:, 0] == 1]
        data2 = granular_ball.data[granular_ball.data[:, 0] == 2]
        plt.plot(data0[:, 1], data0[:, 2], '.', color=color[0], markersize=5)
        plt.plot(data1[:, 1], data1[:, 2], '.', color=color[1], markersize=5)
        plt.plot(data2[:, 1], data2[:, 2], '.', color=color[2], markersize=5)

        theta = numpy.arange(0, 2 * numpy.pi, 0.01)
        x = center[0] + radius * numpy.cos(theta)
        y = center[1] + radius * numpy.sin(theta)

        plt.plot(x, y, color[label], linewidth=0.8)

        plt.plot(center[0], center[1], 'x', color=color[label])

    plt.show()


def main():
    keys = ['fourclass', 'votes', 'breastcancer', 'mushroom', 'svmguide1', 'svmguide3']
    for d in range(len(keys)):
        print(keys[d])
        df = pd.read_csv(r"UCI\\" + keys[d] + ".csv", header=None)  # Load data set
        data = df.values
        data[data[:, 0] == -1, 0] = 0
        numberSample, numberAttribute = data.shape
        minMax = MinMaxScaler()  # Normalize the data
        data = np.hstack((data[:, 0].reshape(numberSample, 1), minMax.fit_transform(data[:, 1:])))
        train_data = data[:, 1:]  # Data set characteristics
        train_target = data[:, 0]  # Data set label
        skf = StratifiedKFold(n_splits=10)  # 10-fold cross-validation divided data
        acc = []
        Acc, Std = 0, 0
        for i in range(10):  # Repeat the experiment 10 times
            Acc = 0
            for train_index, test_index in skf.split(train_data, train_target):
                X_train, X_test = train_data[train_index], train_data[test_index]
                y_train, y_test = train_target[train_index], train_target[test_index]
                numbertrain, numberatrain = X_train.shape
                numbertest, numberatest = X_test.shape
                train = np.hstack((y_train.reshape(numbertrain, 1), X_train))
                index = np.array(range(0, numbertrain)).reshape(numbertrain, 1)  # Index column
                train = np.hstack((train, index))  # Training set with index column
                test = np.hstack((y_test.reshape(numbertest, 1), X_test))  # Test set
                a = fun(train, test)
                avg, std = mean_std(a)
                Acc += avg
                Std += std
            print(Acc / 10)
            Acc_2 = round(Acc / 10, 3)
            acc.append(Acc_2)
        print('最高精度', np.max(acc), '平均精度', np.mean(acc))  # Take the highest and average accuracy in 10 experiments


if __name__ == '__main__':
    main()
