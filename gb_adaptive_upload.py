import collections
import time
import scipy.io
import random
import numpy
import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import k_means

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
    data_no_label = gb[:, 1:3]
    # print(data_no_label)
    center = data_no_label.mean(axis=0)
    radius_mean = np.mean((((data_no_label - center) ** 2).sum(axis=1) ** 0.5))
    # radius_max = numpy.max((((data_no_label - center) ** 2).sum(axis=1) ** 0.5))
    return center, radius_mean


# Calculate distance
def calculate_distances(data, p):
    return ((data - p) ** 2).sum(axis=0) ** 0.5


# draw granular-balls
def plot_gb(granular_ball_list, plt_type=0):
    color = {-1: 'r', 1: 'k', 0: 'b'}
    plt.figure(figsize=(5, 4))
    plt.axis([-1.2, 1.2, -1, 1])
    ball_num_str = str(len(granular_ball_list))
    for granular_ball in granular_ball_list:
        label, p = get_label_and_purity(granular_ball)
        center, radius = calculate_center_and_radius(granular_ball)

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

# Determining the splitting conditions of granular-balls
def splits(purity_init, gb_dict):
    gb_dict_new = {}
    gb_dict_temp = {}
    first = 1
    purity_init_temp = 0
    while True:
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
        # print('gb_dict_temp:', gb_dict_temp.keys())
        for key in gb_dict_temp.keys():
            # print(key)
            gb_single = {}
            after_purity = []  # The purity of the child ball
            weight_p = 0  # Weighted Purity Sum of the child ball
            # Fetch dictionary data, including keys and values
            gb_single[key] = gb_dict_temp[key]
            # print('gb_single:', gb_single.keys())

            gb = gb_single[key][0]
            p = get_label_and_purity(gb)[1]
            # print('p:', p)

            if len(gb) == 1:
                gb_dict_new.update(gb_single)
                continue
            gb_single_temp = gb_single.copy()
            gb_dict_re = splits_ball(gb_single).copy()

            if first == 1:
                for key0 in gb_dict_re.keys():
                    purity_init_temp = max(purity_init_temp, get_label_and_purity(gb_dict_re[key0][0])[1])
                purity_init = purity_init_temp
                first = 0
            # print('gb_dict_re:', gb_dict_re.keys())
            # for key0 in gb_dict_re.keys():
            #     after_purity.append(get_label_and_purity(gb_dict_re[key0][0])[1])
            # # print(after_purity)
            # average_p = np.mean(after_purity)
            for key0 in gb_dict_re.keys():
                weight_p = weight_p + get_label_and_purity(gb_dict_re[key0][0])[1] * (
                        len(gb_dict_re[key0][0]) / len(gb))
            # The weighted purity sum of the child balls is greater than the purity of the parent ball,
            # or the weighted purity sum of the child balls is less than the lower bound of purity
            if p <= purity_init or weight_p > p:
                gb_dict_new.update(gb_dict_re)
            else:
                gb_dict_new.update(gb_single_temp)

        # gb_plot(gb_dict_new)
        gb_dict_new = isOverlap(gb_dict_new)
        # gb_plot(gb_dict_new)
        # record the number after a division
        ball_number_2 = len(gb_dict_new)
        # The number of granular-balls is the same as the number of granular-balls last divided, that is, it will not change
        if ball_number_1 == ball_number_2:
            break

        # draw granular-balls
        # gb_plot(gb_dict_new)

    # draw granular-balls
    # gb_plot(gb_dict_new)
    # print(gb_dict_new)
    return gb_dict_new


# de-overlap
def isOverlap(gb_dict):
    Flag = True
    later_dict = gb_dict.copy()
    while True:
        ball_number_1 = len(gb_dict)
        centers = []  # centers
        keys = []  # keys
        dict_overlap = {}  # overlaped balls
        center_radius = {}  # {center:[center, gb, max_distances, radius]}
        for key in later_dict.keys():
            center, radius_mean = calculate_center_and_radius(later_dict[key][0])
            center_radius[key] = [center, later_dict[key][0], later_dict[key][1], radius_mean]
            center_temp = []
            keys.append(key)
            for center_split in key.split('_'):
                center_temp.append(float(center_split))
            centers.append(center_temp)
        centers = np.array(centers)

        # The first division uses the incoming granular-ball parameters, and the next uses only overlapping granules
        if Flag:
            later_dict = {}
            Flag = False
        for i, center01 in enumerate(centers):
            for j, center02 in enumerate(centers):
                if i < j and center01[0] != center02[0]:
                    # If the labels of the two balls are different and the distance between the centers of
                    # the two balls is less than the sum of the radii of the two balls (the boundaries overlap)
                    if calculate_distances(center_radius[keys[i]][0], center_radius[keys[j]][0]) < \
                            center_radius[keys[i]][3] + center_radius[keys[j]][3]:
                        dict_overlap[keys[i]] = center_radius[keys[i]][1:3]
                        dict_overlap[keys[j]] = center_radius[keys[j]][1:3]

        # gb_plot(gb_dict)
        # print('dict_overlap:', dict_overlap.keys())
        # When the number of overlapping granular-balls is 0, return
        if len(dict_overlap) == 0:
            gb_dict.update(later_dict)
            ball_number_2 = len(gb_dict)
            if ball_number_1 != ball_number_2:
                Flag = True
                later_dict = gb_dict.copy()
            else:
                return gb_dict
        gb_dict_single = dict_overlap.copy()  # Copy a temporary list, and then traverse the value
        for i in range(len(gb_dict_single)):
            gb_single = {}
            # Get dictionary data, including key values
            dict_temp = gb_dict_single.popitem()
            gb_single[dict_temp[0]] = dict_temp[1]
            # print('gb_single:', gb_single)
            # If the granular-ball is a single point, do not continue to divide
            if len(dict_temp[1][0]) == 1:
                later_dict.update(gb_single)
                continue
            gb_dict_new = splits_ball(gb_single).copy()
            # print('gb_dict_new:', gb_dict_new.keys())
            later_dict.update(gb_dict_new)
        # print('later_dict:', later_dict.keys())


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
    np.set_printoptions(suppress=True) # ignore warning
    keys = ['fourclass', 'svmguide1', 'diabetes', 'breastcancer', 'creditApproval',
            'votes', 'svmguide3', 'sonar', 'splice', 'mushrooms']
    for d in range(len(keys)):
        print(keys[d])
        times = 0
        num_gb = 0
        data_mat = scipy.io.loadmat(r'dataset16.mat')  # Load dataset
        data = data_mat[keys[d]]

        # data deduplication:Remove duplicate data and data with different labels but the same attributes
        data = numpy.unique(data, axis=0)
        # print(len(data))
        data_temp = []
        data_list = data.tolist()
        data = []
        for data_single in data_list:
            if data_single[1:] not in data_temp:
                data_temp.append(data_single[1:])
                data.append(data_single)
        data = np.array(data)
        # print(data)
        for i in range(10):  # Repeat 10 times to get the average precision
            # record start time
            start = time.time()
            # print('start time', start)
            ball_list = []


            # Purity Lower Boundary
            purity_init = get_label_and_purity(data)[1]

            # initial random center
            center_init = data[random.randint(0, len(data) - 1), :]
            # center_init = data[:, 1:3].mean(axis=0)
            # print(center_init)

            distance_init = []
            for feature in data:
                # The distance from the initial center to the data
                distance_init.append(calculate_distances(feature[1:], center_init[1:]))
            # print('distance_init:', len(distance_init))

            # packed into a dictionary:{center: [gb, distances]}
            gb_dict = {}
            gb_dict_key = str(center_init.tolist()[0])
            for j in range(1, len(center_init)):
                gb_dict_key += '_' + str(center_init.tolist()[j])
            gb_dict_value = [data, distance_init]
            gb_dict[gb_dict_key] = gb_dict_value

            # draw granular-balls
            # gb_plot(gb_dict)

            # Classification
            gb_dict = splits(purity_init=purity_init, gb_dict=gb_dict)
            k_centers = []
            splits_k = len(gb_dict)
            for key in gb_dict.keys():
                k_center = []
                for k in key.split('_'):
                    k_center.append(float(k))
                k_centers.append(k_center[1:])
            # print(np.array(k_centers))
            # Perform a global division
            label_cluster = k_means(X=data[:, 1:], n_clusters=splits_k, n_init=1, init=np.array(k_centers), random_state=5)[1]
            for single_label in range(splits_k):
                ball_list.append(data[label_cluster == single_label, :])

            # record number
            num_gb += len(ball_list)
            # record end time
            end = time.time()
            times = (end - start) + times

            # draw granular-balls
            # gb_plot(gb_dict)
            plot_gb(ball_list)
            # plot_gb(ball_list, 1)

            # print('Number of granular-balls：', len(gb_dict))
            # print('time', round((end - start) * 1000, 0))
            # break
        print('Number of granular-balls：', round(num_gb / 10, 0))
        print('total average time：%s' % (round(times / 10 * 1000, 0)))
        break


if __name__ == '__main__':
    main()
