import numpy as np
import matplotlib.pyplot as plt
import csv
import random
import math
import operator

"""
This code is adapted from:
# https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/.
"""


def load_data_set(filename, split, training_set=[], test_set=[]):
    """
    The load_data_set loads the data from the csv file and split data into training
    and test set based on the split input. split should be in the range 0f  0-1,
    split * 100 means the percentage of data that got splited into the training set.
    """

    with open(filename, 'r') as csv_file:
        lines = csv.reader(csv_file, delimiter=";")
        next(lines)
        data_set = list(lines)
        for x in range(len(data_set)-1):
            for y in range(4):
                data_set[x][y] = float(data_set[x][y])
            if random.random() < split:
                training_set.append(data_set[x])
            else:
                test_set.append(data_set[x])


def euclidean_distance(instance1, instance2, length):
    """
    The euclideanDistance method calculates the distance between 2 data sets using euclideanDistance.
    """

    distance = 0
    for x in range(length):
        distance += pow((float(instance1[x]) - float(instance2[x])), 2)
    return math.sqrt(distance)


def get_neighbours(training_set, test_instance, k):
    """
    The getNeighbors method returns the k nearest neighbours,
    based on the distance calculated by euclidean_distance method.
    """

    distances = []
    length = len(test_instance)-1
    for x in range(len(training_set)):
        dist = euclidean_distance(test_instance, training_set[x], length)
        distances.append((training_set[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])

    return neighbors


def get_response(neighbors):
    """
    This method returns the average quality from the k-nearest neighbours.
    """

    length = len(neighbors)
    quality = []

    for x in range(0, length):
        c = neighbors[x][-1]
        quality.append(float(c))

    result = np.mean(quality)

    return result


def get_test_error(test_set, predictions):
    """
    This method calculates the mean square error for the test set and returns the result.
    """

    err = []
    mean_err = []

    for x in range(0, len(predictions)):
        error = (float(test_set[x][-1]) - predictions[x]) ** 2
        mean_err.append(error)
        ems = np.sqrt(error)
        error_rate=ems/float(test_set[x][-1])
        err.append(error_rate)

    total = np.sum(mean_err)
    total /= len(predictions)

    return total


def test_error(name):
    """
    The test error method prints out the final result for test error and plots it.
    """

    # setting a seed to get the same pseudo-random results every time
    np.random.seed(30)

    # prepare data
    training_set = []
    test_set = []
    split = 0.80
    load_data_set(name, split, training_set, test_set)

    print('\nPerforming kNN regression:\n')
    print('Train set: ' + str(len(training_set)))
    print('Test set: ' + str(len(test_set)))

    # generating predictions
    k_value = []
    a_value = []
    for k in range(1, 10):
        k_value.append(k)
        predictions = []
        for x in range(len(test_set)):
            neighbors = get_neighbours(training_set, test_set[x], k)
            result = get_response(neighbors)
            predictions.append(result)
            # print('> predicted=' + repr(result) + ', actual=' + repr(test_set[x][-1]))
        error_rate = get_test_error(test_set, predictions)
        print("error rate: %r" % error_rate)
        a_value.append(float(error_rate))
    print("k values:")
    print(k_value)
    print("a values:")
    print(a_value)
    fig = plt.figure()
    fig.suptitle("Test Error Against k ")
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(k_value, a_value, '-', markersize=1)
    ax.set_xlabel("k")
    ax.set_ylabel("test error")
    fig.savefig('../plots/knn/test_error_vs_k.png', fmt='png')
    fig.show()
    optimal_k=k_value[a_value.index(min(a_value))]
    print("optimal k: %r" % optimal_k)


def cross_validation(name):
    """
    The test error method prints out the final result for test error and plots it.
    """
    np.random.seed(30)
    folds = 5
    split = 0.75
    k_value = []
    cross_validation = []
    files = open(name, 'r')
    lines = csv.reader(files, delimiter=";")
    header = next(lines)
    # creating an empty list to store each row of data
    data = []
    for row in lines:
            # for each row of data 
            # converting each element (from string) to float type
            row_of_floats = list(map(float, row))
            # now storing in our data list
            data.append(row_of_floats)
        # print("There are %d entries." % len(data))
        # converting the data (list object) into a numpy array
    data_as_array = np.array(data)
    length = round(len(data_as_array)/folds)
    average = [] 
    for k in range(1, 20):
        k_value.append(k)   
        for y in range(1, folds):
            test_set = data_as_array[int(y*length):int((y+1)*length)]
            training_set = np.delete(data_as_array,[i for i in range(int(y*length), int((y+1)*length))],0)
            error = []
            pred = []
            for x in range(1, 50):
                neighbors = get_neighbours(training_set, test_set[x], k)
                result = get_response(neighbors)
                pred.append(result)
            error_rate = get_test_error(test_set,pred)
            error.append(error_rate)
        average.append(error)
        cross_validation.append(np.mean(error))
        # print(np.mean(error))
    fig1 = plt.figure()
    fig1.suptitle("Cross validation test error MSE against k ")
    ax1=fig1.add_subplot(1, 1, 1)
    ax1.set_xlabel("k")
    ax1.set_ylabel("test error MSE")
    ax1.plot(k_value, cross_validation, '-', markersize=1)
    fig1.show
    plt.show()
