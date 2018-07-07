import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt


def import_csv(name, delimiter, columns):
    with open(name, 'r') as file:
        data_reader = csv.reader(file, delimiter=delimiter)

        # importing the header line separately
        # and printing it to screen
        header = next(data_reader)
        # print("\n\nImporting data with fields:\n\t" + ",".join(header))

        # creating an empty list to store each row of data
        data = []

        for row in data_reader:
            # for each row of data 
            # converting each element (from string) to float type
            row_of_floats = list(map(float, row))

            # now storing in our data list
            data.append(row_of_floats)

        # print("There are %d entries." % len(data))

        # converting the data (list object) into a numpy array
        data_as_array = np.array(data)

        n = data_as_array.shape[1]
        # deleting the last column (quality) from inputs
        inputs = np.delete(data_as_array, n - 1, 1)
        # assigning it as targets instead
        targets = data_as_array[:, n - 1]

        column = 0
        while column < inputs.shape[1]:
            if column in columns:
                column += 1
            else:
                inputs = np.delete(inputs, column, 1)

        if columns is not None and header is not None:
            # thinning the associated field names if necessary
            header = [header[c] for c in columns]

        # returning this array to caller
        return header, inputs, targets


def import_pandas(name):
    with open(name, 'r') as file:
        data_frame = pd.read_csv(file, sep=';')

        return data_frame


def standardise(inputs):
    # inspecting the input data a little more
    # and then standardising them
    # meaning radial basis functions are more helpful
    # creating a copy to standardise and return
    copy = np.copy(inputs)

    for column in range(copy.shape[1]):
        mean = np.mean(copy[:, column])
        st_deviation = np.std(copy[:, column])
        # print("column %r" % column)
        # print("mean = %r" % (np.mean(inputs[:, column])))
        # print("std = %r" % (np.std(inputs[:, column])))
        copy[:, column] = [(copy[:, column][row] - mean) / st_deviation for row in range(len(copy[:, column]))]

    print("\n")

    return copy


def detecting_outliers_based_on_z_score(xs):
    default_threshold = 3
    mean = np.mean(xs)
    st_dev = np.std(xs)
    z_scores = [(x - mean) / st_dev for x in xs]

    return np.where(np.abs(z_scores) > default_threshold)


def histogram(fig, bins, header, data, column):
    # creating a single axis on the supplied figure
    ax = fig.add_subplot(2, 2, (column+1) % 5)

    # x coordinates are the specified column of the data
    xs = data[:, column]

    # plotting a histogram with a certain number of bins
    ax.hist(xs, bins)

    # setting appropriate labels
    ax.set_xlabel(header[column])
    ax.set_ylabel("bin frequency")

    # improving spacing/layout
    fig.tight_layout()


def bar_chart(data):
    series = pd.Series(data)
    value_count = series.value_counts()
    value_count = value_count.sort_index()

    return value_count.plot(kind='bar', title='Wine Quality')


def scatter_plot(fig, header, data, column1, column2):
    # creating a single axis on the supplied figure
    ax = fig.add_subplot(1, 1, 1)

    # x coordinates are the first specified column of the data
    xs = data[:, column1]

    # y coordinates are the second specified column of the data
    ys = data[:, column2]

    # plotting
    ax.plot(xs, ys, 'o', markersize=1)
    ax.set_xlabel(header[column1])
    ax.set_ylabel(header[column2])


def main(name, delimiter, columns):
    # setting a seed to get the same pseudo-random results every time
    np.random.seed(30)

    # importing using csv reader and storing as numpy array
    header, inputs, targets = import_csv(name, delimiter, columns)

    # importing using pandas and storing as data frame
    # data_frame = import_pandas(name)
    # print("\n")

    # print(data_frame.describe())
    print("\n")

    # print(data_frame.corr())

    '''
    # creating an empty figure object
    acidity_figure = plt.figure()
    histogram(acidity_figure, 20, header, inputs, 0)
    histogram(acidity_figure, 20, header, inputs, 1)
    histogram(acidity_figure, 20, header, inputs, 2)

    # saving as pdf
    acidity_figure.savefig("../plots/exploratory/acidity_histogram.png", fmt="png")

    sulfur_dioxide_figure = plt.figure()
    histogram(sulfur_dioxide_figure, 20, header, inputs, 5)
    histogram(sulfur_dioxide_figure, 20, header, inputs, 6)
    sulfur_dioxide_figure.savefig("../plots/exploratory/sulfur_dioxide_histogram.png", fmt="png")
    '''

    quality_figure = plt.figure()
    chart = bar_chart(targets)
    chart.set(xlabel="quality rating", ylabel="count")

    quality_figure.savefig("../plots/exploratory/quality_bar_chart.png", fmt="png")

    plt.show()

    return header, inputs, targets
