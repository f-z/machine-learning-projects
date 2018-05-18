import numpy as np
import matplotlib.pyplot as plt

# for reading csv file and importing data
from import_explore import import_csv

# for performing regression
from regression_train_test import train_and_test_split
from regression_train_test import train_and_test

from regression_models import ml_weights

# for fitting polynomial with multiple co-variates
from multi_poly_fit import multipolyfit

# for fitting polynomial with multiple co-variates
from sklearn.preprocessing import PolynomialFeatures


def polynomial_regression(name, delimiter, columns):
    """
    This function contains example code that demonstrates how to use the
    functions defined in poly_fit_base for fitting polynomial curves to data.
    """

    # choose number of data-points and sample a pair of vectors: the input
    # values and the corresponding target values
    degree = 2

    # importing using csv reader and storing as numpy array
    header, inputs, targets = import_csv(name, delimiter, columns)

    print("\n")

    # getting the train/test split
    train_part, test_part = train_and_test_split(inputs.shape[0], 0.25)

    # breaking the data into train and test parts
    # getting the indices of the train and test portion
    train_inputs = inputs[train_part, :]
    test_inputs = inputs[test_part, :]
    train_targets = targets[train_part]
    test_targets = targets[test_part]

    '''
    expanded_train_inputs = np.array(train_inputs, dtype=object);
    expanded_test_inputs = np.array(test_inputs, dtype=object);
    
        for i in range(train_inputs.shape[0]):
        for j in range(train_inputs.shape[1]):
            expanded_train_inputs[i, j] = []
            expanded_train_inputs[i, j].append(train_inputs[i, j] ** 0)
            expanded_train_inputs[i, j].append(train_inputs[i, j] ** 1)
            expanded_train_inputs[i, j].append(train_inputs[i, j] ** 2)

    train_inputs = np.array(expanded_train_inputs)

    for i in range(test_inputs.shape[0]):
        for j in range(test_inputs.shape[1]):
            expanded_test_inputs[i, j] = []
            expanded_test_inputs[i, j].append(test_inputs[i, j] ** 0)
            expanded_test_inputs[i, j].append(test_inputs[i, j] ** 1)
            expanded_test_inputs[i, j].append(test_inputs[i, j] ** 2)

    test_inputs = np.array(expanded_test_inputs)


    # need to unpack the array elements because they are lists at the moment

    new_train_inputs = np.zeros([train_inputs.shape[0], train_inputs.shape[1]*3])
    for i in range(train_inputs.shape[0]):
        new_train_inputs[i] = np.hstack(train_inputs[i])
    print(new_train_inputs.shape)
    
    for i in range(train_inputs.shape[0]):
        for j in range(train_inputs.shape[1]):
            for k in range(3):
                new_train_inputs[i][j + k] = train_inputs[i][j][k]
    '''
    for i in range(10):
        poly = PolynomialFeatures(i)
        processed_train_inputs = poly.fit_transform(train_inputs)
        processed_test_inputs = poly.fit_transform(test_inputs)

        # now training and evaluating the error on both sets
        # finding the optimal weights (depends on regularisation)
        # using simple least squares approach
        # weights = ml_weights(processed_train_inputs, train_targets, processed_test_inputs, test_targets)

        '''
        prediction_function = multipolyfit(train_inputs, train_targets, 2)
        train_predicts = prediction_function(train_inputs)
        test_predicts = prediction_function(test_inputs)
        '''

        # evaluate the error between the predictions and true targets on both sets
        train_error, test_error = train_and_test(processed_train_inputs, train_targets, processed_test_inputs, test_targets)

        print("Polynomial Regression Degree: %r" % i)
        print("\t(train_error, test_error) = %r" % ((train_error, test_error),))


def main(name, delimiter, columns, has_header=True, test_fraction=0.25):
    polynomial_regression(name, delimiter, columns)

    plt.show()


if __name__ == '__main__':
    import sys
    # this allows you to pass the file name as the first argument when you call
    # your script from the command line
    if len(sys.argv) == 1:
        # reverting to default parameters (red wine, ; delimiter, all features)
        main('../winequality-red.csv', ";", np.arange(0, 11))
    elif len(sys.argv) == 2:
        # passing the file name as the first argument
        main(sys.argv[1], ";", np.arange(0, 11))
    elif len(sys.argv) == 3:
        # passing the delimiter as the second argument
        main(sys.argv[1], sys.argv[2], np.arange(0, 11))
    elif len(sys.argv) == 4:
        # the third argument is a list of columns to use as input features
        # list is separated by ','
        custom_columns = list(map(int, sys.argv[3].split(",")))
        main(sys.argv[1], sys.argv[2], custom_columns)
