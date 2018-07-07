import numpy as np
import matplotlib.pyplot as plt

from import_explore import import_csv
from import_explore import normalise

# for performing regression
from regression_models import construct_rbf_feature_mapping
# for plotting results
from regression_plot import plot_train_test_errors
# two new functions for cross validation
from regression_train_test import create_cv_folds
from regression_train_test import cv_evaluation_linear_model
from regression_train_test import train_and_test_split
from regression_train_test import train_and_test_partition
from regression_train_test import train_and_test


def parameter_search_rbf(inputs, targets, test_fraction, folds):
    """
    """

    n = inputs.shape[0]

    # for the centres of the basis functions sample 10% of the data
    sample_fraction = 0.05
    p = (1-sample_fraction, sample_fraction)
    centres = inputs[np.random.choice([False, True], size=n, p=p), :]
    print("\ncentres.shape = %r" % (centres.shape,))

    scales = np.logspace(0, 4, 20)  # of the basis functions
    reg_params = np.logspace(-16, -1, 20)  # choices of regularisation strength
    # create empty 2d arrays to store the train and test errors
    train_mean_errors = np.empty((scales.size, reg_params.size))
    test_mean_errors = np.empty((scales.size, reg_params.size))

    # iterate over the scales
    for i, scale in enumerate(scales):
        # i is the index, scale is the corresponding scale
        # we must recreate the feature mapping each time for different scales
        feature_mapping = construct_rbf_feature_mapping(centres, scale)
        designmtx = feature_mapping(inputs)
        # partition the design matrix and targets into train and test

        # iterating over the regularisation parameters
        for j, reg_param in enumerate(reg_params):
            # j is the index, reg_param is the corresponding regularisation
            # parameter
            # train and test the data
            train_error, test_error = cv_evaluation_linear_model(
                designmtx, targets, folds, reg_param=reg_param)
            # store the train and test errors in our 2d arrays
            train_mean_errors[i, j] = np.mean(train_error)
            test_mean_errors[i, j] = np.mean(test_error)

    # we have a 2d array of train and test errors, we want to know the (i,j)
    # index of the best value
    best_i = np.argmin(np.argmin(test_mean_errors, axis=1))
    best_j = np.argmin(test_mean_errors[i, :])
    min_place = np.argmin(test_mean_errors)
    best_i_correct = (int)(min_place/test_mean_errors.shape[1])
    best_j_correct = min_place%test_mean_errors.shape[1]
    print("\nBest joint choice of parameters:")
    print(
        "\tscale %.2g and lambda = %.2g" % (scales[best_i_correct], reg_params[best_j_correct]))

    # now we can plot the error for different scales using the best
    # regularisation choice
    fig, ax = plot_train_test_errors(
        "scale", scales, train_mean_errors[:, best_j_correct], test_mean_errors[:, best_j_correct])
    ax.set_xscale('log')
    ax.set_title('Train vs Test Error Across Scales')
    fig.savefig("../plots/rbf_searching_scales.pdf", fmt="pdf")

    # ...and the error for  different regularisation choices given the best
    # scale choice
    fig, ax = plot_train_test_errors(
        "$\lambda$", reg_params, train_mean_errors[best_i_correct, :], test_mean_errors[best_i_correct, :])
    ax.set_xscale('log')
    ax.set_title('Train vs Test Error Across Reg Params')
    fig.savefig("../plots/rbf_searching_reg_params.pdf", fmt="pdf")
    '''
    # using the best parameters found above,
    # we now vary the number of centres and evaluate the performance
    reg_param = reg_params[best_j]
    scale = scales[best_i]
    n_centres_seq = np.arange(1, 20)
    train_errors = []
    test_errors = []
    for n_centres in n_centres_seq:
        # constructing the feature mapping anew for each number of centres
        centres = np.linspace(0, 1, n_centres)
        feature_mapping = construct_rbf_feature_mapping(centres, scale)
        design_matrix = feature_mapping(inputs)

        # evaluating the test and train error for the given regularisation parameter and scale
        train_error, test_error = cv_evaluation_linear_model(
            design_matrix, targets, folds, reg_param=reg_param)

        # collecting the errors
        train_errors.append(train_error)
        test_errors.append(test_error)

    # plotting the results
    fig, ax = plot_train_test_errors(
        "no. centres", n_centres_seq, train_errors, test_errors)
    ax.set_title('Train vs Test Error Across Centre Number')
    fig.savefig("../plots/rbf_searching_number_centres.pdf", fmt="pdf")
    '''

    return scales[best_i_correct], reg_params[best_j_correct]

def evaluate_reg_param(inputs, targets, folds, centres, scale, reg_params=None):
    """
      Evaluate then plot the performance of different regularisation parameters
    """

    # creating the feature mapping and then the design matrix
    feature_mapping = construct_rbf_feature_mapping(centres, scale)
    designmtx = feature_mapping(inputs)

    # choose a range of regularisation parameters
    if reg_params is None:
        reg_params = np.logspace(-15, 2, 20)  # choices of regularisation strength

    num_values = reg_params.size
    num_folds = len(folds)
    # create some arrays to store results
    train_mean_errors = np.zeros(num_values)
    test_mean_errors = np.zeros(num_values)
    train_stdev_errors = np.zeros(num_values)
    test_stdev_errors = np.zeros(num_values)
    #    
    for r, reg_param in enumerate(reg_params):
        # r is the index of reg_param, reg_param is the regularisation parameter
        # cross validate with this regularisation parameter
        train_errors, test_errors = cv_evaluation_linear_model(
            designmtx, targets, folds, reg_param=reg_param)
        # we're interested in the average (mean) training and testing errors
        train_mean_error = np.mean(train_errors)
        test_mean_error = np.mean(test_errors)
        train_stdev_error = np.std(train_errors)
        test_stdev_error = np.std(test_errors)
        # store the results
        train_mean_errors[r] = train_mean_error
        test_mean_errors[r] = test_mean_error
        train_stdev_errors[r] = train_stdev_error
        test_stdev_errors[r] = test_stdev_error

    # Now plot the results
    fig, ax = plot_train_test_errors(
        "$\lambda$", reg_params, train_mean_errors, test_mean_errors)
    # Here we plot the error ranges too: mean plus/minus 1 standard error.
    # 1 standard error is the standard deviation divided by sqrt(n) where
    # n is the number of samples. 
    # (There are other choices for error bars.)
    # train error bars
    lower = train_mean_errors - train_stdev_errors/np.sqrt(num_folds)
    upper = train_mean_errors + train_stdev_errors/np.sqrt(num_folds)
    ax.fill_between(reg_params, lower, upper, alpha=0.2, color='b')
    # test error bars
    lower = test_mean_errors - test_stdev_errors/np.sqrt(num_folds)
    upper = test_mean_errors + test_stdev_errors/np.sqrt(num_folds)
    ax.fill_between(reg_params, lower, upper, alpha=0.2, color='r')
    ax.set_xscale('log')
    # ax.set_xlim([0, 0.02])

    ax.set_title('Train vs Test Error Across Reg Params With Cross-Validation')
    fig.savefig("../plots/rbf_searching_reg_params_cross_validation.pdf", fmt="pdf")


def evaluate_scale(inputs, targets, folds, centres, reg_param, scales=None):
    """
    evaluate then plot the performance of different basis function scales
    """
    # choose a range of scales
    if scales is None:
        scales = np.logspace(0, 6, 20)  # of the basis functions
    #
    num_values = scales.size
    num_folds = len(folds)
    # create some arrays to store results
    train_mean_errors = np.zeros(num_values)
    test_mean_errors = np.zeros(num_values)
    train_stdev_errors = np.zeros(num_values)
    test_stdev_errors = np.zeros(num_values)
    #    
    for s, scale in enumerate(scales):
        feature_mapping = construct_rbf_feature_mapping(centres,scale)
        designmtx = feature_mapping(inputs) 
        # r is the index of reg_param, reg_param is the regularisation parameter
        # cross validate with this regularisation parameter
        train_errors, test_errors = cv_evaluation_linear_model(
            designmtx, targets, folds, reg_param=reg_param)
        # we're interested in the average (mean) training and testing errors
        train_mean_error = np.mean(train_errors)
        test_mean_error = np.mean(test_errors)
        train_stdev_error = np.std(train_errors)
        test_stdev_error = np.std(test_errors)
        # store the results
        train_mean_errors[s] = train_mean_error
        test_mean_errors[s] = test_mean_error
        train_stdev_errors[s] = train_stdev_error
        test_stdev_errors[s] = test_stdev_error

    # Now plot the results
    fig, ax = plot_train_test_errors(
        "scale", scales, train_mean_errors, test_mean_errors)
    # Here we plot the error ranges too: mean plus/minus 1 standard error.
    # 1 standard error is the standard deviation divided by sqrt(n) where
    # n is the number of samples. 
    # (There are other choices for error bars.)
    # train error bars
    lower = train_mean_errors - train_stdev_errors/np.sqrt(num_folds)
    upper = train_mean_errors + train_stdev_errors/np.sqrt(num_folds)
    ax.fill_between(scales, lower, upper, alpha=0.2, color='b')
    # test error bars
    lower = test_mean_errors - test_stdev_errors/np.sqrt(num_folds)
    upper = test_mean_errors + test_stdev_errors/np.sqrt(num_folds)
    ax.fill_between(scales, lower, upper, alpha=0.2, color='r')
    ax.set_xscale('log')
    # ax.set_xlim([0, 100])

    ax.set_title('Train vs Test Error Across Scales With Cross-Validation')
    fig.savefig("../plots/rbf_searching_scales_cross_validation.pdf", fmt="pdf")


def evaluate_num_centres(
        inputs, targets, folds, scale, reg_param, num_centres_sequence=None):
    """
      Evaluate then plot the performance of different numbers of basis
      function centres.
    """

    # choose a range of numbers of centres
    if num_centres_sequence is None:
        num_centres_sequence = np.arange(1, 20)
    num_values = num_centres_sequence.size
    num_folds = len(folds)
    #
    # create some arrays to store results
    train_mean_errors = np.zeros(num_values)
    test_mean_errors = np.zeros(num_values)
    train_stdev_errors = np.zeros(num_values)
    test_stdev_errors = np.zeros(num_values)
    #    
    # run the experiments
    for c, num_centres in enumerate(num_centres_sequence):
        centres = np.linspace(0, 1, num_centres)
        feature_mapping = construct_rbf_feature_mapping(centres, scale)
        designmtx = feature_mapping(inputs) 
        # r is the index of reg_param, reg_param is the regularisation parameter
        # cross validate with this regularisation parameter
        train_errors, test_errors = cv_evaluation_linear_model(
            designmtx, targets, folds, reg_param=reg_param)
        # we're interested in the average (mean) training and testing errors
        train_mean_error = np.mean(train_errors)
        test_mean_error = np.mean(test_errors)
        train_stdev_error = np.std(train_errors)
        test_stdev_error = np.std(test_errors)
        # store the results
        train_mean_errors[c] = train_mean_error
        test_mean_errors[c] = test_mean_error
        train_stdev_errors[c] = train_stdev_error
        test_stdev_errors[c] = test_stdev_error
    #
    # Now plot the results
    fig, ax = plot_train_test_errors(
        "no. centres", num_centres_sequence, train_mean_errors, test_mean_errors)
    # Here we plot the error ranges too: mean plus/minus 1 standard error.
    # 1 standard error is the standard deviation divided by sqrt(n) where
    # n is the number of samples. 
    # (There are other choices for error bars.)
    # train error bars
    lower = train_mean_errors - train_stdev_errors/np.sqrt(num_folds)
    upper = train_mean_errors + train_stdev_errors/np.sqrt(num_folds)
    ax.fill_between(num_centres_sequence, lower, upper, alpha=0.2, color='b')
    # test error bars
    lower = test_mean_errors - test_stdev_errors/np.sqrt(num_folds)
    upper = test_mean_errors + test_stdev_errors/np.sqrt(num_folds)
    ax.fill_between(num_centres_sequence, lower, upper, alpha=0.2, color='r')
    ax.set_title('Train vs Test Error Across Centre Number With Cross-Validation')
    fig.savefig("../plots/rbf_searching_number_centres_cross_validation.pdf", fmt="pdf")


def main(name, delimiter, columns, has_header=True, test_fraction=0.25):
    """
    This function contains example code that demonstrates how to use the 
    functions defined in poly_fit_base for fitting polynomial curves to data.
    """

    # importing using csv reader and storing as numpy array
    header, data = import_csv(name, delimiter)

    print("\n")

    n = data.shape[1]
    # deleting the last column (quality) from inputs
    inputs = np.delete(data, n-1, 1)
    # assigning it as targets instead
    targets = data[:, n-1]

    inputs = normalise(inputs)

    # specifying the centres and scale of some rbf basis functions
    centres = inputs[np.random.choice([False, True], size=inputs.shape[0], p=[0.90, 0.10]), :]

    # the width (analogous to standard deviation) of the basis functions
    scale = 8.5

    # getting the cross-validation folds
    num_folds = 5
    folds = create_cv_folds(data.shape[0], num_folds)

    scale, reg_param = parameter_search_rbf(inputs, targets, test_fraction, folds)

    # evaluating then plotting the performance of different reg params
    evaluate_reg_param(inputs, targets, folds, centres, scale)

    # we found that reg params around 0.01 are optimal
    # evaluating and plotting the performance of different scales
    evaluate_scale(inputs, targets, folds, centres, reg_param)
    # evaluating then plotting the performance of different numbers of basis
    # function centres.
    evaluate_num_centres(inputs, targets, folds, scale, reg_param)

    plt.show()


if __name__ == '__main__':
    import sys

    # default columns
    columns = [0-10]

    if len(sys.argv) == 1:
        main('../winequality-red.csv', ";", columns)  # calls the main function with the default arguments
    elif len(sys.argv) == 2:
        # assumes that the first argument is the input filename/path
        main(sys.argv[1], ";", columns)
    elif len(sys.argv) == 3:
        # assumes that the second argument is the data delimiter
        main(sys.argv[1], sys.argv[2], columns)
    elif len(sys.argv) == 4:
        # assumes that the third argument is the list of columns to import
        columns = list(map(int, sys.argv[3].split(",")))
        main(sys.argv[1], sys.argv[2], columns)
