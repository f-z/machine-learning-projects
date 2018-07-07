import numpy as np
import matplotlib.pyplot as plt

from import_explore import standardise

from regression_models import construct_rbf_feature_mapping
from regression_train_test import train_and_test_split
from regression_train_test import train_and_test_partition
from regression_train_test import train_and_test
from regression_train_test import simple_evaluation_linear_model

from regression_plot import plot_train_test_errors


def main(inputs, targets, test_fraction=0.20):
    """
    To be called when the script is run. This function fits and plots imported data (given a filename is
    provided). Data is multidimensional and real-valued and is fit
    with maximum likelihood 2d gaussian.

    parameters
    ----------
    name -- filename/path of data file.
    delimiter -- delimiter of data values
    has_header -- does the data-file have a header line
    columns -- a list of integers specifying which columns of the file to import
        (counting from 0)    
    """
    # setting a seed to get the same pseudo-random results every time
    np.random.seed(30)

    print("\n")

    # the linear performance
    train_error_linear, test_error_linear, residual_variance = simple_evaluation_linear_model(
        inputs, targets, test_fraction)

    print("\n")
    print("Linear Regression:")
    print("\t(train_error, test_error) = %r" % ((train_error_linear, test_error_linear),))
    print("\tresidual (training error) variance = %r" % residual_variance)
    print("\n")

    std_inputs = standardise(inputs)

    evaluate_rbf_for_various_reg_params(std_inputs, targets, test_fraction, test_error_linear)

    best_scales = []
    best_reg_params = []
    best_no_centres = []

    # increase this to brute-force run the simulations multiple times
    # and negate the effects of choosing the centres as a random 10% of the input sample
    # warning: this will take a very long time as the range increases
    for i in range(1):
        current_best_scale, current_best_reg_param, current_best_no_centres = parameter_search_rbf(
            std_inputs, targets, test_fraction, test_error_linear)
        best_scales.append(current_best_scale)
        best_reg_params.append(current_best_reg_param)
        best_no_centres.append(current_best_no_centres)

    print("\nBest parameter values after %r iteration(s):" % str(i+1))
    print("\t(best scale) = %r" % (np.mean(best_scales)))
    print("\t(best reg param) = %r" % (np.mean(best_reg_params)))
    print("\t(best proportion of centres) = %r" % (np.mean(best_no_centres)))

    plt.show()

    return test_error_linear, np.mean(best_scales), np.mean(best_reg_params), np.mean(best_no_centres)


def evaluate_rbf_for_various_reg_params(
        inputs, targets, test_fraction, test_error_linear):
    # for rbf feature mappings
    # for the centres of the basis functions choose 10% of the data
    n = inputs.shape[0]
    centres = inputs[np.random.choice([False, True], size=n, p=[0.90, 0.10]), :]
    print("centres shape = %r" % (centres.shape,))

    # the width (analogous to standard deviation) of the basis functions
    scale = 6.7  # of the basis functions

    feature_mapping = construct_rbf_feature_mapping(centres, scale)
    design_matrix = feature_mapping(inputs)

    train_part, test_part = train_and_test_split(n, test_fraction=test_fraction)
    train_design_matrix, train_targets, test_design_matrix, test_targets = \
        train_and_test_partition(
            design_matrix, targets, train_part, test_part)

    # outputting the shapes of the train and test parts for debugging
    print("training design matrix shape = %r" % (train_design_matrix.shape,))
    print("testing design matrix shape = %r" % (test_design_matrix.shape,))
    print("training targets shape = %r" % (train_targets.shape,))
    print("testing targets shape = %r" % (test_targets.shape,) + "\n")

    # the rbf feature mapping performance
    reg_params = np.logspace(-15, 6, 30)
    train_errors = []
    test_errors = []

    print("Evaluating reg. parameters...")

    for reg_param in reg_params:
        # print("Evaluating reg. parameter " + str(reg_param))
        train_error, test_error, residual_variance = simple_evaluation_linear_model(
            design_matrix, targets, test_fraction=test_fraction, reg_param=reg_param)
        train_errors.append(train_error)
        test_errors.append(test_error)

    fig, ax = plot_train_test_errors(
        "$\lambda$", reg_params, train_errors, test_errors, test_error_linear)

    ax.set_xscale('log')
    ax.set_title('Evaluating RBF Performance')
    fig.savefig("../plots/rbf/rbf_vs_linear.png", fmt="png")


def parameter_search_rbf(inputs, targets, test_fraction, test_error_linear):

    # input data set size
    n = inputs.shape[0]
    # run all experiments on the same train-test split of the data
    train_part, test_part = train_and_test_split(n, test_fraction=test_fraction)

    # for the centres of the basis functions sample 10% of the data
    sample_fraction = 0.10
    p = (1-sample_fraction, sample_fraction)
    centres = inputs[np.random.choice([False, True], size=n, p=p), :]
    print("\ncentres.shape = %r" % (centres.shape,))

    scales = np.logspace(0, 8, 30)  # of the basis functions
    reg_params = np.logspace(-15, 6, 30)  # choices of regularisation strength
    # create empty 2d arrays to store the train and test errors
    train_errors = np.empty((scales.size, reg_params.size))
    test_errors = np.empty((scales.size, reg_params.size))

    # iterate over the scales
    for i, scale in enumerate(scales):
        # i is the index, scale is the corresponding scale
        # we must recreate the feature mapping each time for different scales
        feature_mapping = construct_rbf_feature_mapping(centres, scale)
        designmtx = feature_mapping(inputs)
        # partition the design matrix and targets into train and test
        train_designmtx, train_targets, test_designmtx, test_targets = \
            train_and_test_partition(
                designmtx, targets, train_part, test_part)

        # iterating over the regularisation parameters
        for j, reg_param in enumerate(reg_params):
            # j is the index, reg_param is the corresponding regularisation
            # parameter
            # train and test the data
            train_error, test_error, residual_variance = train_and_test(
                train_designmtx, train_targets, test_designmtx, test_targets,
                reg_param=reg_param)
            # store the train and test errors in our 2d arrays
            train_errors[i, j] = train_error
            test_errors[i, j] = test_error

    # we have a 2d array of train and test errors, we want to know the (i,j)
    # index of the best value
    minimum_error = test_errors[test_errors != 0].min()
    ij_minimum = np.where(test_errors == minimum_error)
    # extracting
    ij_minimum = tuple([i.item() for i in ij_minimum])
    best_i = ij_minimum[0]
    best_j = ij_minimum[1]

    # other method to find best i and j
    # best_j = np.argmin(np.min(test_errors, axis=1))
    # best_i = np.argmin(test_errors[:, best_j])

    print("\nBest joint choice of parameters:")
    print(
        "\tscale %.2g and lambda = %.2g" % (scales[best_i], reg_params[best_j]))

    # now we can plot the error for different scales using the best
    # regularisation choice
    fig, ax = plot_train_test_errors(
        "scale", scales, train_errors[:, best_j], test_errors[:, best_j], test_error_linear)

    ax.set_xscale('log')
    ax.set_title('Train vs Test Error across Scales')
    fig.savefig("../plots/rbf/rbf_searching_scales.png", fmt="png")

    # ...and the error for  different regularisation choices given the best
    # scale choice
    fig, ax = plot_train_test_errors(
        "$\lambda$", reg_params, train_errors[best_i, :], test_errors[best_i, :], test_error_linear)

    ax.set_xscale('log')
    ax.set_title('Train vs Test Error across Reg. Parameters')
    fig.savefig("../plots/rbf/rbf_searching_reg_params.png", fmt="png")

    # using the best parameters found above,
    # we now vary the number of centres and evaluate the performance
    reg_param = reg_params[best_j]
    scale = scales[best_i]

    train_errors = []
    test_errors = []

    n_centres_seq = np.linspace(start=0.01, stop=1, num=50)

    # for the centres of the basis functions sample 10% of the data
    for centre_percentage in n_centres_seq:
        sample_fraction = centre_percentage
        p = (1 - sample_fraction, sample_fraction)
        # constructing the feature mapping anew for each number of centres
        centres = inputs[np.random.choice([False, True], size=n, p=p), :]

        feature_mapping = construct_rbf_feature_mapping(centres, scale)
        design_matrix = feature_mapping(inputs)

        train_design_matrix, train_targets, test_design_matrix, test_targets = \
            train_and_test_partition(
                design_matrix, targets, train_part, test_part)

        # evaluating the test and train error for the given regularisation parameter and scale
        train_error, test_error, residual_variance = train_and_test(
            train_design_matrix, train_targets, test_design_matrix, test_targets, reg_param)

        # collecting the errors
        train_errors.append(train_error)
        test_errors.append(test_error)

    # now we need to find the minimum test error and the equivalent number of centres (% of input points used)
    minimum_error_centres = test_errors[test_errors != 0].min()
    k_minimum = np.where(test_errors == minimum_error_centres)
    # extracting
    k_minimum = tuple([k.item() for k in k_minimum])
    best_k = k_minimum[0]

    # plotting the results
    fig, ax = plot_train_test_errors(
        "(% of inputs as centres) * 100", n_centres_seq, train_errors, test_errors, test_error_linear)

    ax.set_title('Train vs Test Error across Centre Number (as % of Inputs)')
    fig.savefig("../plots/rbf/rbf_searching_number_centres.png", fmt="png")

    return scales[best_i], reg_params[best_j], n_centres_seq[best_k]
