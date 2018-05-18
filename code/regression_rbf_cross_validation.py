import numpy as np
import matplotlib.pyplot as plt

from import_explore import standardise

# for performing regression
from regression_models import construct_rbf_feature_mapping
# for plotting results
from regression_plot import plot_train_test_errors
# two new functions for cross validation
from regression_train_test import create_cv_folds
from regression_train_test import cv_evaluation_linear_model


def evaluate_reg_param(inputs, targets, folds, centres, scale, test_error_linear, reg_params=None):
    """
      Evaluate, then plot the performance of different regularisation parameters.
    """

    # creating the feature mapping and then the design matrix
    feature_mapping = construct_rbf_feature_mapping(centres, scale)
    design_matrix = feature_mapping(inputs)

    # choose a range of regularisation parameters
    if reg_params is None:
        reg_params = np.logspace(-15, 5, 30)  # choices of regularisation strength

    num_values = reg_params.size
    num_folds = len(folds)
    # create some arrays to store results
    train_mean_errors = np.zeros(num_values)
    test_mean_errors = np.zeros(num_values)
    train_st_dev_errors = np.zeros(num_values)
    test_st_dev_errors = np.zeros(num_values)

    print('Calculating means and standard deviations of train and test errors...')
    for r, reg_param in enumerate(reg_params):
        # r is the index of reg_param, reg_param is the regularisation parameter
        # cross validate with this regularisation parameter
        train_errors, test_errors = cv_evaluation_linear_model(
            design_matrix, targets, folds, reg_param=reg_param)
        # we're interested in the average (mean) training and testing errors
        train_mean_error = np.mean(train_errors)
        test_mean_error = np.mean(test_errors)
        train_st_dev_error = np.std(train_errors)
        test_st_dev_error = np.std(test_errors)
        # storing the results
        train_mean_errors[r] = train_mean_error
        test_mean_errors[r] = test_mean_error
        train_st_dev_errors[r] = train_st_dev_error
        test_st_dev_errors[r] = test_st_dev_error

    # plotting the results
    fig, ax = plot_train_test_errors(
        "$\lambda$", reg_params, train_mean_errors, test_mean_errors, test_error_linear)

    # Here we plot the error ranges too: mean plus/minus 1 standard error.
    # 1 standard error is the standard deviation divided by sqrt(n) where
    # n is the number of samples. 
    # (There are other choices for error bars.)
    # train error bars
    lower = train_mean_errors - train_st_dev_errors/np.sqrt(num_folds)
    upper = train_mean_errors + train_st_dev_errors/np.sqrt(num_folds)
    ax.fill_between(reg_params, lower, upper, alpha=0.2, color='b')
    # test error bars
    lower = test_mean_errors - test_st_dev_errors/np.sqrt(num_folds)
    upper = test_mean_errors + test_st_dev_errors/np.sqrt(num_folds)
    ax.fill_between(reg_params, lower, upper, alpha=0.2, color='r')

    ax.set_xscale('log')
    ax.set_ylim([0, 1])

    ax.set_title('Train vs Test Error across Reg. Param. with Cross-validation')
    fig.savefig("../plots/rbf/rbf_searching_reg_params_cross_validation.png", fmt="png")

    plt.show()


def evaluate_scale(inputs, targets, folds, centres, reg_param, test_error_linear, scales=None):
    """
    Evaluate, then plot the performance of different basis function scales.
    """

    # choosing a range of scales
    if scales is None:
        scales = np.logspace(0, 8, 30)  # of the basis functions
    #
    num_values = scales.size
    num_folds = len(folds)

    # creating some arrays to store results
    train_mean_errors = np.zeros(num_values)
    test_mean_errors = np.zeros(num_values)
    train_st_dev_errors = np.zeros(num_values)
    test_st_dev_errors = np.zeros(num_values)

    for s, scale in enumerate(scales):
        feature_mapping = construct_rbf_feature_mapping(centres,scale)
        design_matrix = feature_mapping(inputs)
        # r is the index of reg_param, reg_param is the regularisation parameter
        # cross validate with this regularisation parameter
        train_errors, test_errors = cv_evaluation_linear_model(
            design_matrix, targets, folds, reg_param=reg_param)
        # we're interested in the average (mean) training and testing errors
        train_mean_error = np.mean(train_errors)
        test_mean_error = np.mean(test_errors)
        train_st_dev_error = np.std(train_errors)
        test_st_dev_error = np.std(test_errors)
        # store the results
        train_mean_errors[s] = train_mean_error
        test_mean_errors[s] = test_mean_error
        train_st_dev_errors[s] = train_st_dev_error
        test_st_dev_errors[s] = test_st_dev_error

    # Now plot the results
    fig, ax = plot_train_test_errors(
        "scale", scales, train_mean_errors, test_mean_errors, test_error_linear)
    # Here we plot the error ranges too: mean plus/minus 1 standard error.
    # 1 standard error is the standard deviation divided by sqrt(n) where
    # n is the number of samples. 
    # (There are other choices for error bars.)
    # train error bars
    lower = train_mean_errors - train_st_dev_errors/np.sqrt(num_folds)
    upper = train_mean_errors + train_st_dev_errors/np.sqrt(num_folds)
    ax.fill_between(scales, lower, upper, alpha=0.2, color='b')
    # test error bars
    lower = test_mean_errors - test_st_dev_errors/np.sqrt(num_folds)
    upper = test_mean_errors + test_st_dev_errors/np.sqrt(num_folds)
    ax.fill_between(scales, lower, upper, alpha=0.2, color='r')
    ax.set_xscale('log')
    ax.set_ylim([0, 1])

    ax.set_title('Train vs Test Error across Scales with Cross-validation')
    fig.savefig("../plots/rbf/rbf_searching_scales_cross_validation.png", fmt="png")

    plt.show()


def evaluate_num_centres(
        inputs, targets, folds, scale, reg_param, test_error_linear, num_centres_sequence=None):
    """
      Evaluate, then plot the performance of different numbers of basis
      function centres.
    """

    # choosing a range of numbers of centres
    if num_centres_sequence is None:
        num_centres_sequence = np.linspace(start=0.01, stop=1, num=20)  # tested with 50, using 20 to speed things up

    num_values = num_centres_sequence.size
    num_folds = len(folds)

    # creating some arrays to store results
    train_mean_errors = np.zeros(num_values)
    test_mean_errors = np.zeros(num_values)
    train_st_dev_errors = np.zeros(num_values)
    test_st_dev_errors = np.zeros(num_values)

    n = inputs.shape[0]

    # running the experiments
    for c, centre_percentage in enumerate(num_centres_sequence):
        sample_fraction = centre_percentage
        p = (1 - sample_fraction, sample_fraction)
        # constructing the feature mapping anew for each number of centres
        centres = inputs[np.random.choice([False, True], size=n, p=p), :]
        # print("\ncentres.shape = %r" % (centres.shape,))
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
        train_st_dev_errors[c] = train_stdev_error
        test_st_dev_errors[c] = test_stdev_error

    # now plotting the results
    fig, ax = plot_train_test_errors(
        "% of inputs as centres * 100", num_centres_sequence, train_mean_errors, test_mean_errors, test_error_linear)
    # Here we plot the error ranges too: mean plus/minus 1 standard error.
    # 1 standard error is the standard deviation divided by sqrt(n) where
    # n is the number of samples. 
    # (There are other choices for error bars.)
    # train error bars
    lower = train_mean_errors - train_st_dev_errors/np.sqrt(num_folds)
    upper = train_mean_errors + train_st_dev_errors/np.sqrt(num_folds)
    ax.fill_between(num_centres_sequence, lower, upper, alpha=0.2, color='b')
    # test error bars
    lower = test_mean_errors - test_st_dev_errors/np.sqrt(num_folds)
    upper = test_mean_errors + test_st_dev_errors/np.sqrt(num_folds)
    ax.fill_between(num_centres_sequence, lower, upper, alpha=0.2, color='r')
    ax.set_ylim([0, 1])

    ax.set_title('Train vs Test Error across Centre Proportion with Cross-validation')
    fig.savefig("../plots/rbf/rbf_searching_number_centres_cross_validation.png", fmt="png")

    plt.show()


def main(inputs, targets, test_error_linear, best_scale=None, best_reg_param=None, best_no_centres=None):
    """
    This function contains example code that demonstrates how to use the 
    functions defined in poly_fit_base for fitting polynomial curves to data.
    """
    # setting a seed to get the same pseudo-random results every time
    np.random.seed(30)

    # defining default values in case they are not provided
    if best_scale is None:
        best_scale = 6.7
    if best_reg_param is None:
        best_reg_param = 9.2e-08

    print("\nPerforming cross-validation...")

    # getting the cross-validation folds
    num_folds = 5
    folds = create_cv_folds(inputs.shape[0], num_folds)

    # standardising for rbf - to make distances equivalent
    std_inputs = standardise(inputs)

    # specifying the centres and scale of some rbf basis functions
    centres = std_inputs[np.random.choice([False, True], size=std_inputs.shape[0], p=[1-best_no_centres, best_no_centres]), :]

    # using the estimated optimal values I found in the external_data file as starting values
    # scale = the width (analogous to standard deviation) of the basis functions
    # evaluating then plotting the performance of different reg params
    print("Evaluating reg. parameters...")
    evaluate_reg_param(std_inputs, targets, folds, centres, best_scale, test_error_linear)

    # evaluating and plotting the performance of different scales
    print("\nEvaluating scales...")
    evaluate_scale(std_inputs, targets, folds, centres, best_reg_param, test_error_linear)

    # evaluating then plotting the performance of different numbers of basis function centres
    print("\nEvaluating proportion of centres...")
    evaluate_num_centres(
        std_inputs, targets, folds, best_scale, best_reg_param, test_error_linear)
