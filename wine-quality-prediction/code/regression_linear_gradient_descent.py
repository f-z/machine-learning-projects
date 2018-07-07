import numpy as np
import matplotlib.pyplot as plt
from import_explore import standardise
from regression_train_test import train_and_test_partition
from regression_train_test import train_and_test
from regression_train_test import simple_evaluation_linear_model
from regression_train_test import cv_evaluation_linear_model
from regression_train_test import create_cv_folds


"""
Gradient Descent method for simple linear regression.
This gradient_descent method was adapted from:
"http://nbviewer.jupyter.org/github/jdwittenauer/ipython-notebooks/blob/master/notebooks/ml/ML-Exercise1.ipynb".
"""


def main(header, inputs, targets, test_fraction=0.25):
    # setting a seed to get the same pseudo-random results every time
    np.random.seed(30)

    raw_inputs = inputs
    # normalise inputs (to investigate if it has an effect on the linear regression)
    # and defining them for gradient descent method
    grad_desc_inputs = standardise(inputs)
    ones = np.ones((inputs.shape[0], 1))
    grad_desc_inputs = np.hstack((ones, grad_desc_inputs))

    # defining target for gradient descent method
    grad_desc_target = targets

    # converting to matrices and initialising theta
    grad_desc_inputs = np.matrix(grad_desc_inputs)
    grad_desc_target = (np.matrix(targets)).T
    # theta2 = np.zeros((12,1)) - this generates a column - we don't want this here
    theta2 = np.zeros((1, inputs.shape[1] + 1))
    theta2 = np.matrix(theta2)

    # defining alpha, the learning rate and the number of iterations for
    # the gradient descent method
    alpha = 0.01
    iterations = 1000

    # performing linear regression on the data set
    g2, cost2 = gradient_descent(grad_desc_inputs, grad_desc_target, theta2, alpha, iterations)

    # get the cost (error) of the model
    compute_cost(grad_desc_inputs, grad_desc_target, g2)

    # plotting the error function against the number of iteration to check if gradient
    # descent is working or not
    # If gradient descent is working correctly the error function should decrease
    # after every iteration until it reaches the local minimum
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(np.arange(iterations), cost2, 'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Error')
    ax.set_title('Error vs Iterations')
    fig.savefig("../plots/simple_linear/cost_vs_iterations_plot.png", fmt="png")

    # Predict function for the gradient descent method
    quality = (grad_desc_inputs * g2.T)

    # calculating the test error for the gradient descent method
    errors = np.sum((np.array(quality) - np.array(grad_desc_target)) ** 2) / len(grad_desc_target)
    test_error_grad = np.sqrt(errors)
    print("Test error obtained from the gradient descent method: {test_error_grad}".format(**locals()))

    """
    Implementing the Linear regression method using the least squares approach.
    """
    # adding a columns of ones to the input matrix
    N, D = inputs.shape
    column = np.ones((N, 1))
    inputs = np.hstack((column, inputs))

    # performing linear regression with normalised inputs
    print("Printing linear regression with normalised inputs:")
    fig, ax, train_error, test_error = evaluate_linear_approx(
        inputs, targets, test_fraction)
    fig.suptitle("Plot of Train and Test Errors with \n Normalised Inputs Against Different Reg. Parameters")
    # plt.ylim(0.5, 0.75)

    # performing linear regression with raw data
    print("Printing linear regression with raw data:")
    fig2, ax2, train_error2, test_error2 = evaluate_linear_approx(
        raw_inputs, targets, test_fraction)
    fig2.suptitle("Plot of Train and Test Errors with \n Raw Inputs Against Different Reg. Parameters")
    # plt.ylim(0.5, 0.75)

    # cross validation
    num_folds = 5
    folds = create_cv_folds(N, num_folds)

    fig2, ax = evaluate_reg_param(raw_inputs, targets, folds, reg_params=None)
    fig2.suptitle(
        "Cross-validation of Train and Test Errors \n with Raw Inputs Against Different Reg. Parameters")
    # plt.ylim(0.6, 0.68)

    plt.show()


def linear_model_predict(designmtx, weights):
    ys = np.matrix(designmtx) * np.matrix(weights).reshape((len(weights), 1))
    return np.array(ys).flatten()


def train_and_test_split(N, test_fraction=None):
    """
    Randomly generates a train/test split for data of size N.

    parameters
    ----------
    N - the dataset size
    test_fraction - a fraction (between 0 and 1) specifying the proportion of
        the data to use as test data.
    """
    if test_fraction is None:
        test_fraction = 0.5
    p = [test_fraction, (1 - test_fraction)]
    train_part = np.random.choice([False, True], size=N, p=p)
    test_part = np.invert(train_part)
    return train_part, test_part


def plot_train_test_errors(
        control_var, experiment_sequence, train_errors, test_errors):
    """
    Plot the train and test errors for a sequence of experiments.

    parameters
    ----------
    control_var - the name of the control variable, e.g. degree (for polynomial)
        degree.
    experiment_sequence - a list of values applied to the control variable.
    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    train_line, = ax.plot(experiment_sequence, train_errors, 'b-')
    test_line, = ax.plot(experiment_sequence, test_errors, 'r-')
    ax.set_xlabel(control_var)
    ax.set_ylabel("$E_{RMS}$")

    ax.legend([train_line, test_line], ["train", "test"])
    return fig, ax


def compute_cost(X, y, theta):
    """
    This methods  compute the cost of a given solution (characterized by
    the parameters theta)
    """
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))


def gradient_descent(X, y, theta, alpha, iters):
    """
    This method performs the gradient descent method.
    """

    temp = np.matrix(np.zeros(theta.shape))

    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)

    for i in range(iters):
        error = (X * theta.T) - y

        for j in range(parameters):
            term = np.multiply(error, X[:, j])
            temp[0, j] = theta[0, j] - ((alpha / len(X)) * np.sum(term))

        theta = temp
        cost[i] = compute_cost(X, y, theta)

    return theta, cost


def evaluate_reg_param(inputs, targets, folds, reg_params=None):
    """
      Evaluate, then plot the performance of different regularisation parameters.
    """

    # create the feature mapping and then the design matrix
    # choose a range of regularisation parameters
    if reg_params is None:
        reg_params = np.logspace(-15, -4, 11)
    num_values = reg_params.size
    num_folds = len(folds)
    # create some arrays to store results
    train_mean_errors = np.zeros(num_values)
    test_mean_errors = np.zeros(num_values)
    train_stdev_errors = np.zeros(num_values)
    test_stdev_errors = np.zeros(num_values)

    for r, reg_param in enumerate(reg_params):
        # r is the index of reg_param, reg_param is the regularisation parameter
        # cross validate with this regularisation parameter
        train_errors, test_errors = cv_evaluation_linear_model(
            inputs, targets, folds, reg_param=reg_param)

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

    print("Printing cross-validation mean train error:")
    print(train_mean_error)
    print("Printing cross-validation mean test error:")
    print(test_mean_error)

    # now plotting the results
    fig, ax = plot_train_test_errors(
        "$\lambda$", reg_params, train_mean_errors, test_mean_errors)
    # Here we plot the error ranges too: mean plus/minus 1 standard error.
    # 1 standard error is the standard deviation divided by sqrt(n) where
    # n is the number of samples.
    # (There are other choices for error bars.)
    # train error bars
    lower = train_mean_errors - train_stdev_errors / np.sqrt(num_folds)
    upper = train_mean_errors + train_stdev_errors / np.sqrt(num_folds)
    ax.fill_between(reg_params, lower, upper, alpha=0.2, color='b')
    # test error bars
    lower = test_mean_errors - test_stdev_errors / np.sqrt(num_folds)
    upper = test_mean_errors + test_stdev_errors / np.sqrt(num_folds)
    ax.fill_between(reg_params, lower, upper, alpha=0.2, color='r')
    ax.set_xscale('log')
    return fig, ax


def evaluate_linear_approx(data_input, targets, test_fraction):
    # the linear performance
    train_error, test_error, residual_variance = simple_evaluation_linear_model(
        data_input, targets, test_fraction=test_fraction)
    print("Linear Regression without Regularisation:")
    print("\t(train_error, test_error) = %r" % ((train_error, test_error),))

    reg_params = np.logspace(-15, -4, 11)
    train_errors = []
    test_errors = []
    for reg_param in reg_params:
        train_error, test_error, residual_variance = simple_evaluation_linear_model(
            data_input, targets, test_fraction=test_fraction, reg_param=reg_param)
        train_errors.append(train_error)
        test_errors.append(test_error)

    fig, ax = plot_train_test_errors(
        "$\lambda$", reg_params, train_errors, test_errors)

    ax.set_xscale('log')

    return fig, ax, train_error, test_error
