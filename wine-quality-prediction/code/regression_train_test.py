import numpy as np
import math

# for fitting
from regression_models import ml_weights
from regression_models import regularised_ml_weights
from regression_models import linear_model_predict


def simple_evaluation_linear_model(
        inputs, targets, test_fraction=None, reg_param=None):
    """
    Will split inputs and targets into train and test parts, then fit a linear
    model to the training part, and test on both parts.

    Inputs can be a data matrix (or design matrix), targets should
    be real valued.

    parameters
    ----------
    inputs - the input design matrix (any feature mapping should already be
        applied)
    targets - the targets as a vector
    reg_param (optional) - the regularisation strength. If provided, then
        regularised least squares fitting is uses with this regularisation
        strength. Otherwise, (non-regularised) least squares is used.

    returns
    -------
    train_error - the training error for the approximation
    test_error - the test error for the approximation
    """

    # getting the train/test split
    train_part, test_part = train_and_test_split(
        inputs.shape[0], test_fraction=test_fraction)

    # breaking the data into train and test parts
    train_inputs, train_targets, test_inputs, test_targets = \
        train_and_test_partition(inputs, targets, train_part, test_part)

    # now training and evaluating the error on both sets
    train_error, test_error, residual_variance = train_and_test(
        train_inputs, train_targets, test_inputs, test_targets,
        reg_param=reg_param)

    return train_error, test_error, residual_variance


def train_and_test(
        train_inputs, train_targets, test_inputs, test_targets, reg_param=None):
    """
    Will fit a linear model with either least squares, or regularised least 
    squares to the training data, then evaluate on both test and training data

    parameters
    ----------
    train_inputs - the input design matrix for training
    train_targets - the training targets as a vector
    test_inputs - the input design matrix for testing
    test_targets - the test targets as a vector
    reg_param (optional) - the regularisation strength. If provided, then
        regularised maximum likelihood fitting is uses with this regularisation
        strength. Otherwise, (non-regularised) least squares is used.

    returns
    -------
    train_error - the training error for the approximation
    test_error - the test error for the approximation
    """

    # finding the optimal weights (depends on regularisation)
    if reg_param is None:
        # using simple least squares approach
        weights = ml_weights(
            train_inputs, train_targets)
    else:
        # using regularised least squares approach
        weights = regularised_ml_weights(
          train_inputs, train_targets,  reg_param)

    # predictions are linear functions of the inputs, we evaluate those here
    train_predicts = linear_model_predict(train_inputs, weights)
    test_predicts = linear_model_predict(test_inputs, weights)

    residuals = (np.array(train_targets).flatten() - np.array(train_predicts).flatten())
    variance_training_error = np.var(residuals)
    sum_joint_log_probabilities = 0
    for n in range(len(test_predicts)):
        if test_predicts[n] <= 0:
            continue
        else:
            sum_joint_log_probabilities += math.log(test_predicts[n])

    sum_joint_log_probabilities *= -1
    # print("Error as negative joint log probability: %r" % sum_joint_log_probabilities)

    # evaluate the error between the predictions and true targets on both sets
    train_error = root_mean_squared_error(train_targets, train_predicts)
    test_error = root_mean_squared_error(test_targets, test_predicts)

    if np.isnan(test_error):
        print("test_predicts = %r" % (test_predicts,))

    return train_error, test_error, variance_training_error


def train_and_test_split(n, test_fraction=None):
    """
    Randomly generates a train/test split for data of size N.

    parameters
    ----------
    n - the data set size
    test_fraction - a fraction (between 0 and 1) specifying the proportion of
        the data to use as test data.

    returns
    -------
    train_part - a boolean vector of length N, where the ith element is
        True if the ith data-point belongs to the training set, and False if
        otherwise.
    test_part - a boolean vector of length N, where the ith element is
        True if the ith data-point belongs to the testing set, and False if
        otherwise
    """

    if test_fraction is None:
        test_fraction = 0.2
    p = [test_fraction, (1-test_fraction)]
    train_part = np.random.choice([False, True], size=n, p=p)
    test_part = np.invert(train_part)

    return train_part, test_part


def train_and_test_partition(inputs, targets, train_part, test_part):
    """
    Splits a data matrix (or design matrix) and associated targets into train
    and test parts.

    parameters
    ----------
    inputs - a 2d numpy array whose rows are the data points, or can be a design
        matrix, where rows are the feature vectors for data points.
    targets - a 1d numpy array whose elements are the targets.
    train_part - A list (or 1d array) of N booleans, where N is the number of
        data points. If the ith element is true, then the ith data point will be
        added to the training data.
    test_part - (like train_part), but specifying the test points.

    returns
    -------     
    train_inputs - the training input matrix
    train_targets - the training targets
    test_inputs - the test input matrix
    test_targets - the test targets
    """

    # getting the indices of the train and test portion
    if len(inputs.shape) == 1:
        # if inputs is a sequence of scalars, we should reshape into a matrix
        inputs = inputs.reshape((inputs.size, 1))

    # getting the indices of the train and test portion
    train_inputs = inputs[train_part, :]
    test_inputs = inputs[test_part, :]
    train_targets = targets[train_part]
    test_targets = targets[test_part]

    return train_inputs, train_targets, test_inputs, test_targets


def root_mean_squared_error(y_true, y_predicted):
    """
    Evaluate how closely predicted values (y_predicted) match the true values
    (y_true, also known as targets)

    Parameters
    ----------
    y_true - the true targets
    y_predicted - the predicted targets

    Returns
    -------
    mse - The root mean squared error between true and predicted targets
    """

    n = len(y_true)

    # be careful, square must be done element-wise
    # (hence conversion to np.array)
    mse = np.sum((np.array(y_true).flatten() - np.array(y_predicted).flatten())**2)/n

    return np.sqrt(mse)


def create_cv_folds(N, num_folds):
    """
    Defines the cross-validation splits for N data-points into num_folds folds.
    Returns a list of folds, where each fold is a train-test split of the data.
    Achieves this by partitioning the data into num_folds (almost) equal
    subsets, where in the ith fold, the ith subset will be assigned to testing,
    with the remaining subsets assigned to training.

    parameters
    ----------
    N - the number of data points
    num_folds - the number of folds

    returns
    -------
    folds - a sequence of num_folds folds, each fold is a train and test array
        indicating (with a boolean array) whether a datapoint belongs to the
        training or testing part of the fold.
        Each fold is a (train_part, test_part) pair where:

        train_part - a boolean vector of length N, where if ith element is
            True if the ith data-point belongs to the training set, and False if
            otherwise.
        test_part - a boolean vector of length N, where if ith element is
            True if the ith data-point belongs to the testing set, and False if
            otherwise.
    """

    # if the number of data points is not divisible by folds then some parts
    # will be larger than others (by 1 data-point). min_part is the smallest
    # size of a part (uses integer division operator //)
    min_part = N//num_folds
    # rem is the number of parts that will be 1 larger
    rem = N % num_folds
    # create an empty array which will specify which part a data point belongs to
    parts = np.empty(N, dtype=int)
    start = 0
    for part_id in range(num_folds):
        # calculate size of the part
        n_part = min_part
        if part_id < rem:
            n_part += 1
        # now assign the part id to a block of the parts array
        parts[start:start+n_part] = part_id*np.ones(n_part)
        start += n_part
    # now randomly reorder the parts array (so that each data point is assigned
    # a random part.
    np.random.shuffle(parts)
    # we now want to turn the parts array, into a sequence of train-test folds
    folds = []
    for f in range(num_folds):
        train = (parts != f)
        test = (parts == f)
        folds.append((train, test))

    return folds


def cv_evaluation_linear_model(inputs, targets, folds, reg_param=None):
    """
    Will split inputs and targets into train and test parts, then fit a linear
    model to the training part, and test on the both parts.

    Inputs can be a data matrix (or design matrix), targets should
    be real valued.

    parameters
    ----------
    inputs - the input design matrix (any feature mapping should already be
        applied)
    targets - the targets as a vector
    num_folds - the number of folds
    reg_param (optional) - the regularisation strength. If provided, then
        regularised least squares fitting is uses with this regularisation
        strength. Otherwise, (non-regularised) least squares is used.

    returns
    -------
    train_errors - the training errors for the approximation
    test_errors - the test errors for the approximation
    """

    # getting the number of folds
    num_folds = len(folds)
    train_errors = np.empty(num_folds)
    test_errors = np.empty(num_folds)
    for f, fold in enumerate(folds):
        # f is the fold id, fold is the train-test split
        train_part, test_part = fold
        # break the data into train and test sets
        train_inputs, train_targets, test_inputs, test_targets = \
            train_and_test_partition(inputs, targets, train_part, test_part)
        # now train and evaluate the error on both sets
        train_error, test_error, residual_variance = train_and_test(
            train_inputs, train_targets, test_inputs, test_targets,
            reg_param=reg_param)
        # print("train_error = %r" % (train_error,))
        # print("test_error = %r" % (test_error,))
        train_errors[f] = train_error
        test_errors[f] = test_error

    return train_errors, test_errors
