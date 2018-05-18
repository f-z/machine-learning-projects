import numpy as np
import matplotlib.pyplot as plt

from import_explore import import_csv
from import_explore import normalise

from regression_models import construct_rbf_feature_mapping
from regression_train_test import train_and_test_split
from regression_train_test import train_and_test_partition
from regression_train_test import train_and_test
from regression_train_test import simple_evaluation_linear_model

# from regression_plot import exploratory_plots
from regression_plot import plot_train_test_errors


def main(name, delimiter, columns, has_header=True, test_fraction=0.25):
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

    # importing using csv reader and storing as numpy array
    header, data = import_csv(name, delimiter)

    print("\n")

    # exploratory_plots(data, header)

    n = data.shape[1]
    # deleting the last column (quality) from inputs
    inputs = np.delete(data, n-1, 1)
    # assigning it as targets instead
    targets = data[:, n-1]

    train_error_linear, test_error_linear = evaluate_linear_approx(
        inputs, targets, test_fraction)

    print("\n")

    inputs = normalise(inputs)

    evaluate_rbf_for_various_reg_params(
        inputs, targets, test_fraction, test_error_linear)

    parameter_search_rbf(inputs, targets, test_fraction)

    plt.show()


def evaluate_linear_approx(inputs, targets, test_fraction):
    # the linear performance
    train_error, test_error = simple_evaluation_linear_model(
        inputs, targets, test_fraction=test_fraction)

    print("Linear Regression:")
    print("\t(train_error, test_error) = %r" % ((train_error, test_error),))

    return train_error, test_error


def evaluate_rbf_for_various_reg_params(
        inputs, targets, test_fraction, test_error_linear):
    # for rbf feature mappings
    # for the centres of the basis functions choose 10% of the data
    n = inputs.shape[0]
    centres = inputs[np.random.choice([False, True], size=n, p=[0.90, 0.10]), :]
    print("centres shape = %r" % (centres.shape,))

    # the width (analogous to standard deviation) of the basis functions
    scale = 8.5  # of the basis functions
    print("centres = %r" % (centres,))
    print("scale = %r" % (scale,))

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
    reg_params = np.logspace(-15, 5, 20)
    train_errors = []
    test_errors = []

    for reg_param in reg_params:
        print("Evaluating reg. parameter " + str(reg_param))
        train_error, test_error = simple_evaluation_linear_model(
            design_matrix, targets, test_fraction=test_fraction, reg_param=reg_param)
        train_errors.append(train_error)
        test_errors.append(test_error)

    fig, ax = plot_train_test_errors(
        "$\lambda$", reg_params, train_errors, test_errors)

    # plotting a straight line showing the linear performance
    x_lim = ax.get_xlim()
    ax.plot(x_lim, test_error_linear*np.ones(2), 'g:')

    ax.set_xscale('log')
    ax.set_title('Evaluating RBF Performance')
    fig.savefig("../plots/rbf_vs_linear.pdf", fmt="pdf")


def parameter_search_rbf(inputs, targets, test_fraction):
    """
    """

    n = inputs.shape[0]
    # run all experiments on the same train-test split of the data
    train_part, test_part = train_and_test_split(n, test_fraction=test_fraction)

    # for the centres of the basis functions sample 10% of the data
    sample_fraction = 0.10
    p = (1-sample_fraction, sample_fraction)
    centres = inputs[np.random.choice([False, True], size=n, p=p), :]
    print("\ncentres.shape = %r" % (centres.shape,))

    scales = np.logspace(0, 6, 20)  # of the basis functions
    reg_params = np.logspace(-15, 5, 20)  # choices of regularisation strength
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
            train_error, test_error = train_and_test(
                train_designmtx, train_targets, test_designmtx, test_targets,
                reg_param=reg_param)
            # store the train and test errors in our 2d arrays
            train_errors[i, j] = train_error
            test_errors[i, j] = test_error

    # we have a 2d array of train and test errors, we want to know the (i,j)
    # index of the best value
    best_i = np.argmin(np.argmin(test_errors, axis=1))
    best_j = np.argmin(test_errors[i, :])
    min_place = np.argmin(test_errors)
    best_i_correct = (int)(min_place / test_errors.shape[1])
    best_j_correct = min_place % test_errors.shape[1]

    print(best_i)
    print(best_j)

    print(best_i_correct)
    print(best_j_correct)

    min = test_errors[test_errors != 0].min()
    ij_min = np.where(test_errors == min)
    ij_min = tuple([i.item() for i in ij_min])

    print(ij_min[1])

    print("\nBest joint choice of parameters:")
    print(
        "\tscale %.2g and lambda = %.2g" % (scales[best_i_correct], reg_params[best_j_correct]))

    # now we can plot the error for different scales using the best
    # regularisation choice
    fig, ax = plot_train_test_errors(
        "scale", scales, train_errors[:, best_j_correct], test_errors[:, best_j_correct])
    ax.set_xscale('log')
    ax.set_title('Train vs Test Error Across Scales')
    fig.savefig("../plots/rbf_searching_scales.pdf", fmt="pdf")

    # ...and the error for  different regularisation choices given the best
    # scale choice
    fig, ax = plot_train_test_errors(
        "$\lambda$", reg_params, train_errors[best_i_correct, :], test_errors[best_i_correct, :])
    ax.set_xscale('log')
    ax.set_title('Train vs Test Error Across Reg Params')
    fig.savefig("../plots/rbf_searching_reg_params.pdf", fmt="pdf")

    # using the best parameters found above,
    # we now vary the number of centres and evaluate the performance
    reg_param = reg_params[best_j_correct]
    scale = scales[best_i_correct]
    n_centres_seq = np.arange(1, 20)
    train_errors = []
    test_errors = []
    for n_centres in n_centres_seq:
        # constructing the feature mapping anew for each number of centres
        centres = np.linspace(0, 1, n_centres)
        feature_mapping = construct_rbf_feature_mapping(centres, scale)
        design_matrix = feature_mapping(inputs)

        train_designmtx, train_targets, test_designmtx, test_targets = \
            train_and_test_partition(
                design_matrix, targets, train_part, test_part)

        # evaluating the test and train error for the given regularisation parameter and scale
        train_error, test_error = train_and_test(
            train_designmtx, train_targets, test_designmtx, test_targets, reg_param)

        # collecting the errors
        train_errors.append(train_error)
        test_errors.append(test_error)

    # plotting the results
    fig, ax = plot_train_test_errors(
        "no. centres", n_centres_seq, train_errors, test_errors)
    ax.set_title('Train vs Test Error Across Centre Number')
    fig.savefig("../plots/rbf_searching_number_centres.pdf", fmt="pdf")


if __name__ == '__main__':
    """
    To run this script on just synthetic data use:

        python regression_external_data.py

    You can pass the data-file name as the first argument when you call
    your script from the command line. E.g. use:

        python regression_external_data.py datafile.tsv

    If you pass a second argument it will be taken as the delimiter, e.g.
    for comma separated values:

        python regression_external_data.py comma_separated_data.csv ","

    for semi-colon separated values:

        python regression_external_data.py comma_separated_data.csv ";"

    If your data has more than 2 columns you must specify which columns
    you wish to plot as a comma separated pair of values, e.g.

        python regression_external_data.py comma_separated_data.csv ";" 8,9

    For the wine quality data you will need to specify which columns to pass.
    """

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
