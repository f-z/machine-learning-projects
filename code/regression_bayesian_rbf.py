import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
import math

from import_explore import standardise

# for performing regression
from regression_models import construct_rbf_feature_mapping
from regression_models import construct_feature_mapping_approx
from regression_models import calculate_weights_posterior
from regression_models import predictive_distribution

# for plotting results
from regression_plot import plot_function_and_data

# for splitting our data into train and test parts, partitioning, and calculating root mean squared error
from regression_train_test import train_and_test_split
from regression_train_test import train_and_test_partition
from regression_train_test import root_mean_squared_error


def main(inputs, targets, scale, best_no_centres, test_fraction=0.20):
    # setting a seed to get the same pseudo-random results every time
    np.random.seed(30)

    print("\n")

    std_inputs = standardise(inputs)

    train_part, test_part = train_and_test_split(std_inputs.shape[0], test_fraction)

    train_inputs, train_targets, test_inputs, test_targets = train_and_test_partition(
        std_inputs, targets, train_part, test_part)

    # specifying the centres of the rbf basis functions
    # choosing 10% of the data for the centres of the basis functions or the optimal proportion from earlier analyses
    centres = train_inputs[np.random.choice(
        [False, True], size=train_inputs.shape[0], p=[1-best_no_centres, best_no_centres]), :]
    print("centres shape = %r" % (centres.shape,))

    # the width (analogous to standard deviation) of the basis functions
    # scale of the basis functions from analysis in external_data file
    # We consider the basis function widths to be fixed for simplicity
    print("scale = %r" % scale)

    # creating the feature mapping
    feature_mapping = construct_rbf_feature_mapping(centres, scale)

    # plotting the basis functions themselves for reference
    display_basis_functions(feature_mapping, train_inputs.shape[1])

    # alpha and beta define the shape of our curve when we start

    # beta is defining the noise precision of our data, as the reciprocal of the target variance
    # it is the spread from the highest point (top) of the curve
    # it corresponds to additive Gaussian noise of variance, which is beta to the power of -1
    beta = np.reciprocal(0.40365849982557295)
    # beta = np.reciprocal(np.var(train_targets))
    # beta = 100
    # higher beta is going to give us higher precision, so less overlap
    # as a side note, could also do beta = 1 / np.var(train_targets)

    # location of the highest point of the initial curve / prior distribution
    # because targets represent quality ranging from 0 to 10
    alpha = mode(targets)[0][0]
    # alpha = 100

    # now applying our feature mapping to the train inputs and constructing the design matrix
    design_matrix = feature_mapping(train_inputs)
    # the number of features (phis) is the width of this matrix
    # it is equal to the number of centres drawn from the train inputs
    # the shape[0] is the number of data points I use for training
    M = design_matrix.shape[1]

    # defining a prior mean and covariance matrix
    # they represent our prior belief over the distribution
    # our initial estimate of the range of probabilities
    m0 = np.zeros(M)

    for m in range(len(m0)):
        m0[m] = mode(targets)[0][0]  # setting to be the mode of targets
        # m0[m] = 0

    S0 = alpha * np.identity(M)

    # diagonal regularisation matrix A to punish over-fitting
    # A = alpha * np.identity(M)
    # E = 0.5 * m0.T * A * m0
    # Zp = regularisation constant
    # prior_m0 = np.exp(-E)/Zp

    # finding the posterior over weights
    # if we have enough data, the posteriors will be the same, no matter the initial parameters
    # because they will have been updated according to Bayes' rule
    mN, SN = calculate_weights_posterior(design_matrix, train_targets, beta, m0, S0)
    # print("mN = %r" % (mN,))

    # the posterior mean (also the MAP) gives the central prediction
    mean_approx = construct_feature_mapping_approx(feature_mapping, mN)

    # getting MAP and calculating root mean squared errors
    train_output = mean_approx(train_inputs)
    test_output = mean_approx(test_inputs)

    bayesian_mean_train_error = root_mean_squared_error(train_targets, train_output)
    bayesian_mean_test_error = root_mean_squared_error(test_targets, test_output)
    print("Root mean squared errors:")
    print("Train error of posterior mean (applying Bayesian inference): %r" % bayesian_mean_train_error)
    print("Test error of posterior mean (applying Bayesian inference): %r" % bayesian_mean_test_error)

    # plotting one input variable on the x axis as an example
    fig, ax, lines = plot_function_and_data(std_inputs[:, 10], targets)

    # creating data to use for plotting
    xs = np.ndarray((101, train_inputs.shape[1]))

    for column in range(train_inputs.shape[1]):
        column_sample = np.linspace(-5, 5, 101)
        column_sample = column_sample.reshape((column_sample.shape[0],))
        xs[:, column] = column_sample

    ys = mean_approx(xs)
    line, = ax.plot(xs[:, 10], ys, 'r-')
    lines.append(line)
    ax.set_ylim([0, 10])

    # now plotting a number of samples from the posterior
    for i in range(20):
        weights_sample = np.random.multivariate_normal(mN, SN)
        sample_approx = construct_feature_mapping_approx(
            feature_mapping, weights_sample)
        sample_ys = sample_approx(xs)
        line, = ax.plot(xs[:, 10], sample_ys, 'm', linewidth=0.5)
    lines.append(line)
    ax.legend(lines, ['data', 'mean approx', 'samples'])

    # now for the predictive distribution
    new_designmtx = feature_mapping(xs)
    ys, sigma2Ns = predictive_distribution(new_designmtx, beta, mN, SN)
    print("(sigma2Ns**0.5).shape = %r" % ((sigma2Ns**0.5).shape,))
    print("np.sqrt(sigma2Ns).shape = %r" % (np.sqrt(sigma2Ns).shape,))
    print("ys.shape = %r" % (ys.shape,))

    ax.plot(xs[:, 10], ys, 'r', linewidth=3)
    lower = ys-np.sqrt(sigma2Ns)
    upper = ys+np.sqrt(sigma2Ns)
    print("lower.shape = %r" % (lower.shape,))
    print("upper.shape = %r" % (upper.shape,))
    ax.fill_between(xs[:, 10], lower, upper, alpha=0.2, color='r')
    ax.set_title('Posterior Mean, Samples, and Predictive Distribution')
    ax.set_xlabel('standardised alcohol content')
    ax.set_ylabel('p(t|x)')
    fig.tight_layout()
    fig.savefig("../plots/bayesian/bayesian_rbf.png", fmt="png")

    plt.show()

    # the predictive distribution
    test_design_matrix = feature_mapping(test_inputs)
    predictions, prediction_sigma2 = predictive_distribution(test_design_matrix, beta, mN, SN)
    sum_joint_log_probabilities = 0
    for n in range(len(predictions)):
        sum_joint_log_probabilities += math.log(predictions[n])

    sum_joint_log_probabilities *= -1
    # joint_log_probabilities = (np.array(test_targets).flatten() - np.array(predictions).flatten())
    # print(np.mean(joint_log_probabilities))
    print("Error as negative joint log probability: %r" % sum_joint_log_probabilities)


def display_basis_functions(feature_mapping, num_columns):
    data_matrix = np.ndarray((101, num_columns))

    for column in range(num_columns):
        column_sample = np.linspace(-5, 5, 101)
        column_sample = column_sample.reshape((column_sample.shape[0],))
        data_matrix[:, column] = column_sample

    design_matrix = feature_mapping(data_matrix)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for column_number in range(design_matrix.shape[1]):
        ax.plot(data_matrix[:, 10], design_matrix[:, column_number])
    ax.set_xlim([-5, 5])
    ax.set_title('Radial Basis Functions: Bayesian Approach')
    ax.set_xlabel('standardised alcohol content')
    ax.set_ylabel('$\Phi$(x)')
    fig.savefig('../plots/bayesian/bayesian_radial_basis_functions.png', fmt='png')


def create_multidimensional_sample_data(inputs):
    # now plotting a number of samples from the posterior
    """
    Creates a (input.shape) n-dimensional array of random variables, where each column
    is a vector of values independently sampled from the Gaussian distribution of the equivalent original input column.
    """

    sampled_data = np.ndarray(inputs.shape)
    num_points = inputs.shape[0]

    for column in range(inputs.shape[1]):
        column_sample = np.random.normal(np.mean(inputs[:, column]), np.std(inputs[:, column]), (num_points, 1))
        column_sample = column_sample.reshape((column_sample.shape[0],))
        sampled_data[:, column] = column_sample

    # print(sampled_data)
    return sampled_data


if __name__ == '__main__':
    import sys
    # this allows you to pass the file name as the first argument when you call
    # your script from the command line
    if len(sys.argv) == 1:
        # reverting to default parameters (red wine, ; delimiter, all features, scale from previous analyses)
        main('../winequality-red.csv', ";", np.arange(0, 11), 6.7)
    elif len(sys.argv) == 2:
        # passing the file name as the first argument
        main(sys.argv[1], ";", np.arange(0, 11), 6.7)
    elif len(sys.argv) == 3:
        # passing the delimiter as the second argument
        main(sys.argv[1], sys.argv[2], np.arange(0, 11), 6.7)
    elif len(sys.argv) == 4:
        # the third argument is a list of columns to use as input features
        # list is separated by ','
        custom_columns = list(map(int, sys.argv[3].split(",")))
        main(sys.argv[1], sys.argv[2], custom_columns, 6.7)
