import numpy as np
import numpy.linalg as linear_alg
import matplotlib.pyplot as plt

# for performing regression
from regression_models import expand_to_monomials
from regression_models import construct_polynomial_approx

# for plotting results
from regression_plot import plot_function_data_and_approximation


def least_squares_weights(processed_inputs, targets):
    """
    This method returns the weights that give the best linear fit between
    the processed inputs and the targets.
    """

    phi = np.matrix(processed_inputs)
    targets = np.matrix(targets).reshape((len(targets), 1))
    # targets = np.array(targets).reshape((len(targets), 1))
    weights = linear_alg.inv(phi.transpose()*phi)*phi.transpose()*targets

    return np.array(weights).flatten()


def linear_regression(header, inputs, targets):
    """
   This function contains code that demonstrates how to use the
   functions defined in poly_fit_base for fitting polynomial curves to data.
   In this case we are using a degree of 1, making it a linear regression.
   We are also regressing quality (dependent variable) on each input/predictor one-by-one and plotting them.
   """

    # choosing degree: linear
    degree = 1

    for i in range(inputs.shape[1]):
        variable = inputs[:, i]

        # converting our inputs (we just imported) into a matrix where each row
        # is a vector of monomials of the corresponding input
        processed_inputs = expand_to_monomials(variable, degree)

        # finding the weights that fit the data in a least squares way
        weights = least_squares_weights(processed_inputs, targets)

        # using weights to create a function that takes inputs and returns predictions
        linear_approx = construct_polynomial_approx(degree, weights)

        fig, ax, hs = plot_function_data_and_approximation(
            linear_approx, variable, targets)

        ax.set_xlabel(header[i])
        ax.set_ylabel('quality')
        ax.set_title('Regressing Quality on ' + header[i].title())
        ax.set_xlim([min(variable), max(variable)])
        ax.set_ylim([0, 10])

        fig.tight_layout()
        fig.savefig("../plots/line_best_fit/regression_linear_" + header[i] + "_quality.png", fmt="png")


def scatter_plots_with_best_fit_lines(inputs, targets):
    # create an empty figure object
    fig = plt.figure()

    # create a single axis on that figure
    ax = fig.add_subplot(2, 2, 1)
    alcohol = inputs[:, 10]
    quality = targets
    ax.plot(alcohol, quality, 'o', markersize=2)
    ax.plot(alcohol, np.poly1d(np.polyfit(alcohol, quality, 1))(alcohol), color='r', linewidth=2.0)
    ax.set_xlabel("Alcohol content")
    ax.set_ylabel("Quality of wine")
    ax.set_title('Alcohol content vs quality of wine', fontsize=10)

    ax = fig.add_subplot(2, 2, 2)
    sulphates = inputs[:, 9]
    ax.plot(sulphates, quality, 'o', markersize=2)
    ax.plot(sulphates, np.poly1d(np.polyfit(sulphates, quality, 1))(sulphates), color='r', linewidth=2.0)
    ax.set_xlabel("Sulphates")
    ax.set_ylabel("Quality")
    ax.set_title('Sulphates vs quality of wine', fontsize=10)

    ax = fig.add_subplot(2, 2, 3)
    residual_sugar = inputs[:, 3]
    ax.plot(residual_sugar, quality, 'o', markersize=2)
    ax.plot(residual_sugar, np.poly1d(np.polyfit(residual_sugar, quality, 1))(residual_sugar), color='r', linewidth=2.0)
    ax.set_xlabel("Residual Sugar")
    ax.set_ylabel("Quality")
    ax.set_title('Residual sugar vs quality of wine', fontsize=10)
    fig.tight_layout()
    fig.savefig("../plots/exploratory/best_fit_lines.png", fmt="png")


def main(header, inputs, targets, has_header=True, test_fraction=0.20):
    # setting a seed to get the same pseudo-random results every time
    np.random.seed(30)

    # linear_regression(header, inputs, targets)

    scatter_plots_with_best_fit_lines(inputs, targets)

    plt.show()
