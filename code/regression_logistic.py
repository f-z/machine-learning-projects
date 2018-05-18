import csv
import numpy as np
from scipy.optimize import fmin_bfgs
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl

"""

Uses L2 Logistic Regression on the wine data.  L2 Regression is used to help control overfitting.
This is shown for a variety of lambdas (the regularisation parameters).

The wine data has been split into "Good" or "Bad", with 5 or below being the cutoff
for "Bad" wine.  This can easily be adjusted by modifying the cutoff specified
in the initialisation of the Wine_Data class.

The number of training/testing inputs/targets can be adjusted in __main__, as can
the input columns if the effect of lowering the number of dimensions is to be
investigated.

The code and theory was adapted from Luke Dickens' Lecture 6 material,
the Coursera course by the University of Washington
(https://www.coursera.org/learn/ml-classification/lecture/DBTNt/l2-regularized-logistic-regression)
and the optimisation and graph plotting functions were adapted from
Danny Tarlow' blog (http://blog.smellthedata.com/2009/06/python-logistic-regression-with-l2.html).

"""


def panda_logistic_information(panda_data):
    """
    Panda information on log. regression:
    """

    print("Data summary:")
    print(panda_data.describe())
    print()

    print("Standard Deviations:")
    print(panda_data.std())
    print()

    # Histograms:
    panda_data.hist()
    plt.tight_layout()
    plt.show()


def sigmoidal(a):
    """
    Returns the S-shaped sigmoid of an input.
    """

    return 1.0 / (1.0 + np.exp(-a))


def create_logistic_vector(input_vector, cutoff):
    """
    Creates a vector of 0s and 1s based on an input vector of numbers with a cut-off point.
    """

    output_vector = np.zeros(len(input_vector))
    n = 0
    for i in range(len(input_vector)):
        if input_vector[i] > cutoff:
            output_vector[i] = 1
        else:
            output_vector[i] = -1  # Set to -1 rather than 0 to help make later calculations easier.
        n += 1

    return output_vector


def main(header, inputs, targets):
    """
    Randomly select N instances of the data for training and N instances for testing.
    The wine quality scores will be converted to 1 or -1, dependent on whether the wine
    is "Good" or "Bad".
    """
    # setting a seed to get the same pseudo-random results every time
    np.random.seed(30)

    print("\nLogistic regression analysis:\n")

    N = 50  # Change for different number of inputs/targets

    # print(data[:,target_col])
    targets = create_logistic_vector(targets, 5)  # The quality scores are converted to logistic form.
    # print(targets)

    # Select a random vector of indexes for the training and testing:
    training_is = np.random.randint(inputs.shape[0], size=N)
    test_is = np.random.randint(inputs.shape[0], size=N)
    
    TRAINING_INPUTS = inputs[training_is, :]
    TRAINING_TARGETS = targets[training_is]

    TEST_INPUTS = inputs[test_is, :]
    TEST_TARGETS = targets[test_is]
    
    # Run the tests for several different regularisation strengths.  This will show
    # the degree of over-fitting when lambda is close to 0
    lams = [0, .001, .01, .1, 1]
    for j, l in enumerate(lams):

        lr = Logistic_Regression(training_inputs=TRAINING_INPUTS, training_targets=TRAINING_TARGETS,
                                 test_inputs=TEST_INPUTS, test_targets=TEST_TARGETS, lam=l)

        print("Initial likelihood of the data:")
        print(lr.lik(lr.betas))

        lr.train_data()

        print("Final beta values:")
        print(lr.betas)
        print("Final likelihood:")
        print(lr.lik(lr.betas))

        plt.subplot(len(lams), 2, 2*j + 1)
        lr.plot_training_reconstruction()
        plt.ylabel("$\lambda$=%s" % l)
        if j == 0:
            plt.title("Training Data Reconstructions")

        plt.subplot(len(lams), 2, 2*j + 2)
        lr.plot_test_predictions()
        if j == 0:
            plt.title("Test Data Predictions")

    plt.savefig("../plots/simple_linear/logistic_regression.png", fmt="png")
    plt.show()


class Logistic_Regression():
    """
    A Logistic Regression model that uses L2 regularisation to help control
    over-fitting.
    Priors are assumed to be Gaussian with zero mean.
    """

    def __init__(self, training_inputs=None, training_targets=None, test_inputs=None, test_targets=None,
                    lam=.1, synthetic=False):

        # Lambda is the regularisation parameter:
        self.lam = lam

        # Set the wine quality data:
        self.set_data(training_inputs, training_targets, test_inputs, test_targets)

        # Initialise the parameters to zero:
        self.betas = np.zeros(self.training_inputs.shape[1])

    def lik(self, betas):
        """
        Returns the likelihood of the data.  The first sum is the likelihood of the data, the
        second is the likelihood of the prior subtracted (this is the regularisation)
        """

        likelihood = 0

        for i in range(self.n):
            likelihood += np.log(sigmoidal(self.training_targets[i] * \
                            np.dot(betas, self.training_inputs[i,:])))

        for j in range(1, self.training_inputs.shape[1]):
            likelihood -= (self.lam / 2.0) * self.betas[j]**2

        return likelihood
        
    def negative_lik(self, betas):
        return -1 * self.lik(betas)
        
    def plot_test_predictions(self):
        """
        Plots the test data.
        """

        plt.plot(np.arange(self.n), .5 + .5*self.test_targets, 'go')
        plt.plot(np.arange(self.n), self.test_predictions(), 'rx')
        plt.ylim([-.1, 1.1])
        
    def plot_training_reconstruction(self):
        """
        Plots the training data.
        """

        plt.plot(np.arange(self.n), .5 + .5*self.training_targets, 'go')
        plt.plot(np.arange(self.n), self.training_reconstruction(), 'rx')
        plt.ylim([-.1, 1.1])

    def set_data(self, training_inputs, training_targets, test_inputs, test_targets):
        """
        Allows us to set the wine data into this class for training.
        """

        self.training_inputs = training_inputs
        self.training_targets = training_targets
        self.test_inputs = test_inputs
        self.test_targets = test_targets
        self.n = training_targets.shape[0]
        
    def test_predictions(self):
        """
        Tests the predictions made using the training set on the test set.
        """

        p_y1 = np.zeros(self.n)
        for i in range(self.n):
            p_y1[i] = sigmoidal(np.dot(self.betas, self.test_inputs[i,:]))

        return p_y1

    def training_reconstruction(self):
        """
        Calculates the densities of y (target) being 1 for the training set, from which
        the betas were calculated.
        """

        p_y1 = np.zeros(self.n)
        for i in range(self.n):
            p_y1[i] = sigmoidal(np.dot(self.betas, self.training_inputs[i,:]))

        return p_y1
        

    def train_data(self):
        """
        Defines the gradient to be trained.  This is then optimised using scipy's fmin_bfgs function.
        """

        # Define derivative of likelihood wrt beta_k.
        # This is then multiplied by -1 because we will be minimising the derivative.
        dB_k = lambda B, k : (k > 0) * self.lam * B[k] - np.sum([ \
                                        self.training_targets[i] * self.training_inputs[i, k] * \
                                        sigmoidal(-self.training_targets[i] *\
                                                np.dot(B, self.training_inputs[i,:])) \
                                        for i in range(self.n)])

        # Full gradient is an array of componentwise derivatives:
        dB = lambda B : np.array([dB_k(B, k) \
                                    for k in range(self.training_inputs.shape[1])])

        # Optimise:
        self.betas = fmin_bfgs(self.negative_lik, self.betas, fprime=dB)
