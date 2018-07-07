import numpy as np
import numpy.linalg as linear_alg


def ml_weights(input_matrix, targets):
    """
    This method returns the weights that give the best linear fit between
    the processed inputs and the targets.
    """

    # phi = np.matrix(input_matrix)
    phi = np.array(input_matrix)
    # targets = np.matrix(targets).reshape((len(targets), 1))
    targets = np.array(targets).reshape((len(targets), 1))
    # weights = linear_alg.inv(phi.transpose()*phi)*phi.transpose()*targets
    # a = phi.transpose()* phi
    a = np.dot(phi.T, phi)
    inv = linear_alg.inv(a)
    dot = np.dot(inv, phi.T)
    weights = np.dot(dot, targets)
    # weights = np.poly1d(np.polyfit(input_matrix, targets, 1))

    return np.array(weights).flatten()


def regularised_ml_weights(
        input_matrix, targets, reg_param):
    """
    This method returns the weights that give the best linear fit between
    the processed inputs and the targets, penalised by some regularisation term
    (reg_param).
    """

    phi = np.matrix(input_matrix)
    targets = np.matrix(targets).reshape((len(targets), 1))
    i = np.identity(phi.shape[1])
    weights = linear_alg.inv(reg_param*i + phi.transpose()*phi)*phi.transpose()*targets

    return np.array(weights).flatten()


def linear_model_predict(design_matrix, weights):
    ys = np.matrix(design_matrix)*np.matrix(weights).reshape((len(weights), 1))

    return np.array(ys).flatten()


# this is the feature mapping for a polynomial of given degree in 1d
def expand_to_monomials(inputs, degree):
    """
    Create a design matrix from a 1d array of input values, where columns
    of the output are powers of the inputs from 0 to degree (inclusive).

    So if input is: inputs=np.array([x1, x2, x3])  and degree = 4 then
    output will be design matrix:
        np.array( [[  1.    x1**1   x1**2   x1**3   x1**4   ]
                   [  1.    x2**1   x2**2   x2**3   x2**4   ]
                   [  1.    x3**1   x3**2   x3**3   x3**4   ]]).
    """

    expanded_inputs = []
    for i in range(degree+1):  # because inclusive
        expanded_inputs.append(inputs**i)

    return np.array(expanded_inputs).transpose()


def construct_polynomial_approx(degree, weights):
    """
    This function creates and returns a prediction function based on a
    feature mapping and some weights.

    The returned prediction function takes a set of input values and returns
    the predicted output for each.
    """

    # here is a function that is created on the fly from the input feature
    # mapping and weights
    def prediction_function(xs):
        expanded_xs = np.matrix(expand_to_monomials(xs, degree))
        ys = expanded_xs*np.matrix(weights).reshape((len(weights), 1))

        return np.array(ys).flatten()

    # We return the function reference (handle) itself. This can be used like
    # any other function.
    return prediction_function


def construct_feature_mapping_approx(feature_mapping, weights):
    """
    This function creates and returns a prediction function based on a
    feature mapping and some weights.

    The returned prediction function takes a set of input values and returns
    the predicted output for each.
    """
    # here is a function that is created on the fly from the input feature
    # mapping and weights
    def prediction_function(xs):
        design_matrix = np.matrix(feature_mapping(xs))
        return linear_model_predict(design_matrix, weights)

    # returning the function reference (handle) itself,
    # which can be used like any other function
    return prediction_function


def construct_rbf_feature_mapping(centres, scale):
    """
    parameters
    ----------
    centres - a DxM matrix (numpy array) where D is the dimension of the space
        and each row is the central position of an rbf basis function.
        For D=1 can pass an M-vector (numpy array).
    scale - a float determining the width of the distribution. Equivalent role
        to the standard deviation in the Gaussian distribution.

    returns
    -------
    feature_mapping - a function which takes an NxD data matrix and returns
        the design matrix (NxM matrix of features)
    """

    #  to enable python's broadcasting capability we need the centres
    # array as a 1xDxM array
    if len(centres.shape) == 1:
        centres = centres.reshape((1, 1, centres.size))
    else:
        centres = centres.reshape((1, centres.shape[1], centres.shape[0]))
    # the denominator
    denom = 2*scale**2

    # now create a function based on these basis functions
    def feature_mapping(datamtx):
        #  to enable python's broadcasting capability we need the datamtx array
        # as a NxDx1 array
        if len(datamtx.shape) == 1:
            # if the datamtx is just an array of scalars, turn this into
            # a Nx1x1 array
            datamtx = datamtx.reshape((datamtx.size, 1, 1))
        else:
            # if datamtx is NxD array, then we reshape matrix as a
            # NxDx1 array
            datamtx = datamtx.reshape((datamtx.shape[0], datamtx.shape[1], 1))
        return np.exp(-np.sum((datamtx - centres)**2,1)/denom)
    # return the created function
    return feature_mapping


def construct_knn_approx(train_inputs, train_targets, k):  
    """
    For 1 dimensional training data, it produces a function:reals-> reals
    that outputs the mean training value in the k-Neighbourhood of any input.
    """

    # creating Euclidean distance
    distance = lambda x, y: (x-y)**2
    train_inputs = np.resize(train_inputs, (1, train_inputs.size))

    def prediction_function(inputs):
        inputs = inputs.reshape((inputs.size, 1))
        distances = distance(train_inputs, inputs)
        predicts = np.empty(inputs.size)
        for i, neighbourhood in enumerate(np.argpartition(distances, k)[:, :k]):
            # the neighbourhood is the indices of the closest inputs to xs[i]
            # the prediction is the mean of the targets for this neighbourhood
            predicts[i] = np.mean(train_targets[neighbourhood])
        return predicts

    # returning a handle to the locally defined function
    return prediction_function


def calculate_weights_posterior(designmtx, targets, beta, m0, S0):
    """
    Calculates the posterior distribution (multivariate gaussian) for weights
    in a linear model.

    parameters
    ----------
    designmtx - 2d (N x M) array of inputs (data-matrix or design-matrix) where
        N is the number of data-points and each row is that point's representation
    targets - 1d (N)-array of target values
    beta - the known noise precision
    m0 - prior mean (vector) 1d-array (or array-like) of length M
    S0 - the prior covariance matrix 2d-array

    returns
    -------
    mN - the posterior mean (vector)
    SN - the posterior covariance matrix
    """

    N, M = designmtx.shape
    Phi = np.matrix(designmtx)
    t = np.matrix(targets).reshape((N, 1))
    m0 = np.matrix(m0).reshape((M, 1))
    S0_inv = np.matrix(np.linalg.inv(S0))
    SN = np.linalg.inv(S0_inv + beta * Phi.transpose() * Phi)
    mN = SN * (S0_inv * m0 + beta * Phi.transpose() * t)

    return np.array(mN).flatten(), np.array(SN)


def predictive_distribution(designmtx, beta, mN, SN):
    """
    Calculates the predictive distribution a linear model. This amounts to a
    mean and variance for each input point.

    parameters
    ----------
    designmtx - 2d (N x K) array of inputs (data-matrix or design-matrix) where
        N is the number of data-points and each row is that point's
        representation
    beta - the known noise precision
    mN - posterior mean of the weights (vector) 1d-array (or array-like)
        of length K
    SN - the posterior covariance matrix for the weights 2d (K x K)-array

    returns
    -------
    ys - a vector of mean predictions, one for each input datapoint
    sigma2Ns - a vector of variances, one for each input data-point
    """

    N, K = designmtx.shape
    Phi = np.matrix(designmtx)
    mN = np.matrix(mN).reshape((K, 1))
    SN = np.matrix(SN)
    ys = Phi * mN
    # create an array of the right size with the uniform term
    sigma2Ns = np.ones(N) / beta
    for n in range(N):
        # now calculate and add in the data dependent term
        # NOTE: I couldn't work out a neat way of doing this without a for-loop
        # NOTE: but if anyone can please share the answer.
        phi_n = Phi[n, :].transpose()
        sigma2Ns[n] += phi_n.transpose() * SN * phi_n
    return np.array(ys).flatten(), np.array(sigma2Ns)
