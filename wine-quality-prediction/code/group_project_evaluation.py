import numpy as np
import import_explore
import regression_best_fit
import regression_linear_gradient_descent
import regression_rbf
import regression_rbf_cross_validation
import regression_bayesian_rbf
import regression_logistic
import regression_kNN


def main(name, delimiter, columns):
    data_frame = import_explore.import_pandas(name)
    regression_logistic.panda_logistic_information(data_frame)

    header, inputs, targets = import_explore.main(name, delimiter, columns)

    regression_best_fit.main(header, inputs, targets)

    regression_linear_gradient_descent.main(header, inputs, targets)

    test_error_linear, best_scale, best_reg_param, best_no_centres = regression_rbf.main(inputs, targets)

    regression_rbf_cross_validation.main(inputs, targets, test_error_linear,
                                         best_scale, best_reg_param, best_no_centres)

    regression_bayesian_rbf.main(inputs, targets, best_scale, best_no_centres)

    regression_logistic.main(header, inputs, targets)

    regression_kNN.test_error('winequality-red.csv')
    regression_kNN.cross_validation('winequality-red.csv')


if __name__ == '__main__':
    import sys
    # this allows you to pass the file name as the first argument when you call
    # your script from the command line
    if len(sys.argv) == 1:
        # reverting to default parameters (red wine, ; delimiter, all features)
        main('winequality-red.csv', ";", np.arange(0, 11))
    elif len(sys.argv) == 2:
        # passing the file name as the first argument
        main(sys.argv[1], ";", np.arange(0, 11))
    elif len(sys.argv) == 3:
        # passing the delimiter as the second argument
        main(sys.argv[1], sys.argv[2], np.arange(0, 11))
    elif len(sys.argv) == 4:
        # the third argument is a list of columns to use as input features
        # list is separated by ','
        custom_columns = list(map(int, sys.argv[3].split(",")))
        main(sys.argv[1], sys.argv[2], custom_columns)
