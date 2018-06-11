import import_explore
import regression_best_fit
from sklearn.linear_model import LinearRegression
import numpy as np
import statsmodels.api as sm


def main(name, delimiter):
    header, inputs, targets = import_explore.main(name, delimiter)

    regression_best_fit.main(header, inputs, targets)

    print("Both BVit and Eit as predictors")
    X = inputs
    y = targets
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()
    print(model.summary())

    print("Only BVit as predictor")
    bvit = inputs[:, 0]
    bvit = sm.add_constant(bvit)
    model = sm.OLS(y, bvit).fit()
    print(model.summary())

    print("Only Eit as predictor")
    eit = inputs[:, 1]
    eit = sm.add_constant(eit)
    model = sm.OLS(y, eit).fit()
    print(model.summary())


if __name__ == '__main__':
    import sys
    # this allows you to pass the file name as the first argument when you call
    # your script from the command line
    if len(sys.argv) == 1:
        # reverting to default parameters (red wine, ; delimiter, all features)
        main('data.csv', ",")


def print(s):
    with open('models.txt', 'w+') as f:
        print(s, file=f)


