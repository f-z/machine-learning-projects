import numpy as np
import pandas as pd


def import_from_csv(ifname, input_cols=None, target_col=None, classes=None):
    """
    Imports the iris data-set and generates exploratory plots

    parameters
    ----------
    ifname -- filename/path of data file.
    input_cols -- list of column names for the input data
    target_col -- column name of the target data
    classes -- list of the classes to plot

    returns
    -------
    inputs -- the data as a numpy.array object  
    targets -- the targets as a 1d numpy array of class ids
    input_cols -- ordered list of input column names
    classes -- ordered list of classes
    """
    # if no file name is provided then use synthetic data
    dataframe = pd.read_csv(ifname)
    N = dataframe.shape[0]
    # if no target name is supplied we assume it is the last colunmn in the 
    # data file
    if target_col is None:
        target_col = dataframe.columns[-1]
        potential_inputs = dataframe.columns[:-1]
    else:
        potential_inputs = list(dataframe.columns)
        # target data should not be part of the inputs
        potential_inputs.remove(target_col)
    # if no input names are supplied then use them all
    if input_cols is None:
        input_cols = potential_inputs
    print("input_cols = %r" % (input_cols,))
    # if no classes are specified use all in the dataset
    if classes is None:
        # get the class values as a pandas Series object
        class_values = dataframe[target_col]
        classes = class_values.unique()
    else:
        # construct a 1d array of the rows to keep
        to_keep = np.zeros(N,dtype=bool)
        for class_name in classes:
            to_keep |= (dataframe[target_col] == class_name)
        # now keep only these rows
        dataframe = dataframe[to_keep]
        # there are a different number of dat items now
        N = dataframe.shape[0]
        # get the class values as a pandas Series object
        class_values = dataframe[target_col]
    print("classes = %r" % (classes,))
    # We now want to translate classes to targets, but this depends on our 
    # encoding. For now we will perform a simple encoding from class to integer.
    targets = np.empty(N)
    for class_id, class_name in enumerate(classes):
        is_class = (class_values == class_name)
        targets[is_class] = class_id
    #print("targets = %r" % (targets,))

    # We're going to assume that all our inputs are real numbers (or can be
    # represented as such), so we'll convert all these columns to a 2d numpy
    # array object (don't be fooled by the name of the method as_matrix it does
    # not produce a numpy matrix object
    inputs = dataframe[input_cols].as_matrix()
    return inputs, targets, input_cols, classes

