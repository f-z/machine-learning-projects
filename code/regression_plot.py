import numpy as np
import matplotlib.pyplot as plt


def plot_train_test_errors(
        control_var, experiment_sequence, train_errors, test_errors, test_error_linear):
    """
    Plot the train and test errors for a sequence of experiments.

    parameters
    ----------
    control_var - the name of the control variable, e.g. degree (for polynomial).
    experiment_sequence - a list of values applied to the control variable.
    """

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    train_line, = ax.plot(experiment_sequence, train_errors, 'b-')
    test_line, = ax.plot(experiment_sequence, test_errors, 'r-')
    linear_test_line, = ax.plot(experiment_sequence, np.repeat(test_error_linear, len(experiment_sequence)), 'g:')
    ax.set_xlabel(control_var)
    ax.set_ylabel("$E_{RMS}$")
    ax.legend([train_line, test_line, linear_test_line], ["train error", "test error", "linear test error"])

    return fig, ax


def plot_function_and_data(inputs, targets, marker_size=2, **kwargs):
    """
    Plot a function and some associated regression data in a given range.

    parameters
    ----------
    inputs - the input data
    targets - the targets
    marker_size (optional) - the size of the markers in the plotted data
    <for other optional arguments see plot_function>

    returns
    -------
    fig - the figure object for the plot
    ax - the axes object for the plot
    lines - a list of the line objects on the plot
    """

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    line, = ax.plot(inputs, targets, 'bo', markersize=marker_size)
    return fig, ax, [line]


def plot_function_data_and_approximation(
        predict_func, inputs, targets, line_width=2, x_lim=None,
        **kwargs):
    """
    Plot regression data and an approximation
    in a given range.

    parameters
    ----------
    predict_func - the approximating function
    inputs - the input data
    targets - the targets
    <for optional arguments see plot_function_and_data>

    returns
    -------
    fig - the figure object for the plot
    ax - the axes object for the plot
    lines - a list of the line objects on the plot
    """

    if x_lim is None:
        x_lim = (5, 17.5)
    fig, ax, lines = plot_function_and_data(
        inputs, targets, linewidth=line_width, xlim=x_lim, **kwargs)
    xs = np.linspace(0, 100, 101)
    ys = predict_func(xs)
    line, = ax.plot(xs, ys, 'r-', linewidth=line_width)
    lines.append(line)

    return fig, ax, lines


def exploratory_plots(data, field_names=None):
    # getting the number of dimensions in the data
    dim = data.shape[1]
    # creating an empty figure object
    fig = plt.figure()
    # creating a grid of four axes
    plot_id = 1
    for i in range(dim):
        for j in range(dim):
            ax = fig.add_subplot(dim, dim, plot_id)
            # if it is a plot on the diagonal we histogram the data
            if i == j:
                ax.hist(data[:, i])
            # otherwise we scatter plot the data
            else:
                ax.plot(data[:, i], data[:, j], 'o', markersize=1)
            # we're only interested in the patterns in the data, so there is no
            # need for numeric values at this stage
            ax.set_xticks([])
            ax.set_yticks([])
            # if we have field names, then label the axes
            if field_names is not None:
                if i == (dim - 1):
                    ax.set_xlabel(field_names[j])
                if j == 0:
                    ax.set_ylabel(field_names[i])
            # incrementing the plot_id
            plot_id += 1
    plt.tight_layout()
