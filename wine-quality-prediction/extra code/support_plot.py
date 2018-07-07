import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

def plot_class_histograms(inputs, targets, bins=20, colors=None, ax=None):
    """
    Plots histograms of 1d input data, split according to class

    parameters
    ----------
    inputs - 1d vector of input values (array-like)
    targets - 1d vector of class values as integers (array-like)
    colors (optional) - a vector of colors one per class
    ax (optional) - pass in an existing axes object (otherwise one will be
        created)
    """
    class_ids = np.unique(targets)
    num_classes = len(class_ids)
    # calculate a good division of bins for the whole data-set
    _, bins = np.histogram(inputs, bins=bins)
    # create an axes object if needed
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
    # create colors for classes if needed
    if colors is None:
        colors = cm.rainbow(np.linspace(0,1,num_classes))
    # plot histograms
    for i, class_id in enumerate(class_ids):
        class_inputs = inputs[targets==class_id]
        ax.hist(class_inputs, bins=bins, color=colors[i], alpha=0.6)
    return ax


def exploratory_plots_classes(data, targets, field_names=None, classes=None):
    """
    Plots scatter plots of input data, split according to class

    parameters
    ----------
    inputs - 2d data matrix of input values (array-like)
    targets - 1d vector of class values as integers (array-like)
    field_names - list of input field names
    """
    # the number of dimensions in the data
    dim = data.shape[1]
    class_ids = np.unique(targets)
    num_classes = len(class_ids)
    # acolor for each class
    colors=cm.rainbow(np.linspace(0,1,num_classes))
    # create an empty figure object
    fig = plt.figure()
    # create a grid of four axes
    plot_id = 1
    for i in range(dim):
        for j in range(dim):
            ax = fig.add_subplot(dim,dim,plot_id)
            lines = []
            for class_id, class_color in zip(class_ids, colors):
                class_rows = (targets==class_id)
                class_data = data[class_rows,:]
                # if it is a plot on the diagonal we histogram the data
                if i == j:
                   ax.hist(class_data[:,i],color=class_color, alpha=0.6)
                # otherwise we scatter plot the data
                else:
                    line, = ax.plot(
                        class_data[:,i],class_data[:,j], 'o',color=class_color,
                        markersize=1)
                    lines.append(line)
                # we're only interested in the patterns in the data, so there is no
                # need for numeric values at this stage
            if not classes is None and i == 0 and j == (dim-1):
                ax.legend(lines, classes)
            ax.set_xticks([])
            ax.set_yticks([])
            # if we have field names, then label the axes
            if not field_names is None:
                if i == (dim-1):
                    ax.set_xlabel(field_names[j])
                if j == 0:
                    ax.set_ylabel(field_names[i])
            # increment the plot_id
            plot_id += 1
    plt.tight_layout()
  

