"""utility libraries for AI"""

import numpy as np
import pandas as pd


def eval_hypothesis_function(w, x):
    """Evaluate the hypothesis function"""
    return np.matmul(w.T, x)


def compute_gradient_of_cost_function(x, y, w):
    """compute gradient of cost function"""

    # compute number of training samples
    N = x.shape[1]

    # evaluate hypotesis function
    hypothesis_function = eval_hypothesis_function(w, x)

    # compute difference between hypothesis function and labels
    residual =  np.subtract(hypothesis_function, y)

    # multiply the residual by the input features x; sum the result up
    # and divide the total by the number of samples N
    gradient_of_cost_function = ((residual*x).sum(axis=1)/N)

    # reshape the gradient of cost function from a 1x2 to a 2x1 matrix
    gradient_of_cost_function = np.reshape(gradient_of_cost_function,(2,1))

    # return the gradient of cost function
    return gradient_of_cost_function


def compute_L2_norm(gradient_of_cost_function):
    """compute L2-norm of gradient of cost function"""
    return np.sqrt(np.sum(gradient_of_cost_function**2))


def load_data(path_and_filename):
    """ load data from comma-separated-value (CSV) file """

    # load training-data file
    training_data = pd.read_csv('training-data.csv')

    # retrieve x data from training data; this is converted from Pandas.Series to
    # numpy data
    x_data = pd.Series.to_numpy(training_data['x'])

    # create a ones-vector of size as x
    one_data = 1

    # concatenate one_data with x_data vertically
    x = np.vstack((np.ones(len(x_data)),x_data))

    # retrieve labels from training data; these are converted from Pandas.Series
    # to numpy data
    y = pd.Series.to_numpy(training_data['y'])

    # return x and y data
    return x,y


def gradient_descent(x_training, y_training, w, stopping_criteria, learning_rate):
    """ run the gradient descent algorith for optimisation"""

    # gradient descent algorithm
    L2_norm = 100.0
    while L2_norm > stopping_criteria:

        # compute gradient of cost function
        gradient_of_cost_function = compute_gradient_of_cost_function(x_training,
                                                                      y_training,
                                                                      w)
        # update parameters
        w = w - learning_rate*gradient_of_cost_function

        # compute L2 Norm
        L2_norm = compute_L2_norm(gradient_of_cost_function)

        # print parameters
        print('w:{}, L2:{}'.format(w, L2_norm))

    return None
