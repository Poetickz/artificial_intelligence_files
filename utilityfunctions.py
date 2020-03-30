"""utility libraries for AI"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

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

def feature_scaling(data):
    """ standarize the x data and saves mean value & std value"""
    mean_value = data.mean()
    std_value =  data.std()
    scaling_scores = (data - mean_value) / std_value
    return scaling_scores, mean_value, std_value

def load_data(path_and_filename):
    """ load data from comma-separated-value (CSV) file """

    # load training-data file
    training_data = pd.read_csv(path_and_filename)

    # retrieve x data from training data; this is converted from Pandas.DataFrame
    #to numpy
    n_rows, n_columns = training_data.shape

    x_data = pd.DataFrame.to_numpy(training_data.iloc[:,0:n_columns-1])

    # retrieve y data from training data; this is converted from Pandas.DataFrame
    # to numpy

    y = pd.DataFrame.to_numpy(training_data.iloc[:,-1]).reshape(n_rows, 1)

    # prints x and y data 
    print("--"*23)
    print("Training Data Outputs")
    print("--"*23)
    for x,y in zip(x_data,y):
        print(x,y)

    # creates list to save mean, std and scaled x values
    new_scaled_x = []
    mean_list = []
    std_list = []

    # scales features
    for feature in x_data.T:
        new_data, mean_value, std_value = feature_scaling(feature)
        new_scaled_x.append(new_data)
        mean_list.append(mean_value)
        std_list.append(std_value)
    
    # Transpose to columns
    x_data = np.array(new_scaled_x).T

    # prints scaled x data
    print("\n")
    print("--"*23)
    print("Training Data Scaled")
    print("--"*23)
    for x,y in zip(x_data, x_data):
        print(x)
    
    # create a ones-vector of size as x
    one_data = np.ones(shape=n_rows)

    # concatenate one_data with x_data
    x = np.column_stack((x_data,one_data))
    
    # return x and y data
    return x, y, mean_list, std_list


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
