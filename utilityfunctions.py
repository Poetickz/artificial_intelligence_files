"""utility libraries for AI"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

def eval_hypothesis_function(w, x):
    """Evaluate the hypothesis function"""
    return np.matmul(w.T,x.T)


def compute_gradient_of_cost_function(x, y, w):
    """compute gradient of cost function"""

    # compute number of training samples
    Nfeatures = x.shape[1]
    Nsamples = x.shape[0]

    # evaluate hypotesis function
    hypothesis_function = eval_hypothesis_function(w, x)

    # compute difference between hypothesis function and label
    y = y.T
    residual =  hypothesis_function - y

    # multiply the residual by the input features x; 
    gradient_of_cost_function = np.matmul(residual,x)

    # sum the result up and divide the total by the number of samples N
    gradient_of_cost_function = sum(gradient_of_cost_function)/Nsamples
    
    # reshape the gradient of cost function from a 1xNsample to a Nsamplex1 matrix
    gradient_of_cost_function = np.reshape(gradient_of_cost_function,(Nfeatures,1))

    # return the gradient of cost function
    return gradient_of_cost_function


def compute_L2_norm(gradient_of_cost_function):
    """compute L2-norm of gradient of cost function"""
    return np.linalg.norm(gradient_of_cost_function)

def feature_scaling(data, mean_value, std_value):
    """ standarize the x data and saves mean value & std value"""
    if mean_value == 0 and std_value == 0:
        std_value=data.std()
        mean_value=data.mean()
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
    y_data = y

    # prints x and y data 
    print("--"*23)
    print("Training Data")
    print("--"*23)
    for x,y in zip(x_data,y):
        print(x,y)

    # creates list to save mean, std and scaled x values
    new_scaled_x = []
    mean_list = []
    std_list = []

    # scales features
    for feature in x_data.T:
        new_data, mean_value, std_value = feature_scaling(feature,0,0)
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
    x = np.column_stack((one_data,x_data))
    
    # return x and y data
    return x, y_data, mean_list, std_list


def gradient_descent(x_training, y_training, w, stopping_criteria, learning_rate):
    """ run the gradient descent algorith for optimisation"""

    # gradient descent algorithm
    while True:

        # compute gradient of cost function
        gradient_of_cost_function = compute_gradient_of_cost_function(x_training,
                                                                      y_training,
                                                                      w)
        # update parameters
        w = w - learning_rate*gradient_of_cost_function

        # compute L2 Norm
        L2_norm = compute_L2_norm(gradient_of_cost_function)
        if L2_norm < stopping_criteria:
            break
    # Print w parameters
    print("\n")
    print("--"*23)
    print("w parameter")
    print("--"*23)
    for i in range(0,len(w)):
        print("w"+str(i)+": "+str(w[i][0]))

    return w

def load_testing_data(path_and_filename, mean_list, std_list):
    """ load testing data from comma-separated-value (CSV) file """

    # load testing-data file
    testing_data = pd.read_csv(path_and_filename)
    
    # getting sizes
    n_rows, n_columns = testing_data.shape
    
    # Converting dataframe to numpy array
    x_data = pd.DataFrame.to_numpy(testing_data.iloc[:,0:n_columns])

    # prints traininh data
    print("--"*23)
    print("Testing Data")
    print("--"*23)
    for i in x_data:
        print(i)

    # standarize x_data values
    new_scaled_x = []
    for x,mean,std in zip(x_data.T,mean_list,std_list):
        new_data = feature_scaling(x,mean,std)
        new_scaled_x.append(np.array(new_data[0]))
    x_data = np.array(new_scaled_x).T

    # prints scaled x data
    print("\n")
    print("--"*23)
    print("Testing Data Scaled")
    print("--"*23)
    for x,y in zip(x_data, x_data):
        print(x)
    
    # create a ones-vector of size as x
    one_data = np.ones(shape=n_rows)

    # concatenate one_data with x_data
    x = np.column_stack((one_data, x_data))

    # return standarize x_data values in a numpy array
    return x

def predict(w,x,text):
    """ predict y with the w """

    # predicts
    print("\n")
    print("--"*23)
    print(text)
    print("--"*23)
    for i in range(0,len(x)):
        print(np.matmul(x[i],w.T[0]))