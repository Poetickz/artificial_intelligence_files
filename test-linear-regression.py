#!/usr/bin/python3.5
""" test-linear-regression.py
    This script tests the Gradient Descent algorithm for multivariate
    linear regression.

    Author: Alan Rocha González
    Institution: Universidad de Monterrey
    First created: Sun 30 March, 2020
    Email: alan.rocha@udem.edu
"""
# import standard libraries
import numpy as np
import pandas as pd

# import user-defined libraries
import utilityfunctions as uf

# load training data
x_training, y_training, mean_list, std_list = uf.load_data('training-data-multivariate.csv')

# declare and initialise hyperparameters
learning_rate = 0.05

# Initailize all w
w = np.array([[0.0]*x_training.T.shape[0]]).T

# define stopping criteria
stopping_criteria = 0.01

# run the gradient descent method for parameter optimisation purposes
w = uf.gradient_descent(x_training, y_training, w, stopping_criteria, learning_rate)

#