#!/usr/bin/python3.5
""" test-linear-regression.py
    This script tests the Gradient Descent algorithm for multivariate
    linear regression.

    Author: Andres Hernandez G.
    Institution: Universidad de Monterrey
    First created: Sun 8 March, 2020
    Email: andres.hernandezg@udem.edu

    Logs:
        + 2020-03-09: read data from file using Pandas
        + test multivariate linear regression
"""
# import standard libraries
import numpy as np
import pandas as pd

# import user-defined libraries
import utilityfunctions as uf

# load training data
x_training, y_training = uf.load_data('training-data.csv')

# declare and initialise hyperparameters
learning_rate = 0.0001
w = np.array([[0.0],[0.0]])

# define stopping criteria
stopping_criteria = 0.01

# run the gradient descent method for parameter optimisation purposes
w = uf.gradient_descent(x_training, y_training, w, stopping_criteria, learning_rate)
