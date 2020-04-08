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

def main():
  # runs main code
  """
  INPUT: NONE
  OUTPUT: NONE
  """


  # load training data
  x_training, y_training, mean_list, std_list = uf.load_data('training-data-multivariate.csv')

  # declare and initialise hyperparameters
  learning_rate = 0.0005

  # Initailize all w
  w = np.array([[0.0]*x_training.T.shape[0]]).T

  # define stopping criteria
  stopping_criteria = 0.01

  # run the gradient descent method for parameter optimisation purposes
  w = uf.gradient_descent(x_training, y_training, w, stopping_criteria, learning_rate)

  # load testing data
  x_testing = uf.load_testing_data('testing-data-multivariate.csv', mean_list, std_list)

  # predict with testing data and w
  uf.predict(w,x_testing,"Last-mile cost [predicted]")

  # estimate all this learning rate
  learning_rates = [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
  # generate time table
  uf.times(x_training, y_training, stopping_criteria,learning_rates)

main()