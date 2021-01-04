""" 
    Linear logistic regression suitable for large datasets
    P. Biedenkopf - 16.12.2020

"""

import numpy as np

def sigmoid(z):
    """
        Returns value of the sigmoid function for z
    """
    return 1 / (1 + np.exp(-z));