""" 
    Linear logistic regression suitable for large datasets
    P. Biedenkopf - 16.12.2020

"""

import logistic_regression as lr
import numpy as np


# Manufacturing quality testing dataset
dataset = './datasets/dataset.csv'
data = np.genfromtxt(dataset, delimiter=',', skip_header=1)
y = data[:,2]
X = data[:,:2]

# Create algorithm instance
model = lr.LinearLogisticRegression(X, y, verbose=True, method='L-BFGS-B')

# Fit model with training data
model.train()
model.plotDecisionBoundary2D()

# Make prediction for new datapoint
pred = [50, 85]
model.predict(pred)
