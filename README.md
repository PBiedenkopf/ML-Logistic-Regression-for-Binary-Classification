# Binary Classification with Logistic Regression
Logistic regression uses a model to predict the probability for a specific event to happen. So it is 
capable to classify data into a so-called positive and negative group. To make a predictions the 
algorithm needs a training set to build a model based on a logistic function. This implementation of 
a simple logistic regression algorithm can separate data with a descision boundary that consits of a 
linear combination of the features. To train the model the [scipy](https://www.scipy.org/)-library is 
used. The default optimization algorithm in this implemenation is the 
[L-BFGS](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html) algorithm.
To learn more complex descision boundaries the logistic regression algorithm needs a regularisation, 
which will be added soon.

## Datasets
The given dataset is a simple example with two input features. The dataset contains the results of two 
tests of several students as features (X1, X2) and a classification whether the student is accepted 
to a university or not. So the task for our logistic regression model is to predict the probability of 
a student to get accepted, based on the results of the two tests.

![Alt text](./descision_boundary.JPG?raw=true "Title")

The figure shows the plotted data with the descision boundary for being accepted or not. The linear 
descision boundary predicts the results of the training data about 89% correctly. To increase the model
accuracy the model has to learn a more complex hypothesis.


## License
MIT License

Copyright (c) 2020 Philipp Biedenkopf
