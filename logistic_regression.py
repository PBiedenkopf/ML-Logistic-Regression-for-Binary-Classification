""" 
    Linear logistic regression suitable for large datasets
    P. Biedenkopf - 16.12.2020

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

class LinearLogisticRegression:
    """ 
        Logistic regression class.
        members: 
            verbose     -   bool for getting detailed output
            costHist    -   Objective function history list
            method      -   optimization algorithm
            OptMaxIter  -   Max iteration number
            ftol        -   Stop criterion for optimization algorithm
            iter        -   Current iteration, is initialized as 0
            X           -   dataset for which a value gets predicted by the model
            y           -   labels of dataset (vector)
    """
    def __init__(self, X, y, method='L-BFGS-B', optMaxIter=300, verbose=False, ftol=1e-7):
        self.X = np.insert(X, 0, np.ones(len(y)), axis=1) # leading ones-vector for constant term theta0
        self.y = y
        self.optMaxIter = optMaxIter
        self.costHist = []
        self.iter = 0
        self.verbose = verbose
        self.ftol = ftol
        self.method = method
        
        
    def train(self):
        """ 
            Trains the model for the dataset given in constructor.
        """
        # initialize theta
        self.theta = np.zeros(self.X.shape[1])
        
        opts = {'disp': False, 'maxiter': self.optMaxIter, 'ftol': self.ftol}
        sol = opt.minimize(self.costFun, self.theta, method=self.method, jac=self.costFunGrad, options=opts)
        if sol.success:
            print("Optimization succeeded: {} | Solution: {:e} | {}".format(sol.success, sol.fun, sol.x))
            self.theta = np.array(sol.x)
            self.accuracy()

        if self.verbose:
            self.plotConvergence()
            print(f'Model accuracy is: {self.acc:.1f}% on training data')
        
    def predict(self, x):
        """ 
            Predicts value for a given datapoint x.
            variables: 
                t - datapoint for which a value gets predicted by the model
        """
        x = np.insert(x, 0, 1)
        prob = sigmoid(x.dot(self.theta));
        if self.verbose:
            print(f'Prediction for {x} is: {100*prob:.1f}%')
        return prob
    
    def accuracy(self, threshold=0.5):
        """ 
            Computes the accuracy of the trained model for the training set.
            variables: 
                threshold - threshold for accepting a value as 1
        """
        p = np.zeros([len(self.y), ]);
        for i in range(len(self.y)):
            if sigmoid(self.X[i].dot(self.theta)) >= threshold:
                p[i] = 1;
            else:
                p[i] = 0;
                
        self.acc = np.mean(p == self.y) * 100
    
    def costFun(self, theta):
        """
                Returns objective value for measuring fitness of model.
                variables: 
                    theta   - current model parameters
        """
        if self.verbose:
            print("Iter: {} | theta: {}".format(self.iter, theta))
        J = 0
        m = len(self.y)
        
        cost = -self.y*np.log(sigmoid(self.X.dot(theta))) - (np.ones([m,])-self.y)*np.log(1-sigmoid(self.X.dot(theta)))
        J = 1/m * sum(cost)
        
        self.costHist.append(J)
        self.theta = theta
        self.iter += 1
        return J
    
    def costFunGrad(self, theta):
        """
                Returns gradient of the objective.
                variables: 
                    theta   - current model parameters
        """
        Grad = np.zeros(theta.shape);
        m = len(self.y)
        
        for i in range(len(theta)):
            Grad[i] = 1/m * sum( (sigmoid(self.X.dot(theta)) - self.y) * self.X[:,i] )
            
        return Grad
    
    def plotConvergence(self):
        """
            Plots the convergence of the model over the iterations
        """
        plt.figure(2)
        plt.plot(range(0,len(self.costHist)), self.costHist)
        plt.grid(True)
        plt.xlabel("Iteration")
        plt.ylabel("Objective")
        plt.title("Convergence")
        plt.legend()
        
    def plotDecisionBoundary2D(self):
        """
            Plots the dataset and the decision boundary of the model
        """
        plt.figure(3)
        X1 = self.X[:,1]
        X2 = self.X[:,2]
        pos = np.where(self.y == 1)
        neg = np.where(self.y == 0)
        plt.scatter(X1[pos], X2[pos], label='positive', marker='.', color='b')
        plt.scatter(X1[neg], X2[neg], label='negative', marker='+', color='r')
        plt.xlabel(r'$X_2$')
        plt.ylabel(r'$X_1$')
           
        # decision boundary
        x = np.linspace(np.amin(X1), np.amax(X1))
        y = (-1/self.theta[2])*(self.theta[1]*x + self.theta[0])
        plt.plot(x, y, label="Decision Boundary", color='black')
        plt.legend()
        
    def __str__(self):
        return "Logistic Regression with regularisation: {} and max. {} Iterations.".format(self.Lambda, self.optMaxIter)


def sigmoid(z):
    """
        Returns value of the sigmoid function for z
    """
    return 1 / (1 + np.exp(-z));
