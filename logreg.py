#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import math
import matplotlib.pyplot as plt

class MyLogRegressor():

    def __init__(self, kappa=0.1, max_iter=200):
        self._kappa = kappa
        self._max_iter = max_iter

    def fit(self, X, y):
        X = self.__feature_rescale(X)
        X = self.__feature_prepare(X)
        log_like = self.__batch_gradient_descent(X, y)
        return log_like

    def predict(self, X, w):
        ##############################
        
        self.z = X.dot(self._w)
        prediction = 1 / (1 + np.exp(self.z)) # gives predictions by using sigmoid function
        
        #  put your code here
        #
        ##############################
        #pass
        return prediction
    
    def __batch_gradient_descent(self, X, y):
        N, M = X.shape
        niter = 0
        iterator = 0
        iter_n = []
        ll = []
        error = []
        self._w = np.zeros(X.shape[1])
        
        ##############################
        
        for niter in range(self._max_iter):
                
                self._w = self._w + self._kappa * (1/N)*(X.T.dot( y - (np.exp(X.dot(self._w))*self.predict(X, self._w))))
                print (self.__log_like(X, y, self._w))
               
                ll.append(self.__log_like(X, y, self._w))
                
                iterator = iterator + 1
                iter_n.append(iterator) 
        fig1 = plt.gcf()       
        plt.plot(iter_n,ll)
      
        plt.xlabel('Iteration')
        plt.ylabel('loglikelihood')
        plt.show()
        fig1.savefig('Log_batch_kappa=0.1.png')
   
        #return ll

    def __total_error(self, X, y, w):
        ##############################
        error = (1/2)*(np.mean(np.power((y-np.dot(X,w)),2)))
        #  put your code here
        #
        ##############################
        return error

    def __log_like(self, X, y, w):
        ##############################
        ll = np.sum(y*X.dot(self._w) - np.log(1 + np.exp(X.dot(self._w))) ) # Output the loglikelihood
        #  put your code here
        #
        ##############################
        return ll
    

    # add a column of 1s to X
    def __feature_prepare(self, X_):
        M, N = X_.shape
        X = np.ones((M, N+1))
        X[:, 1:] = X_
        return X

    # rescale features to mean=0 and std=1
    def __feature_rescale(self, X):
        self._mu = X.mean(axis=0)
        self._sigma = X.std(axis=0)
        return (X - self._mu)/self._sigma


if __name__ == '__main__':
    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer()
    X, y = data['data'], data['target']
    mylinreg = MyLogRegressor()
    print(mylinreg.fit(X, y))
