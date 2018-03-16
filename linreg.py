#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

class MyLinearRegressor():

    def __init__(self, kappa=0.1, lamb=0, max_iter=200, opt='sgd'):
        self._kappa = kappa
        self._lamb = lamb # for bonus question
        self._opt = opt
        self._max_iter = max_iter

    def fit(self, X, y):
        X = self.__feature_rescale(X)
        X = self.__feature_prepare(X)
        error = []
        if self._opt == 'sgd':
            error = self.__stochastic_gradient_descent(X, y)
        elif self._opt == 'batch':
            error = self.__batch_gradient_descent(X, y)
        else:
            print('unknow opt')
        return error

    def predict(self, X):
        pass

    def __batch_gradient_descent(self, X, y):
        N, M = X.shape
        iterator = 0
        iter_n = [] 
        error = []
        self._w = np.ones(X.shape[1])
        print('X[0] is: ', X[0])
        print('y[0] is :' , y[0])
        
        ##############################
        for niter in range(self._max_iter):
            self._w = self._w - self._kappa * (X.T.dot(X.dot(self._w)-y)/N)
           
            #y_hat = np.dot(X,self._w)
            #self._w = self._w + self._kappa *np.dot(X.transpose(),y_hat)
            
           
            print('in iteration ' , niter , 'error is: ' , self.__total_error(X, y, self._w))
            error.append(self.__total_error(X, y, self._w))
            iterator = iterator + 1
            #print (error)
            
            iter_n.append(iterator)
        #plt.plot(iter_n, error)
        #print (iter_n)
        #print (error)
        fig1 = plt.gcf()
        plt.plot( iter_n,error )
        plt.xlabel('iteration')
        plt.ylabel('cost function')
        plt.show()
        fig1.savefig('batch_kappa=0.1.png')
            #  put your code here
            #
            ##############################
        return error

    def __stochastic_gradient_descent(self, X, y):
        N, M = X.shape
        niter = 0
        
        iterator = 0
        iter_n = [] 
        error = []
        self._w = np.ones(X.shape[1])
        
        ##############################
        #np.random.shuffle(X)
        for niter in range(self._max_iter):
            for i in range(N):
        #np.random.shuffle(X)
            
                self._w = self._w + self._kappa * (X[i]*(y[i] - X[i].dot(self._w)))
            
           
                
            error.append(self.__total_error(X, y, self._w))
            iterator = iterator + 1
            iter_n.append(iterator)
        fig2 = plt.gcf()    
        plt.plot( iter_n,error)
        plt.xlabel('Iteration')
        plt.ylabel('Cost function')
        plt.show()
        fig2.savefig('sgd_kappa=0.1.png')
        #  put your code here
        #
        ##############################
        return error

    def __total_error(self, X, y, w):
        ##############################
        error = (1/2)*(np.mean(np.power((y-np.dot(X,w)),2)))
        #  put your code here
        #
        ##############################
        return error

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
    from sklearn.datasets import load_boston

    data = load_boston()
    X, y = data['data'], data['target']
    mylinreg = MyLinearRegressor()
    mylinreg.fit(X, y)
    
    
    #print('This is X')
    #print(X)
    #print('size of X')
    #print(np.shape(X))
    #print('size of y')
    #print(np.shape(y))
    #print('This is y')
    #print(y)
