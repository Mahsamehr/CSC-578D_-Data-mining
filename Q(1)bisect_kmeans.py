#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from numpy.random import randint

class MyBisectKmeans():

    def __init__(self, K):
        self._K = K

    def fit(self, X):
        #ArrayCluster is an array that stores labels of points
        #Firstly, all points are assigned to 0 cluster
        ArrayCluster = np.zeros(578)
        
        #labelTrack variable is defined to label the points
        labelTrack = 3
        
        for k in range(self._K-1):
                    
                    ArrayCluster = np.int_(ArrayCluster)
                    
                    #MostFrequentLabel finds the largest cluster to split
                    MostFrequentLabel = np.argmax(np.bincount(ArrayCluster))
                    
                    #indx stores the place of points belonging to the larger cluster then we can reach them and change their labels after applying 2-means on them
                    indx = np.where(ArrayCluster == MostFrequentLabel)
                    indx = np.asanyarray(indx[0])
                    
                    #Input is the points in dataset that belong to the worst cluster and should be fed in to 2-means
                    Input = np.array([])
                    for j in range(X.shape[0]):
                       if j in indx:
                        
                          Input = np.append(Input, X[j][:])
                         
                    Input = Input.reshape(indx.shape[0],2)

                 
                    OutPut = KMeans(n_clusters=2, random_state=0).fit(Input)
                    #out keeps the outcome (labels) of the 2-means
                    out = OutPut.labels_
                    
                    np.place(out, out==0, labelTrack)
                    labelTrack=labelTrack+1
                    np.place(out, out==1, labelTrack)
                    labelTrack=labelTrack+1  
                          
                    for t,s in zip(indx, out):
                        ArrayCluster[t] = s
                
        print(ArrayCluster)           
        return ArrayCluster           
        # your code goes here
        # you are welcome to use scikit learn's implementation of kmeans
        # but you must implement the bisecting algorithm yourself
        # you should return an array of ints corresponding to the cluster 
        # assignments [0 ... K-1]
        # This is stand in code that just randomly assigns points to clusters
        #return randint(0,self._K,len(X))


# plot clusters and color them based on the cluster assignment in preds
def plot_clusters(data_in, preds):
    plt.clf()
    plt.scatter(data_in[:, 0], data_in[:, 1], c=preds)
    plt.axis('equal')
    plt.title("Cluster Assignments")
    plt.show()
    # May be of use for saving your plot:    plt.savefig(filename)


if __name__ == '__main__':
    # This is an easy way to make data sampled from clusters
    # with equal variance.  You can use the same method to change
    # the variance of individual clusters
    n_samples = 578
    X, y = make_blobs(n_samples=n_samples, centers = 4)

    mbs = MyBisectKmeans(4)
    clusters_out = mbs.fit(X)
    plot_clusters(X, clusters_out)

    # Generate data with covarying dimensions
    # A linear algebra reminder of how to make transformation
    # matrices http://mathforum.org/mathimages/index.php/Transformation_Matrix
    random_state = 170
    X, y = make_blobs(n_samples=n_samples, centers = 4)
    transformation = [[1, 0], [1.5, 1]]
    X_shear = np.dot(X, transformation)

    clusters_out = mbs.fit(X_shear)
    plot_clusters(X_shear, clusters_out)





