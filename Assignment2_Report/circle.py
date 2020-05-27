# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 16:06:53 2018

@author: mahsa
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


NUM_BATCH = 500 
BATCH_SIZE = 256  
PRINT_INTERVAL = 20


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
     
      
        self.l1 = nn.Linear(2,2048)#features=2, neurons of hidden layers = 100
        self.l2 = nn.Linear(2048,1024)
        self.l3 = nn.Linear(1024,512)
        self.l4 = nn.Linear(512,256)
        self.l5 = nn.Linear(256,128)
        self.l6 = nn.Linear(128,1)#output= 1 (binary classification)

    def forward(self, x):
        x = self.l1(x) 
        x = F.relu(x) 
        x = self.l2(x) 
        x = F.relu(x)
        x = self.l3(x)
        x = F.relu(x)
        x = self.l4(x)
        x = F.relu(x)
        x = self.l5(x)
        x = F.relu(x)
        x = self.l6(x)
        x = F.relu(x)
        return x #probability of belonging to one class
# may be of use to you
# returns the percentage of predictions (greater than threshold) 
# that are equal to the labels provided

def percentage_correct(pred, labels, threshold = 0.5):
    #pred_label = np.ones(labels.shape[0])
    #corr_count = 0
    pred = pred > threshold # if pred > threshold labels it as 1, otherwise label it as 0
    pred_corr = torch.eq(pred.long(),labels.long()) 
    return (torch.div(pred_corr.long().sum().float(), pred.shape[0]))

# This code generates 2D data with label 1 if the point lies
# outside the unit circle.
def get_batch(batch_size):
    # Data has two dimensions, they are randomly generated
    data = (torch.rand(batch_size,2)-0.5)*2.5
    # square them and sum them to define the decision boundary
    # (x_1)^2 + (x_2)^2 = 1
   # print(data)
    square = torch.mul(data,data)
    square_sum = torch.sum(square,1,keepdim=True)
    # Generate the labels
    # outside the circle is 1
    #labels are 0 (inside circle), 1(outside circle)
    labels = square_sum>1

    return Variable(data), Variable(labels.float())

def plot_decision_boundary(data_in, preds):
    dic= defaultdict(lambda: "r")
    dic[0] = 'b'
    colour = list(map(lambda x: dic[x[0]], preds.data.numpy()>0.5))
    x = data_in.data.numpy()[:,0]
    y = data_in.data.numpy()[:,1]
    
    #plt.clf()
    fig2 = plt.gcf() 
    plt.scatter(x,y,c=colour)
    plt.title("Decision Boundary of a Neural Net Trained to Classify the Unit Circle")
    plt.show()
    # May be of use for saving your plot:    plt.savefig(filename)
    fig2.savefig('decision_boundary.png')
def plot_percent_correct(data_in, percent_corr):
    fig1 = plt.gcf() 
    plt.plot(data_in, percent_corr)
    plt.xlabel("Iteration") 
    plt.ylabel("Percentage Correct")
    plt.show()
    fig1.savefig('percent correct.png')

# optimization

model = Classifier()
o = torch.optim.SGD(model.parameters(), lr = 0.001)  #this is optimizer
loss = nn.BCELoss() #loss function, binary cross entropy-N
# plot decision boundary for new data

percent_corr = [] # stores percent correct plotting
i_list = []
n =0 # for plotting x-axis for question 1a
model.train()
for i in range(NUM_BATCH):
    data, labels = get_batch(BATCH_SIZE)
    pred = model(data) #call model with input
    error = loss(pred, labels)
    o.zero_grad() # reset the gradients to zero
    error.backward() 
    o.step()
    n += 1
    i_list.append(n)
    percent_corr.append(percentage_correct(pred, labels).data.numpy())
    
plot_percent_correct(i_list, percent_corr)

# plt.plot(i_list, percent_corr)
# plt.show()


d, labels = get_batch(BATCH_SIZE)

plot_decision_boundary(d, model(d)) 



