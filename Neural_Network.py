# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 11:24:50 2017

@author: kavya
"""

# Neural Networks
from math import exp
from random import seed
import numpy as np
from collections import defaultdict
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import LeaveOneOut

#forward propagation
def forward_propagation(network, ip):
    inputs = ip
    for layer in network:
        new_ips = []
        for n in layer:
            a = activate(n['weights'], inputs)
            n['output'] = 1.0 / (1.0 + exp(-a))
            new_ips.append(n['output'])
        inputs = new_ips
    return inputs


# back propagation
def backward_propagation(network, ex_output):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network) - 1:
            for j in range(len(layer)):
                error = 0.0
                for n in network[i + 1]:
                    error += (n['weights'][j] * n['bias'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                n = layer[j]
                errors.append(ex_output[j] - n['output'])
        for j in range(len(layer)):
            n = layer[j]
            n['bias'] = errors[j] * (n['output']* (1.0 - n['output']))
            
#activation function
def activate(weights, ips):
    a = weights[-1]
    for i in range(len(weights) - 1):
        a += weights[i] * ips[i]
    return a

# Update  weights with error through backpropagation
def update_weights(network, row, learning_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i !=0:
            inputs = [n['output'] for n in network[i - 1]]
        for n in network[i]:
            for j in range(len(inputs)):
                n['weights'][j] += learning_rate * n['bias'] * inputs[j]
            n['weights'][-1] += learning_rate * n['bias']


#preperaing the iris dataset
seed(1)
iris_dataset = datasets.load_iris()
X1 = iris_dataset.data[50:150,:]
X = np.array(X1[:,2:4])

#Normalizing the data using MinMax normalization
scaler = MinMaxScaler()
scaler.fit(X)
X = (scaler.transform(X))
target= np.array(iris_dataset.target[50:150]).reshape(100,1)

#converting 1's to 0's and 2's to 1's
target[target==1]=0
target[target==2]=1

no_inputs = 2
no_outputs = 1
no_iter=10
iter_list = []
e = []
e_1 =[]
weight1_list=[]
weight2_list=[]

wt={}


loo = LeaveOneOut()

for train_index, test_index in loo.split(X1):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    print(X_train, X_test, target)

    
    #initializing weights with random values between 0 and 1    
    network = list()
    hidden_layer = [{'weights': [np.random.uniform(0.0, 1.0) for i in range(no_inputs + 1)]} for i in range(2)]
    network.append(hidden_layer)
    output_layer = [{'weights': [np.random.uniform(0.0, 1.0) for i in range(3)]} for i in range(no_outputs)]
    network.append(output_layer)
    newwt=defaultdict(list)

    #training the network
    for n in range(no_iter):
        error = 0
        for row in X_train:
            outputs = forward_propagation(network, row)
            expected_op = [0 for i in range(no_outputs)]
            #expected[int(row[-1])] = 1
            #cost 
            error += sum(([np.sum(np.multiply(expected_op[i], np.log(outputs[i])) + np.multiply((1-expected_op[i]), np.log(1-outputs[i]))) for i in range(len(expected_op))]))
            error = -(error)
            backward_propagation(network, expected_op)
            update_weights(network, row, 0.5)
        #Appending the weights of individual parameters to the lists
        iter_list.append(n)
        e.append(error)
        error = int((error)*100)
    print('>Iteration=%d, error=%.3f' % (n, error))
    e_1.append(error)
    
length = len(e_1)    
average = ((sum(e_1))/length)
print("Average error rate for prediction= %.2f %%"%(average))


