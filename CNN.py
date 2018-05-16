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



# Calculate neuron activation for an input
def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]
    return activation



# Forward propagate input to a network output
def forward_propagation(network, k):
    inputs = k
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = 1.0 / (1.0 + exp(-activation))
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs


# Backpropagate error and store in neurons
def backward_propagation(network, expected_op):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network) - 1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected_op[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * (neuron['output']* (1.0 - neuron['output']))


# Update network weights with error
def update_weights(network, row, l_rate):
      for i in range(len(network)):
        if i == 0:
            ll = 'layer1'
        else:
            ll = 'layer2'
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        n = 0
        for neuron in network[i]:
            n = n + 1

            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
                neu = ll + 'neuron' + str(n) + 'input' + str(j)
                newwt[neu].append(neuron['weights'][j])
            neuron['weights'][-1] += l_rate * neuron['delta']
            neudel = ll + 'neuron' + str(n) + 'delta'
            newwt[neudel].append(neuron['weights'][-1])



import matplotlib.pyplot as plt

seed(1)
sample1 = [[0.05, 0.1, 0.01, 0.99]]
no_inputs = 2
no_outputs = 2
no_iter=200
iteration_list = []
error_list = []
weight1_list=[]
weight2_list=[]

wt={}

#initializing weights with random values between 0 and 1
network = list()
hidden_layer = [{'weights': [np.random.uniform(0.0, 1.0) for i in range(no_inputs + 1)]} for i in range(2)]
network.append(hidden_layer)
output_layer = [{'weights': [np.random.uniform(0.0, 1.0) for i in range(3)]} for i in range(no_outputs)]
network.append(output_layer)
newwt=defaultdict(list)

#training the network
for epoch in range(no_iter):
        error = 0
        for k in sample1:
            outputs = forward_propagation(network, k)
            expected_output = [0 for i in range(no_outputs)]
            expected_output[int(k[-1])] = 1
            error += sum(([0.5*((expected_output[i]-outputs[i])**2) for i in range(len(expected_output))]))
            backward_propagation(network, expected_output)
            learning_rate = 0.5
            update_weights(network, k, learning_rate)
        #Appending the weights of individual parameters to the lists
        print('>Iteration=%d, lrate=%.3f, error=%.3f' % (epoch+1, learning_rate, error))
        iteration_list.append(epoch)
        error_list.append(error)

# plotting the scatter plots
plt.scatter(x=iteration_list, y=error_list)
plt.title("Plot between iterations and total cost")
plt.xlabel("Iterations")
plt.ylabel("Total cost")
plt.show()


# plotting the scatter plots
plt.scatter(x=iteration_list, y=newwt['layer1neuron1delta'])
plt.title("Plot between iterations and parameters")
plt.xlabel("Iterations")
plt.ylabel("theta_10_1")
plt.show()

plt.scatter(x=iteration_list, y=newwt['layer1neuron1input0'])
plt.title("Plot between iterations and parameters")
plt.xlabel("Iterations")
plt.ylabel("theta_11_1")
plt.show()

plt.scatter(x=iteration_list, y=newwt['layer1neuron1input1'])
plt.title("Plot between iterations and parameters")
plt.xlabel("Iterations")
plt.ylabel("theta_12_1")
plt.show()

# plotting the scatter plots
plt.scatter(x=iteration_list, y=newwt['layer1neuron2delta'])
plt.title("Plot between iterations and parameters")
plt.xlabel("Iterations")
plt.ylabel("theta_20_1")
plt.show()

plt.scatter(x=iteration_list, y=newwt['layer1neuron2input0'])
plt.title("Plot between iterations and parameters")
plt.xlabel("Iterations")
plt.ylabel("theta_21_1")
plt.show()

plt.scatter(x=iteration_list, y=newwt['layer1neuron2input1'])
plt.title("Plot between iterations and parameters")
plt.xlabel("Iterations")
plt.ylabel("theta_22_1")
plt.show()



# plotting the scatter plots
plt.scatter(x=iteration_list, y=newwt['layer2neuron1delta'])
plt.title("Plot between iterations and parameters")
plt.xlabel("Iterations")
plt.ylabel("theta_10_2")
plt.show()

plt.scatter(x=iteration_list, y=newwt['layer2neuron1input0'])
plt.title("Plot between iterations and parameters")
plt.xlabel("Iterations")
plt.ylabel("theta_11_2")
plt.show()

plt.scatter(x=iteration_list, y=newwt['layer2neuron1input1'])
plt.title("Plot between iterations and parameters")
plt.xlabel("Iterations")
plt.ylabel("theta_12_2")
plt.show()


# plotting the scatter plots
plt.scatter(x=iteration_list, y=newwt['layer2neuron2delta'])
plt.title("Plot between iterations and parameters")
plt.xlabel("Iterations")
plt.ylabel("theta_20_2")
plt.show()

plt.scatter(x=iteration_list, y=newwt['layer2neuron2input0'])
plt.title("Plot between iterations and parameters")
plt.xlabel("Iterations")
plt.ylabel("theta_21_2")
plt.show()

plt.scatter(x=iteration_list, y=newwt['layer2neuron2input1'])
plt.title("Plot between iterations and parameters")
plt.xlabel("Iterations")
plt.ylabel("theta_22_2")
plt.show()

