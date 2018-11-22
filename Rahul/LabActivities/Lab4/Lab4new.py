import random
from math import exp
from random import seed
import numpy as np

def init_network(n_inputs,n_hidden,n_outputs):
    network = list()
    hidden_layer = [ {'weights':[np.random.randn(1)[0] for i in range(n_inputs+1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [ {'weights':[np.random.randn(1)[0] for i in range(n_hidden+1)]} for i in range(n_outputs)]
    network.append(output_layer)
    i=1
    print("Neural Network is \n")
    for layer in network:
        j=1
        for sub in layer:
            print("\n i: ",i,"j: ",j,"subis: ",sub,"\n")
            j = j+1
        i = i+1
    return network


def activate(weights,inputs):
    activation = weights[-1]
    for i in range(len(inputs)):
        activation += inputs[i]*weights[i]
    return activation


def transfer(activation):
    return 1.0/(1.0+exp(-activation))

def forward(network,row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'],inputs)
            neuron['output']=transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs


def transfer_der(output):
    return output*(1-output)

def backward_prop(network,expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        
        if i != len(network)-1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i+1]:
                    error += (neuron['weights'][j]*neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j]-neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j]*transfer_der(neuron['output'])


def update_weights(network,row,l_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i-1]]
            for neuron in network[i]:
                for j in range(len(inputs)):
                    neuron['weights'][j] += l_rate*neuron['delta']*inputs[j]
                neuron['weights'][-1] += l_rate* neuron['delta']



def train_network(network,train,l_rate,n_epoch,n_output):
    print("Begin training\n")
    for epoch in range(n_epoch):
        sum_error = 0.0
        for row in train:
            outputs = forward(network,row)
            expected = [0 for i in range(n_output)]
            expected[int(row[-1])]=1
            sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
            backward_prop(network,expected)
            update_weights(network,row,l_rate)
        print("epoch: ",epoch,"error: ",sum_error,"\n")

def predict(network, row):
	outputs = forward(network, row)
	return outputs.index(max(outputs))


import pandas as pd
df = pd.read_csv("wData.csv")
dataset = df.values.tolist()

n_inputs = len(dataset[0])-1
n_output = len(set([row[-1] for row in dataset]))

network = init_network(n_inputs,2,n_output)
print("Initial weights:")
print(network)

train_network(network,dataset,0.5,20,n_output)
print("Final Network")
i =1
for layer in network:
    j =1
    for sub in layer:
        print("\n layer i: ",i," j: ",j," Node: ",sub)
        j = j+1
    i = i+1
total =0.0;
count =0.0;
for row in dataset:
	prediction = predict(network,row)
	total=total+1
	print("expected=%d, got=%d"%(row[-1],prediction))
	if(row[-1]==prediction):
		count=count+1;
print("accuracy: "+str((count/total)*100))
