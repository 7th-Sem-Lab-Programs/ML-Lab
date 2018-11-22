import random
from math import exp
from random import seed


def init_network(n_inputs,n_hidden,n_outputs):
    network = list()
    hidden_layer = [ {'weights':[random.uniform(-0.5,0.5) for i in range(n_inputs+1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [ {'weights':[random.uniform(-0.5,0.5) for i in range(n_hidden+1)]} for i in range(n_outputs)]
    network.append(output_layer)
    i=1
    print("NN is \n")
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
            #print(activation)
            neuron['output']=transfer(activation)
            #print()
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


import pandas as pd
df = pd.read_csv("Data.csv")

dataset = df.values.tolist()


n_inputs = len(dataset[0])-1
print(n_inputs)
n_output = len(set([row[-1] for row in dataset]))-1
print(n_output)
network = init_network(n_inputs,2,n_output)
print(network)
train_network(network,dataset,0.5,20,n_output)
print("Final")
i =1
for layer in network:
    j =1
    for sub in layer:
        print("\n layer i: ",i," j: ",j," Node: ",sub)
        j = j+1
    i = i+1


'''dataset = [[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]];
'''
