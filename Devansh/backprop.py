from random import shuffle, random, seed
from csv import reader
from math import exp

# globals
network = []
ds = None

def load_csv(fname):
	global ds

	ds = list(reader(open(fname, 'r')))
	shuffle(ds)

def normalize_dataset():
	global ds

	for i in range(len(ds[0]) - 1):
		for row in ds:
			row[i] = float(row[i].strip())

	for row in ds:
		row[len(ds[0]) - 1] = int(row[len(ds[0]) - 1].strip())

	min_max = [[min(column), max(column)] for column in zip(*ds)]
	for row in ds:
		for i in range(len(row) - 1):
			row[i] = (row[i] - min_max[i][0]) / (min_max[i][1] - min_max[i][0])

def split_dataset(ds, r):
	return ds[:int(len(ds) * r)], ds[int(len(ds) * r) + 1:]

def init_network(n_input, n_hidden, n_output):
	global network

	hidden_layer = {'weights': [[random() for i in range(n_input)] for j in range(n_hidden)], 'bias': random(), 'outputs': [None for i in range(n_hidden)]}
	output_layer = {'weights': [[random() for i in range(n_hidden)] for j in range(n_output)], 'bias': random(), 'outputs': [None for i in range(n_output)]}

	network.append(hidden_layer)
	network.append(output_layer)

def activate(out):
	return 1 / (1 + exp(-out))

def forward_propagate(row, *args):
	global network
	arg = args[0]

	hidden_layer = network[0]
	for i in range(arg[1]):
		out = hidden_layer['bias'] 
		for j in range(arg[0]):
			out += hidden_layer['weights'][i][j] * row[j]
		hidden_layer['outputs'][i] = activate(out)

	output_layer = network[1]
	for i in range(arg[2]):
		out = output_layer['bias']
		for j in range(arg[1]):
			out += output_layer['weights'][i][j] * hidden_layer['outputs'][j]
		output_layer['outputs'][i] = activate(out)

def update_weights(delta_wts, *args):
	global network
	arg = args[0]

	hidden_layer = network[0]
	output_layer = network[1]

	k = 0
	for i in range(arg[1]):
		for j in range(arg[0]):
			hidden_layer['weights'][i][j] += delta_wts[0][k]
			k += 1
	
	k = 0
	for i in range(arg[2]):
		for j in range(arg[1]):
			output_layer['weights'][i][j] += delta_wts[1][k]
			k += 1

def backward_propagate(row, *args):
	global network
	arg = args[0]

	output_layer = network[1]
	hidden_layer = network[0]

	delta_wts = [[],[]]
	delta_out = []
	delta_hid = []

	for i in range(arg[2]):
		delta_out.append(output_layer['outputs'][i] * (1 - output_layer['outputs'][i]) * (row[-1] * output_layer['outputs'][i]))
		for j in range(arg[1]):
			delta_wts[1].append(arg[4] * delta_out[i] * hidden_layer['outputs'][i])
	
	term = 0
	for i in range(arg[1]):
		for j in range(arg[2]):
			term += output_layer['weights'][j][i] * delta_out[j]
		delta_hid.append(hidden_layer['outputs'][i] * (1 - hidden_layer['outputs'][i]) * term)
		for j in range(arg[0]):
			delta_wts[0].append(arg[4] * delta_hid[i] * row[j])

	update_weights(delta_wts, arg)

def train_network(train_data, *args):
	global network

	for epoch in range(args[3]):
		print("Epoch ", epoch)
		for row in train_data:
			forward_propagate(row, args)
			backward_propagate(row, args)

def check_accuracy(outputs):
	global network

	count = 0
	for i, j in outputs:
		if i == j:
			count += 1

	return count / len(outputs)

def test_network(test_data, *args):
	global network

	outputs = []
	for row in test_data:
		hidden_layer = network[0]
		for i in range(args[1]):
			out = hidden_layer['bias'] 
			for j in range(args[0]):
				out += hidden_layer['weights'][i][j] * row[j]
			hidden_layer['outputs'][i] = activate(out)

		output_layer = network[1]
		for i in range(args[2]):
			out = output_layer['bias']
			for j in range(args[1]):
				out += output_layer['weights'][i][j] * hidden_layer['outputs'][j]
			output_layer['outputs'][i] = activate(out)
		print(output_layer['outputs'])
		outputs.append([output_layer['outputs'].index(max(output_layer['outputs'])) + 1, row[-1]])
	print(outputs)

	return check_accuracy(outputs)

# main
seed(1)

load_csv('seeds.csv')

normalize_dataset()

n = input('Enter split ratio : ')
split_ratio = float(n) if float(n) < 1 else 0.75
train_data, test_data = split_dataset(ds, split_ratio)

n_input = len(ds[0]) - 1
n_hidden = 3
n_output = len(set([ds[i][len(ds[0]) - 1] for i in range(len(ds))]))
init_network(n_input, n_hidden, n_output)

l_rate = 0.1
n_epoch = 20
train_network(train_data, n_input, n_hidden, n_output, n_epoch, l_rate)
print(network)

ratio = test_network(test_data, n_input, n_hidden, n_output)
print("Accuracy = ", ratio * 100, "%")
