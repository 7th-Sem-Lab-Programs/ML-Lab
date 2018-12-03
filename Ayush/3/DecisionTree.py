import pandas as pd
import numpy as np
import math

class Node:
	def __init__(self,l):
		self.label=l
		self.branches={}

def entropy(data):
	tot = len(data)
	pos = len(data.ix[data["PlayTennis"]=="Y"])
	neg = len(data.ix[data["PlayTennis"]=="N"])
	entropy = 0
	if pos > 0:
		entropy = -1*(pos/float(tot))*(math.log(pos,2)-math.log(tot,2))
	if neg > 0:
		entropy += -1*(neg/float(tot))*(math.log(neg,2)-math.log(tot,2))
	return entropy

def gain(s, data, attrib):
	vals = set(data[attrib])
	gain = s
	for val in vals:
		gain -= len(data.ix[data[attrib]==val])/float(len(data))*entropy(data.ix[data[attrib]==val])
	return gain

def get_Attribute(data):
	entropy_s = entropy(data)
	attribute = ""
	max_gain = 0
	for attr in data.columns[:-1]:
		g = gain(entropy_s, data, attr)
		if g > max_gain:
			max_gain = g
			attribute = attr
	return attribute

def get_Tree(data):
	root = Node("NULL")
	if(entropy(data)==0):
		if(len(data.ix[data[data.columns[-1]]=="Y"])==len(data)):
			root.label="Y"
			return root
		else:
			root.label="N"
			return root
	if(len(data.columns)==1):
		return
	else:
		attr = get_Attribute(data)
		root.label = attr
		values = set(data[attr])
		for v in values:
			root.branches[v] = get_Tree(data.ix[data[attr]==v].drop(attr,axis=1))
		return root

def get_Rules(root, rule, rules):
	if not root.branches:
		rules.append(rule[:-2]+" => "+root.label)
		return rules
	for i in root.branches:
		get_Rules(root.branches[i],rule+root.label+"="+i+" ^ ",rules)
	return rules

def test(tree, test_str):
	if not tree.branches:
		return tree.label
	return test(tree.branches[test_str[tree.label]],test_str)

data = pd.read_csv('Data3.csv')
entropy_s = entropy(data)
tree = get_Tree(data)
rules = get_Rules(tree,"",[])
print("Tree is\n")
for r in rules:
	print(r)

test_str={}
print("\nEnter Test Data")
for i in data.columns[:-1]:
	test_str[i] = input(i+" : ")
print("\nOutcome is ")
print(test(tree,test_str))
