from numpy import * 
import operator
from os import listdir 
import matplotlib
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np1 
import numpy.linalg as np
from scipy.stats.stats import pearsonr


def kernel(point,xmat, k): 
	m,n = np1.shape(xmat)
	weights = np1.mat(np1.eye((m))) 
	for j in range(m):
		diff = point -X[j]
		weights[j,j] = np1.exp(diff*diff.T/(-2.0*k**2)) 
	return weights

def localWeight(point,xmat,ymat,k): 
	wei = kernel(point,xmat,k)
	W=(X.T*(wei*X)).I*(X.T*(wei*ymat.T)) 
	return W

def localWeightRegression(xmat,ymat,k): 

	m,n = np1.shape(xmat)
	ypred = np1.zeros(m)
	print(xmat[0]) 
	print(xmat[0]*[[1],[1]])
	for i in range(m):
		ypred[i] = xmat[i]*localWeight(xmat[i],xmat,ymat,k) 
	return ypred

# load data points
data = pd.read_csv('data10.csv') 
bill = np1.array(data.total_bill) 
tip = np1.array(data.tip)

#preparing and add 1 in bill 
mbill = np1.mat(bill)
mtip = np1.mat(tip)

m= np1.shape(mbill)[1]
print(np1.shape(mbill))
one = np1.mat(np1.ones(m))



X= np1.hstack((one.T,mbill.T))

#set k here
ypred = localWeightRegression(X,mtip,2)

SortIndex = X[:,1].argsort(0) 
xsort = X[SortIndex][:,0]

print(tip.shape)
print(ypred)
for x in range(tip.shape[0]):
	plt.plot(bill[x],ypred[x],'ro-')
for x in range(tip.shape[0]):
	plt.plot(bill[x],tip[x],'b.')

plt.show()
