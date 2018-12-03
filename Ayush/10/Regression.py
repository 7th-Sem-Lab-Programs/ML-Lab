import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def kernel(point, xmat, k):
	m,n = np.shape(xmat)
	weights = np.mat(np.eye(m))
	for j in range(m):
		diff = point - X[j]
		weights[j,j] = np.exp(diff*diff.T/(-2.0*k**2))
	return weights

def localWeight(point,xmat,ymat,k):
	wei = kernel(point,xmat,k)
	W = (X.T*(wei*X)).I*(X.T*(wei*ymat.T))
	return W

def lWR(xmat, ymat, k):
	m,n = np.shape(xmat)
	ypred = np.zeros(m)
	for i in range(m):
		ypred[i] = xmat[i]*localWeight(xmat[i],xmat,ymat,k)
	return ypred

def plotting(X,ypred):
	sortindex = X[:,1].argsort(0)
	xsort = X[sortindex][:,0]
	fig = plt.figure().add_subplot(1,1,1)
	fig.scatter(bill, tip, color='green')
	fig.plot(xsort[:,1],ypred[sortindex],color='red', linewidth=2)
	plt.xlabel('Total Bill')
	plt.ylabel('Tip')
	plt.show()	

data = pd.read_csv('Data10.csv')
bill = np.array(data.total_bill)
tip = np.array(data.tip)

mbill = np.mat(bill)
mtip = np.mat(tip)
m = np.shape(mbill)[1]
ones = np.mat(np.ones(m))
X = np.hstack((ones.T, mbill.T))
ypred = lWR(X,mtip,0.5)
plotting(X,ypred)
