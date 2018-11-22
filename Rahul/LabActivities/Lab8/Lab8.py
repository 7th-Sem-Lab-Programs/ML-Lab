import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture

iris = datasets.load_iris()
X = pd.DataFrame(iris.data)
X.columns = ['Sepal_length','Sepal_width','Petal_length','Petal_width',]
y = pd.DataFrame(iris.target)
y.columns = ['Target']

#KMeans algorithm
model = KMeans(n_clusters=3)
model.fit(X)

#EM algorithm using GMM
gmm = GaussianMixture(n_components=3)
gmm.fit(X)
gmm_y = gmm.predict(X)

plt.figure(figsize = (14,14))
plt.rcParams.update({'font.size': 8})
colormap = np.array(['red','lime','pink'])

plt.subplot(2,2,1)
plt.scatter(X.Petal_length,X.Petal_width,c=colormap[y.Target],s=40)
plt.title("Real clusters")
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.subplot(2,2,2)
plt.scatter(X.Petal_length,X.Petal_width,c=colormap[model.labels_],s=40)
plt.title("KMeans Clusters")
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.subplot(2,2,3)
plt.scatter(X.Petal_length,X.Petal_width,c=colormap[gmm_y],s=40)
plt.title("\nGMM Clusters")
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.show()
print("Iris data= ",iris.data.shape)
print("Actual values= ",iris.target)
print("KMeans predicted values= ",model.labels_)
print("EM predicted values= ",gmm_y)
