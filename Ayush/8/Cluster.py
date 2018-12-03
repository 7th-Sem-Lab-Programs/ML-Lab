import pandas as pd
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X = pd.DataFrame(iris.data)
X.columns = ['SepalLength','SepalWidth','PetalLength','PetalWidth']
y = pd.DataFrame(iris.target)
y.columns = ['Target']

plt.figure(figsize=(14,14)).subplots_adjust(hspace=0.4, wspace=0.4)
colormap = np.array(['red','lime','black'])

from sklearn.cluster import KMeans
model = KMeans(n_clusters=3)
model.fit(X)
plt.subplot(2,2,1)
plt.scatter(X.PetalLength,X.PetalWidth,c=colormap[y.Target],s=40)
plt.title('Real Cluster')
plt.xlabel('PetalLength')
plt.ylabel('PetalWidth')
plt.subplot(2,2,2)
plt.scatter(X.PetalLength,X.PetalWidth,c=colormap[model.labels_],s=40)
plt.title('K-Means Cluster')
plt.xlabel('PetalLength')
plt.ylabel('PetalWidth')

from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=3)
gmm.fit(X)
gmm_y = gmm.predict(X)
plt.subplot(2,2,3)
plt.scatter(X.PetalLength,X.PetalWidth,c=colormap[y.Target],s=40)
plt.title('Real Cluster')
plt.xlabel('PetalLength')
plt.ylabel('PetalWidth')
plt.subplot(2,2,4)
plt.scatter(X.PetalLength,X.PetalWidth,c=colormap[gmm_y],s=40)
plt.title('K-Means Cluster')
plt.xlabel('PetalLength')
plt.ylabel('PetalWidth')
plt.show()
