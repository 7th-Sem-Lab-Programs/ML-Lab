import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
iris=datasets.load_iris()
X= pd.DataFrame(iris.data) #column and rows, dict like container
X.columns= ['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width']
y=pd.DataFrame(iris.target) #target-class labels
y.columns=['Targets']
model=KMeans(n_clusters=3)
model.fit(X)
plt.figure(figsize=(14,14)).subplots_adjust(hspace=0.4, wspace=0.4)
colormap=np.array(['red','lime','black'])
plt.subplot(2,2,1)
plt.scatter(X.Petal_Length,X.Petal_Width, c=colormap[y.Targets],s=40)
plt.title('Real Cluster')
plt.xlabel('Petal_Length')
plt.ylabel('Petal_Width')
plt.subplot(2,2,2)
plt.scatter(X.Petal_Length,X.Petal_Width, c=colormap[model.labels_],s=40)
plt.title('KMeans Cluster')
plt.xlabel('Petal_Length')
plt.ylabel('Petal_Width')

from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=3)
gmm.fit(X)
gmm_y=gmm.predict(X)

plt.subplot(2,2,3)
plt.scatter(X.Petal_Length,X.Petal_Width, c=colormap[y.Targets],s=40)
plt.title('\nReal Cluster')
plt.xlabel('Petal_Length')
plt.ylabel('Petal_Width')
plt.subplot(2,2,4)
plt.scatter(X.Petal_Length,X.Petal_Width, c=colormap[gmm_y],s=40)
plt.title('\nGMM Cluster')
plt.xlabel('Petal_Length')
plt.ylabel('Petal_Width')
plt.show()
