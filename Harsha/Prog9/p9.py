import sklearn.utils
import numpy as np
from sklearn import datasets
iris = datasets.load_iris()
data = iris.data
target = iris.target

X,Y=sklearn.utils.shuffle(data,target)
X_train = X[:100]
X_test = X[100:]
Y_train = Y[:100]
Y_test = Y[100:]

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
knn.fit(X_train,Y_train)

correct=0
correct_pred=[]
incorrect_pred=[]
for x,y in zip(X_test,Y_test):	
	predicted=knn.predict([x])
	if predicted == y:
		correct_pred.append(x)
		correct += 1
	else:
		incorrect_pred.append(x)
print("Accuracy: ",(correct/float(len(X_test)))*100,"%")
print("Correctly Predicted instances:\n",correct_pred)
print("Incorrectly Predicted instanes:\n",incorrect_pred)

