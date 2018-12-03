import numpy as np
from sklearn import datasets
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
X = iris.data
y = iris.target

np.random.seed(0)
indices = np.random.permutation(len(X))

X_train = X[indices[:-20]]
X_test = X[indices[-20:]]
y_train = y[indices[:-20]]
y_test = y[indices[-20:]]

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train,y_train)
predicted = neigh.predict(X_test)
actual = y_test

print("Predicted classes: ")
print(predicted)
print("Actual classes:")
print(actual)

print(classification_report(actual, predicted))
print("Confusion matrix is: \n")
print(confusion_matrix(actual, predicted))

print("\n\nAccuracy is: ")
print(accuracy_score(actual, predicted))
