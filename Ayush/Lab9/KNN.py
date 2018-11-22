from sklearn import datasets
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

cat = ["Iris setosa", "Iris versicolor", "Iris virginica"]

iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target

target_names = np.unique(iris_y)

np.random.seed(0)
indices = np.random.permutation(len(iris_X))

iris_X_train = iris_X[indices[:-20]]
iris_y_train = iris_y[indices[:-20]]
iris_X_test = iris_X[indices[-20:]]
iris_y_test = iris_y[indices[-20:]]

neigh = KNeighborsClassifier(n_neighbors=3)

neigh.fit(iris_X_train, iris_y_train)

predicted = neigh.predict(iris_X_test)
actual = iris_y_test

print("Predicted classes: ")
print(predicted)
print("Actual classes:")
print(actual)

print(classification_report(actual, predicted, target_names=None))
print("Confusion matrix is: \n")
print(confusion_matrix(actual, predicted))

print("\n\nAccuracy is: ")
print(accuracy_score(actual, predicted))
