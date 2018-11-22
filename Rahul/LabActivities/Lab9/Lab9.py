from sklearn import neighbors, datasets
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

#Load Dataset
iris = datasets.load_iris()
X = iris.data  
y = iris.target
print(X.shape)
# Splitting the data 80:20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
print("Training dataset size: ", X_train.shape)
print("Testing dataset size: ", X_test.shape)

# Initializing the classifier
classifier = neighbors.KNeighborsClassifier(n_neighbors=3)

# Fitting the model
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Computing the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix is: ")
print(cm)

# Computing The Accuracy Score rounded to 2
accuracy = accuracy_score(y_test, y_pred)*100
print(('Accuracy of our model is equal to ' + str(round(accuracy, 2)) + ' %.'))

# Printing what is expected an what is predicted
for i in range(len(y_test)):
	print("Expected: ", y_test[i], " Got: ", y_pred[i])

# Finding the Failure Percentage
cnt = 0
for i in range(len(y_test)):
	if(y_test[i] != y_pred[i]):
		cnt = cnt+1

#Printing number of mis-classified instances
print("Number of mis-classified are: ", cnt)
