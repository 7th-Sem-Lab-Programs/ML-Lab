from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import load_boston
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import numpy

boston = load_boston()
X = boston.data  
y = boston.target

y_pred_list = list()
corr_list = list()
for i in range(10):
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
	neigh = KNeighborsRegressor(n_neighbors=i+1)
	neigh.fit(X_train, y_train) 

	y_pred = neigh.predict(X_test)
	y_pred_list.append(y_pred)
	corr_list.append(numpy.corrcoef(y_test,y_pred)[0][1])

cnt = 0
for i in corr_list:
	if(max(corr_list) == i):
		break
	cnt = cnt +1
plt.title("Y_TEST vs Y_PRED 45 degree expected")
plt.scatter(y_test, y_pred_list[cnt],c ='r')
plt.show()

