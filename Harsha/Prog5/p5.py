import pandas as pd
from random import sample
import numpy as np

def splitDatasetNumpy(data,splitRatio):
	index = np.array(sample(range(len(data)),int(splitRatio*len(data))))
	train = data.ix[index]
	test = data.drop(index)
	return train,test

#Split the Data into test and train data based on the split ratio
def splitDataset(data,splitRatio):
	train = data.sample(frac = splitRatio)
	test = data.drop(train.index)
	return train,test

#Function to find the prior probablities
def find_prior(data):
	classes = set(data[data.columns[-1]])
	prior = {}
	for cls in classes:
		prior[cls] = len(data.ix[data[data.columns[-1]] == cls])/len(data)
	return prior

#Function to generate the frequency table for the given data
def generate_freq_table(data,attibuteValues):
	freq_table = pd.DataFrame(columns = ['Probablity'],index = set(data.columns[:-1]))

	for column in data.columns[:-1]:
		temp = pd.DataFrame(columns = set(data[data.columns[-1]]),index=attibuteValues[column])
		for val in attibuteValues[column]:
			d = data.ix[data[column] == val]
			for cls in temp.columns:
				temp[cls][val] = len(d.ix[d[data.columns[-1]] == cls])/len(data.ix[data[data.columns[-1]] == cls])
		freq_table['Probablity'][column] = temp
	return freq_table

#Function to predict the class labels
def predict(freq_table,test,cols,classes):
	p_t_given_cls = {}
	for cls in classes:
		val = 1
		i=0
		for attr in cols:
			temp = freq_table['Probablity'][attr]
			val *= temp[cls][test[i]]
			i += 1
		p_t_given_cls[cls] = val
	p_t = 0
	for cls in classes:
		p_t += (p_t_given_cls[cls]*prior[cls])

	p_cls_given_t = {}
	for cls in classes:
		if(p_t == 0):
			p_cls_given_t[cls]=0
		else:
			p_cls_given_t[cls] = p_t_given_cls[cls]*prior[cls]/p_t
	return max(p_cls_given_t,key = p_cls_given_t.get)

#Function to determine the accuracy of the trained model
def det_accuracy(freq_table,data,cols):
	actual_classes = list(data[data.columns[-1]])
	predicted_classes = []
	testData = []
	for tupl in data.itertuples():
		testData.append(tupl[1:])
	for tupl in testData:
		predicted_classes.append(predict(freq_table,tupl[:-1],cols,set(actual_classes)))
	correct = 0
	# print(actual_classes)
	# print(predicted_classes)
	for i in range(len(actual_classes)):
		if actual_classes[i] == predicted_classes[i]:
			correct +=1
	print("Correctly predicted instances: ",correct)
	print("Accuracy of the model: ",round(correct/float(len(predicted_classes))*100,2),"%")
	

data = pd.read_csv('data.csv')
print("Dataset Loaded!!!\n")
classes = set(data[data.columns[-1]])
attibuteValues = {}
for col in data.columns[:-1]:
	values = set(data[col])
	attibuteValues[col] = values
splitRatio = float(input("Enter the split ratio: "))
# splitRatio=0.50
train, test = splitDataset(data,splitRatio)
print(len(train)," instances for Training")
print(len(test)," instances for Testing")
prior = find_prior(train)
freq_table = generate_freq_table(train,attibuteValues)
#freq_table.to_csv('pothole_pred.csv')
print("\nTraining Done!!\nTesting...")
det_accuracy(freq_table,test,data.columns[:-1])
