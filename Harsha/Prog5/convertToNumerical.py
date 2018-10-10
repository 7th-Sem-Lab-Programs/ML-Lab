import pandas as pd 
data=pd.read_csv('data.csv')
mapping={}
for attr in data.columns:
	for attr,value in zip(set(data[attr]),list(range(len(set(data[attr]))))):
		mapping[attr]=value
print(mapping)
for index in list(data.index):
	for col in data.columns:
		data[col][index]=mapping[data[col][index]]
print(data)
data.to_csv('modified.csv',header=None)