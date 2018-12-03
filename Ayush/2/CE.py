import csv

def predict(data, model):
	flag=True
	for i, ele in enumerate(model):
		if ele!='?' and ele!=data[i]:
			flag=False
			break
	return flag

def train():
	S=['%','%','%','%','%','%']
	G=[['?','?','?','?','?','?']]
	print("S0: ",S)
	print("G0: ",G)
	
	with open("Data2.csv") as fl:
		data = csv.reader(fl, delimiter=',')
		for i, row in enumerate(data):
			#+ve Examples
			if row[-1].upper() == "P":
				for j, col in enumerate(row[:-1]):
					if S[j] == '%':
						S[j] = col
					elif S[j] != col:
						S[j] = '?'
				n=len(G)
				m=0
				while m<n:
					if predict(row[:-1],G[m])==False:
						G=G[:m]+G[m+1:]
						n=n-1
					m=m+1
			#-ve Examples
			else:
				for m in range(len(G)):
					if(predict(row[:-1],G[m])==True):
						toS = G[m]
						G=G[:m]+G[m+1:]
						
						for j, col in enumerate(toS):
							if S[j]!='?' and S[j]!=col and S[j]!=row[j]:
								temp=toS[:]
								temp[j]=S[j]
								G+= [temp]
			print("S",i+1,": ",S)
			print("G",i+1,": ",G)
						

train()
