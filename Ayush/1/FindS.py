import csv
hypo = ['%','%','%','%','%']

with open("Data1.csv") as fl:
	contents = csv.reader(fl, delimiter=',');
	print("The Training Examples are")
	data=[]
	for x in contents:
		print(x)
		if x[-1].upper() == "Y":
			data.append(x)
print("\nPositive Examples are ")
for x in data:
	print(x)
print("\nSteps are")
for x in data:
	for l in range(len(x)-1):
		if hypo[l] == '%':
			hypo[l]=x[l]
		elif hypo[l] != x[l]:
			hypo[l]='?'
	print(hypo)
print("\nFinal hypothesis")
print(hypo)
