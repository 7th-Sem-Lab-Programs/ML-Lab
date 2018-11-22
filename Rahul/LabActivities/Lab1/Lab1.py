import csv

fr = open('Data1.csv')
csv_r = csv.reader(fr)

fl = next(csv_r)
cols = len(fl)
tar = fl[cols-1]

list1 = ["p"]*(cols-1)
for row in csv_r:
	if row[cols-1] == tar:
		for i in range(len(row)-1):
			if list1[i] == "p":
				list1[i] = row[i]
			elif list1[i] != row[i]:
				list1[i] = "?"
print("The Concept is: ")
print(list1)
flag = 1
for i in range(cols-1):
	temp = raw_input("Enter the Column Value: ")
	if list1[i] != "?":
		if list1[i] != temp:
			flag = 0
if flag == 0:
	print("Not Accepted!!!")
else:
	print ("Accepted!!!")
