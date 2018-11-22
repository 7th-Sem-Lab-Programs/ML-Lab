import csv
 
Data = [["Name", "Class", "Grade"],
          ['Rahul', '6th Class', 'A'],
          ['Ramesh', '7th Class', 'B']]
 
myFile = open('ex.csv', 'w')
with myFile:
    writer = csv.writer(myFile)
    writer.writerows(Data)
     
print("Finished wriitng")

with open('ex.csv') as File: 
    reader = csv.reader(File)
    for row in reader:
        print(row)
