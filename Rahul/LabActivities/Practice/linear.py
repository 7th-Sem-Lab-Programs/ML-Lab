list1 = [1,2,3,4,5,6,7]
print "List is: ", list1
key=(int)(input("Enter Key is:"))
for i in range(len(list1)):
	if(list1[i] == key):
		print "Found ",key,"in pos: ",(i+1)
	
