import csv

def s_0(n):
	return ["0"]*n
def g_0(n):
	return ["?"]*n

def more_general(h1, h2):
	n = len(h1)
	l1 = [False]*n
	for i in range(n):
		if h1[i] == "?" or h1[i]==h2[i]:
			l1[i] = True
		elif h1[i]!=h2[i] or h1[i]=="0":
			l1[i] = False
	#print l1
	return all(l1)

def h2acceptsh1(h1, h2):
	return more_general(h1, h2)


def updatespecificbound(s, h):
	for i in s:
		if h2acceptsh1(i, h) == False:
			for j in range(len(h)-1):
				if i[j] == "0":
					i[j] = h[j]
				elif i[j] != h[j]:
					i[j] = "?"
	#print s
	b = list()
	for sublist in s:
    		if sublist not in b:
        		b.append(sublist)
	#print b
	s = b
	#print s
	
	for i in s:
		for j in s:
			if i!=j:
				if more_general(j, i) and j in b:
					b.remove(j)
	s=b
	#print "After removing general from specific", s
	
def specificconsistency(s, g):
	for i in s:
		flag = 0
		for j in g:
			if more_general(j, i):
				flag = 1
				break
		if flag == 0:
			s.remove(i)
	#print s


def replaceshit(ind, s, cur_hyp, g, h):
	for i in s:
		temp = list(cur_hyp)
		temp[ind] = i[ind]
		if not more_general(temp, h):
			g.append(temp)



def updategeneralbound(g, h, s):
	for i in g:
		if h2acceptsh1(i, h) == True:
			for j in range(len(h)-1):
				if i[j] == "?":
					replaceshit(j, s , i, g, h)
			g.remove(i)
	#Removing Duplicates
	b = list()
	for sublist in g:
    		if sublist not in b:
        		b.append(sublist)
	g = b
	#print g
	
	#Keeping most general
	for i in g:
		for j in g:
			if i!=j:
				if more_general(j, i) and i in b:
					b.remove(i)
	g=b
	#print "After removing Specific from General", g
	
def generalconsistency(s, g):
	for i in g:
		flag = 0
		for j in s:
			if more_general(i, j):
				flag = 1
				break
		if flag == 0:
			g.remove(i)
	#print s

def removegeneral(g, h):
	for i in g:
		if not more_general(i, h):
			g.remove(i)

def removespecific(s, h):
	for i in s:
		if more_general(i, h):
			s.remove(i)

def ff():

	fr = open('Data1.csv')
	csv_r = csv.reader(fr)

	fl = next(csv_r)
	cols = len(fl)

	s = list()
	g = list()
	s.append(s_0(cols-1))
	g.append(g_0(cols-1))

	tar = "P"
	for row in csv_r:
		print "\n", row
		if row[cols-1] == tar:
			updatespecificbound(s, row)
			removegeneral(g, row)
			specificconsistency(s, g)
			print "\nCURRENT S: ", s, "\nCurrent G: ", g
			
		else:
			updategeneralbound(g, row, s)
			removespecific(s, row)
			generalconsistency(s, g)
			print "\nCURRENT S: ", s, "\nCurrent G: ", g
	print "\n\n\nFINAL S and G:", s, g

ff()

#updategeneralbound([["?", "?", "?", "?", "?"]], ["USA", "Toyota", "Red", "1980", "Sports"], [["Japan","?","?","?", "Eco"]])
#	print "More General"
#else:
#	print "Less General"
