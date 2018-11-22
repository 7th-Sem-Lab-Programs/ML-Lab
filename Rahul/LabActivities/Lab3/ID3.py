import pandas as pd
import math
import collections


df = pd.DataFrame(pd.read_csv("DataID3.csv"))

tar = df.columns[-1]
def entropy(probs):
	return sum(-prob*math.log(prob,2) for prob in probs)
	
def ent_of_list(a_list):
	from collections import Counter
	cnt = Counter(x for x in a_list)
	#print(cnt)
	num_inst=len(a_list)*1.0
	probs=[x/num_inst for x in cnt.values()]
	return entropy(probs)

def info_gain(df, split_attrname,tar_attr):
	df_split = df.groupby(split_attrname)
	df_ent = ent_of_list(df[tar])
	a = list()
	for name,group in df_split:
		a.append(ent_of_list(group[tar])*len(group)/len(df))	
	nobs = len(df.index)*1.0
	x=0.0
	for i in a:
		x+=i
	#print "Information gain of attr ",split_attrname," is ",(df_ent-x)
	return (df_ent-x)
	
def id3(df,tar_attrname,attrnames,def_class,tree):
	from collections import Counter
	cnt = Counter(x for x in df[tar_attrname])
	if(len(cnt)==1):
		#print max(cnt.keys())
		#tree[name1] = max(cnt.keys())
		return max(cnt.keys())
	elif len(df)==0 or not(attrnames):
		#print def_class
		#tree[name1] = def_class	
		return def_class
	def_class=max(cnt.keys())
	gainz = list()
	for attr in attrnames:
		gainz.append(info_gain(df,attr,tar_attrname))
	#print(gainz)
	index_of_max= gainz.index(max(gainz))	
	#print "Splitting attribute is ",attrnames[index_of_max]
	
	temp = {}
	df_split = df.groupby(attrnames[index_of_max])
	tree[attrnames[index_of_max]] = dict()
	p  = attrnames[index_of_max]
	#attrnames.remove(attrnames[index_of_max])
	for name,group in df_split:
		#print "For ",name 
		temp[name] = {}
		b = id3(group,tar_attrname,attrnames,def_class,temp[name])
		if not isinstance(b,dict):
			temp[name] = b
	tree[attrnames[index_of_max]]=temp
	return temp

attr = list(df.columns[:-1].values)
if 1 in df[tar]:
	default = 1
else: 
	default = 'Y' 

tree = {}	
id3(df,tar,attr,default,tree)	
print tree

example = {}
for at in attr:
	example[at]=(raw_input("Enter "+at+" value: "))
#print example

def fun(tree):
	#print tree
	if isinstance(tree,dict):
		if(len(tree.keys()) == 1):
			for k,v in tree.iteritems():
				#if isinstance(v,dict):
				val = v[example[k]]
				fun(val)
				#else:
				#	print v	
				
		else:
			pass
	else:
		print tree
fun(tree)
