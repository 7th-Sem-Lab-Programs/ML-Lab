mat1 = [[1,2,3],[4,5,6],[7,8,9]]
mat2 = [[1,0,0],[0,1,0],[0,0,1]]

mat3 = [[-1]*len(mat1)]*len(mat2[0])
def matmul():
	for i in range(len(mat1)):
		for j in range(len(mat2[0])):
			mat3[i][j] = 0
			for k in range(len(mat2)):
				mat3[i][j]+=mat1[i][k]*mat2[k][j]	

print mat1,mat2, mat3
matmul()
print "After Mul: "
print mat3
