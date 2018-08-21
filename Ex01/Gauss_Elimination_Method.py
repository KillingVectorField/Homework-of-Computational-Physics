import math

def GaussEliminate(A,b):
    '''高斯消去，不选主元，省略归一化操作'''
    n=len(A)
    for k in range(n-1):
        if A[k][k]==0:
            print('Fail!')
            break
        for i in range(k+1,n):
            c=-A[i][k]/A[k][k]              #计算倍乘因子
            for j in range(k+1,n):
                A[i][j]+=c*A[k][j]          #更新矩阵元素
            b[i]+=c*b[k]
    return A,b

def GaussEliminatePermutated (A,b):
    n=len(A)
    for k in range(n-1):
        Tmp_column=[abs(A[i][k]) for i in range(k,n)]
        max_i=Tmp_column.index(max(Tmp_column))+k               #找到最大主元在第几行
        if max_i!=k:
            Tmp_row=A[k]                        #交换系数矩阵的当前行与最大主元所在行
            A[k]=A[max_i]
            A[max_i]=Tmp_row
            Tmp_b=b[k]                          #交换b的当前行与最大主元所在行
            b[k]=b[max_i]
            b[max_i]=Tmp_b
        if A[k][k]==0:                      #主元为0，无法继续
            print('Fail!')
            break
        for i in range(k+1,n):
            c=-A[i][k]/A[k][k]              #计算倍乘因子
            for j in range(k+1,n):
                A[i][j]+=c*A[k][j]          #更新矩阵元素
            b[i]+=c*b[k]
    return A,b


def HilbertMatrix (n):
    '''产生n阶希尔伯特矩阵'''
    H=[[1.0/(i+j+1) for j in range(n)]for i in range(n)]
    return H

def BackSubstitutionU(A,b):
    '''系数矩阵为上三角矩阵时的回代求解'''
    n=len(A)
    x=[0]*n
    for i in range (n-1,-1,-1):
        if A[i][i]==0:
            print('A pivot is 0. BackSubstitutionU Failed./n')
            break
        x[i]=b[i]
        for j in range(i+1,n):
            x[i]-=A[i][j]*x[j]
        x[i]/=A[i][i]
    return x

def BackSubstitutionL(A,b):
    '''系数矩阵为下三角矩阵时的回代求解'''
    n=len(A)
    x=[0]*n
    for i in range (n):
        if A[i][i]==0:
            print('A pivot is 0. BackSustitutionL Failed./n')
            break
        x[i]=b[i]
        for j in range(i):
            x[i]-=A[i][j]*x[j]
        x[i]/=A[i][i]
    return x

def transpose(A):
    AT=[[A[i][j] for i in range(len(A))]for j in range(len(A[0]))]
    return AT

def GEM(A,b):
    GaussEliminate(A,b)
    return BackSubstitutionU(A,b)

def GEMPermutated(A,b):         #部分选主元的GEM算法
    GaussEliminatePermutated(A,b)
    return BackSubstitutionU(A,b)

def CholeskyDecomposition(A):
    '''对称正定矩阵的Cholesky分解,分解成下三角矩阵L及其转置的乘积，返回L（其中只有下三角部分有意义）'''
    n=len(A)
    for j in range(n):
        for k in range(j):                                  #利用A[j][j]=Sum k from 1 to j L[j][k]^2，求出L[j][j]
            A[j][j]-=A[j][k]*A[j][k]
        if(A[j][j]>=0):A[j][j]=math.sqrt(A[j][j])
        else:                                               #Hilbert矩阵过于奇异，由于误差导致A[j][j]值变成负
            print("A[j][j]<0!")
            A[j][j]=float('nan')
        for i in range(j+1,n):                              #利用A[i][j]=Sum k from 1 to j L[i][k]*L[j][k]， 求第j列剩余的元素L[i][j]
            for k in range(j):
                A[i][j]-=A[i][k]*A[j][k]
            A[i][j]/=A[j][j]
    return A

'''A=[[0.0000000000000001,2],[3,1]]
b=[2,4]
x=GEM(A,b)
print(x)
A=[[0.0000000000000001,2],[3,1]]
b=[2,4]
x=GEMPermutated(A,b)
print(x)
print(A,b)#A,b的值被修改
B=[[5,-1,-1],[-1,3,-1],[-1,-1,5]]
L=CholeskyDecomposition(B)
print(L)'''

while True:
    print("n=",end='')
    dim=int(input())                        #输入问题的维数
    H=HilbertMatrix(dim)
    b=[1]*dim
    #print('Hilbert Matrix:\n',H)
    L=CholeskyDecomposition(H)
    LT=transpose(L)
    #print('Lower Triangular Matrix (The upper part is meaningless):\n',L)
    #解L.LT.x=b
    x1=BackSubstitutionL(L,b)#求解L.x1=b
    x_CH=BackSubstitutionU(LT,x1)#求解Lt.x=x1
    print('Cholesky Method Solution:\n',x_CH)
    x_GEM=GEM(HilbertMatrix(dim),[1]*dim)
    print('GEM Solution:\n',x_GEM)
    x_GEMP=GEMPermutated(HilbertMatrix(dim),[1]*dim)
    print('GEM(Permutated) Solution:\n',x_GEMP)
    print("Continue?(y or n)")
    if input()=='n':break
    else:print()