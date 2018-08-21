import numpy as np
N=10

A=2*np.eye(N,dtype=float)
for i in range(A.shape[0]):
    A[i-1,i]=-1
    A[(i+1)%N,i]=-1

#print('A=',A,sep='\n')

def Power_Iteration(A,eps):
    q=np.random.random_sample(A.shape[0])        #产生一个随机的N维矢量
    err,tmp_err=100,0
    while(err>eps and abs(tmp_err-err)>eps*1e-2):#精度达到要求或者迭代趋于特征值已经非常慢
        z=A.dot(q)
        tmp=z/np.linalg.norm(z)
        tmp_err=err
        err=np.max(abs(q-tmp))      #用本征矢的无穷模为判据
        q=tmp
    v=q.dot(A.dot(q))
    return(v,q)

result=Power_Iteration(A,10**(-8))
print(result)
