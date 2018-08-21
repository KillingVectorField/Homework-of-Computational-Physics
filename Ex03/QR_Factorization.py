import numpy as np
import time
from scipy import linalg

def Householder_Reflection(x,i=0,re='vector'):
    '''
    用Householder反射消去向量x中第i个分量。注意输入向量不能为0！
    第一个返回量为反射后的向量的第i个分量（其余分量为0），
    若选择re='vector'，返回w，使得Householder变换矩阵H=I-2ww^T
    或选择re='matrix',返回Householder变换矩阵
    '''
    temp=x[i+1:].dot(x[i+1:])
    sigma=np.copysign(np.sqrt(temp+x[i]**2),x[i])
    w=np.copy(x)
    w[i]+=sigma                                     #w是反射面的法向向量
    w=w/np.sqrt(temp+w[i]**2)                           #归一化
    if re=='vector':return(-sigma,w)
    elif re=='matrix':return (-sigma,np.eye(len(x))-2*w[:,np.newaxis]*w) #H=I-2ww^T
    elif re=='both':return(-sigma,w,np.eye(len(x))-2*w[:,np.newaxis]*w)

def Householder_Decomposition(A):
    '''
    Householder变换做QR分解
    '''
    Q=np.eye(A.shape[0])#向量乘法，正着乘
    for i in range(A.shape[1]):
        if list(A[i+1:,i])!=[0]*(len(A)-i-1):            #第i列对角线下方并非全部为0
            (A[i,i],w)=Householder_Reflection(A[i:,i],re='vector')
            A[i:,i+1:]-=2*np.outer(w,w.dot(A[i:,i+1:]))
            Q[i:]-=2*np.outer(w,w.dot(Q[i:]))                              
    for i in range(A.shape[0]):#将下三角部分清零
        A[i,:i]=np.zeros_like(A[i,:i])
    return (Q.transpose(),A)
    '''
    #向量乘法，倒着乘
    H=[]                                         #用于记录householder变换向量
    for i in range(len(A[0])):
        if list(A[i+1:,i])!=[0]*(len(A)-i-1):            #第i列对角线下方并非全部为0
            (A[i,i],w)=Householder_Reflection(A[i:,i])                
            A[i:,i+1:]-=2*np.outer(w,w.dot(A[i:,i+1:]))      #对剩余各列做变换
            H.append(w)
    #以下求出矩阵Q
    H_old=np.array([])
    for w in H[::-1]:           #从后往前（从小到大）乘出H
        H_new=np.eye(len(w))
        H_new[H_new.shape[0]-H_old.shape[0]:,H_new.shape[0]-H_old.shape[0]:]=H_old
        H_new-=2*np.outer(w,w.dot(H_new))
        H_old=H_new
    if H_old.shape[0]<A.shape[0]:               #最后H_old维数不够，用单位矩阵补上
        H_new=np.eye(A.shape[0])
        H_new[H_new.shape[0]-H_old.shape[0]:,H_new.shape[0]-H_old.shape[0]:]=H_old
        H_old=H_new                         
    for i in range(A.shape[0]):#将下三角部分清零
        A[i,:i]=np.zeros_like(A[i,:i])
    return (H_old,A)
    '''

def Givens_Rotation(x1,x2):
    '''
    对二维向量Givens旋转变换，可以消去第2个分量
    返回alpha，c和s，G=[[c,s],[-s,c]]
    '''
    if x2==0:
        return (x1,1,0)            #第二个分量已经为0
    else:
        alpha=np.sqrt(x1**2+x2**2)
        return (alpha,x1/alpha,x2/alpha)

def Givens_Decomposition(A):
    '''
    用Givens旋转变换做QR分解
    '''
    Q=np.eye(A.shape[0])
    for i in range (A.shape[1]):
        #if list(A[i+1:,i])!=[0]*(len(A)-i-1):               #如果第i列对角线下方并非全部为0
        for j in range(i+1,A.shape[0]):#对第i列下方每个元素进行givens旋转变换
            (A[i,i],c,s)=Givens_Rotation(A[i,i],A[j,i])
            G=np.array([[c,s],[-s,c]])
            Q[i],Q[j]=G.dot(np.array([Q[i],Q[j]]))
            A[i,i+1:],A[j,i+1:]=G.dot(np.array([A[i,i+1:],A[j,i+1:]]))
    for i in range(A.shape[0]):#将下三角部分清零
        A[i,:i]=np.zeros_like(A[i,:i])
    return (Q.transpose(),A)

def test_QR(testMatrix,needScipy=True):
    '''同时检验两种（以及标准库）方法的结果是否正确'''
    print('Test for correctness:')
    print('test matrix:',testMatrix,sep='\n')
    if needScipy:#用标准的scipy中的qr分解做参照
        Result=linalg.qr(np.copy(testMatrix))
        print("QR Decomposition By Standard Scipy.linalg:",'Q:',Result[0],'R:',Result[1],'Check QR:',Result[0].dot(Result[1]),"Check Q'Q:",np.transpose(Result[0]).dot(Result[0]),sep='\n')
    Result=Householder_Decomposition(np.copy(testMatrix))
    print("QR By Householder Reflection:",'Q:',Result[0],'R:',Result[1],'Check QR:',Result[0].dot(Result[1]),"Check Q'Q:",np.transpose(Result[0]).dot(Result[0]),sep='\n')
    Result=Givens_Decomposition(np.copy(testMatrix))
    print("QR By Givens Rotation:",'Q:',Result[0],'R:',Result[1],'Check QR:',Result[0].dot(Result[1]),"Check Q'Q:",np.transpose(Result[0]).dot(Result[0]),sep='\n')
    return

num_of_matrices=int(input('Number of matrices:'))
order_of_matrix=int(input('Order of matrix:'))
Samples=[]
for i in range(num_of_matrices):
    Samples.append(2*np.random.random_sample((order_of_matrix,order_of_matrix))-1)  #产生(0,1)上的随机数，通过2*X-1，使矩阵元在(-1,1)上Uniformly distributed
linalg.qr(np.ones((2,2)))#预调用一下，以准确试验标准库所需的运算时间
time_start=time.clock()
for sample in Samples:
    linalg.qr(np.copy(sample))
time_scipy=time.clock()
for sample in Samples:
    Householder_Decomposition(np.copy(sample))
time_Householder=time.clock()
for sample in Samples:
    Givens_Decomposition(np.copy(sample))
time_Givens=time.clock()
print('Time consumed for ',num_of_matrices,' ',order_of_matrix ,'-order matrices:',sep='')
print('Standard Scipy:',time_scipy-time_start)
print('Householder:',time_Householder-time_scipy)
print('Givens:',time_Givens-time_Householder)
test_QR(Samples[0],needScipy=True)#试验一下算法的正确性

#总结：在阶数为6时，两种方法几乎是差不多的；当阶数n变大时，Householder方法明显占优。