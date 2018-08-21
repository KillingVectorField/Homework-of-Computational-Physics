import numpy as np

N=int(input('N='))#矩阵A的规模

if input('Need Plot? (y or n)')=='y':#是否需要作图？
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    needplot=True
else: needplot=False

def Power_Iteration(A,eps):#用于求J的谱半径
    '''幂次法求矩阵模最大的本征值'''
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

h=1/(N+1)
#X1=np.array([h*i for i in range(1,N+1)]),X2=np.array([h*i for i in range(1,N+1)])
lattice=np.array([[[x1,x2] for x2 in range(1,N+1)] for x1 in range(1,N+1)])#整数格点
#lattice.resize((N**2,2))#把N*N个格点展成1维向量（N^2个），按如下顺序排列：(1,1),(1,2),...,(1,N),(2,1),...
def source(X):
    '''Poisson方程的源'''
    return 2*(np.pi**2)*np.sin(np.pi*X[0])*np.sin(np.pi*X[1])
#矢量b表示了所有已知的量的组合(包括已知的电荷密度和边界的信息等等),N^2维向量
b=h**2*np.array([source(h*X) for X in lattice.reshape(N**2,2)])
#构造A，O(N^2)
A=np.zeros((N,N,N,N))
for X1 in range(N):
    for X2 in range(N):
        A[X1,X2,X1,X2]=4
        if X1>0:
            A[X1,X2,X1-1,X2]=-1
        if X1<N-1:
            A[X1,X2,X1+1,X2]=-1
        if X2>0:
            A[X1,X2,X1,X2-1]=-1
        if X2<N-1:
            A[X1,X2,X1,X2+1]=-1
A.resize((N**2,N**2))#将A调整回N^2*N^2

#求Jacobi算法中J的谱半径
'''
rho_J=np.max(abs(1-np.linalg.eigvals(A)/4))#用标准库函数求本征值
print(rho_J,end-start)#Jacobi方法中J的谱半径
'''
rho_J=Power_Iteration(np.eye(A.shape[0])-A/4,1e-4)[0]#用第二小题的幂次法近似求J的谱半径
print('J的谱半径：',rho_J)#Jacobi方法中J的谱半径

def Exact_Solution(X):
    return np.sin(np.pi*X[0])*np.sin(np.pi*X[1])

def Jacobi_Method(A,b):
    '''Jacobi迭代法'''
    #B=D=np.diag(np.diag(A))#B^(-1)就是4*Identity
    #B_inv=np.diag(1/np.diag(A))#B^(-1)就是1/4*Identity
    #B_inv_A=B.dot(A)#就是1/4 * A
    #B_inv_b=B_inv.dot(b)#就是1/4 * b
    z=np.zeros(len(b))
    time=0
    temp=0
    while max(abs(temp-b))/(h**2)>=1e-4:
        z+=-temp/4+b/4
        temp=A.dot(z)
        time+=1
    print('Iterating times of Jacobi Method:',time,'残差 r=',max(abs(temp-b))/(h**2))
    return z

def Gauss_Seidel_Method(A,b):
    '''Gauss_Seidel 迭代法'''
    B=np.copy(A)
    for i in range(B.shape[0]):
        B[i,i+1:]=np.zeros_like(B[i,i+1:])#B是A的下三角部分
    B_inv=np.linalg.solve(B,np.identity(len(b)))#由于B是下三角矩阵，直接回代求B的逆
    #B_inv_A=B_inv.dot(A)
    #B_inv_b=B_inv.dot(b)
    z=np.zeros(len(b))
    time=0
    temp=b
    while max(abs(temp))/(h**2)>=1e-4:
        z+=B_inv.dot(temp)
        temp=-A.dot(z)+b
        time+=1
    print('Iterating times of Jacobi Method:',time,'残差 r=',max(abs(temp))/(h**2))
    return z

def OverRelaxation_Method(A,b,rho_J):
    '''弛豫算法，rho_J为J的谱半径'''
    omega=2/(1+np.sqrt(1-rho_J**2))#（10.46）式
    #D就是4*I
    #B=4*np.eye(A.shape[0])/omega-E，-E为A的下三角部分
    B=np.copy(A)#先构造-E部分
    for i in range(B.shape[0]):
        B[i,i:]=np.zeros_like(B[i,i:])#-E是A的下三角部分(不包含对角元）
    B+=4*np.eye(A.shape[0])/omega
    B_inv=np.linalg.inv(B)
    #B_inv_A=B_inv.dot(A)
    #B_inv_b=B_inv.dot(b)
    z=np.zeros(len(b))
    time=0
    temp=b
    while max(abs(temp))/(h**2)>=1e-4:
        z+=-B_inv.dot(temp)
        temp=A.dot(z)-b
        time+=1
    print('Iterating times of Overrelaxation Method:',time,'残差 r=',max(abs(temp))/(h**2))
    return z

def Plot3D_Exact_Solution():
    '''准确解的函数图'''
    fig = plt.figure()
    ax = Axes3D(fig)
    X1 = np.linspace(0, 1, N+2)
    X2 = np.linspace(0, 1, N+2)
    X1, X2 = np.meshgrid(X1, X2)
    U_Exact=Exact_Solution([X1,X2])
    ax.plot_surface(X1, X2, U_Exact, rstride=1, cstride=1, cmap='rainbow')
    plt.title('Exact Solution')
    plt.show()

def Plot3D_My_Solution(u,title):
    fig = plt.figure()
    ax = Axes3D(fig)
    X1 = np.linspace(0, 1, N+2)
    X2 = np.linspace(0, 1, N+2)
    X1, X2 = np.meshgrid(X1, X2)
    U=np.zeros((N+2,N+2))
    U[1:N+1,1:N+1]=u.reshape(N,N)
    ax.plot_surface(X1, X2, U, rstride=1, cstride=1, cmap='rainbow')
    plt.title(title)
    plt.show()

Exact_u=np.array([Exact_Solution(h*X) for X in lattice.reshape(N**2,2)])

def test(Method,plot):
    if Method=='Jacobi':u=Jacobi_Method(A,b)
    elif Method=='Gauss-Seidel':u=Gauss_Seidel_Method(A,b)
    elif Method=='Overrelaxation':u=OverRelaxation_Method(A,b,rho_J)
    elif Method=='direct-solve':u=np.linalg.solve(A,b)#直接解线性方程组
    if plot:
        Plot3D_My_Solution(u,Method)#做近似解图
        Plot3D_My_Solution(u-Exact_u,'error of %s'%Method)#作图显示误差的分布

if needplot:
    Plot3D_Exact_Solution()#精确解
test('Jacobi',needplot)
test('Gauss-Seidel',needplot)
test('Overrelaxation',needplot)