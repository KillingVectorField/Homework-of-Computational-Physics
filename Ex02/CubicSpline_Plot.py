import numpy as np
import math
import matplotlib.pyplot as plt

def TripleDiagnalMatrix(a,b,c,f):#a从2开始，b,c,f从1开始，
    '''三对角矩阵的求解（追赶法）'''
    n=len(f)
    m=a[:]#m下标从2开始
    x=[0]*n#存放解
    for i in range (n-1):
        m[i]/=b[i]
        b[i+1]-=m[i]*c[i]
        f[i+1]-=m[i]*f[i]
    x[n-1]=f[n-1]/b[n-1]
    for i in range(n-2,-1,-1):
        x[i]=(f[i]-c[i]*x[i+1])/b[i]
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

def CubicSpline(x,y,f2a=0,f2b=0,f1a=0,f1b=0):
    '''
    三次样条函数内插，可补充两端的一阶导或二阶导信息，默认采用二阶导为零的条件（自然边界条件）
    '''
    n=len(x)-1                              #x0,x1,...,xn，共len=n+1个，n=len-1
    h=[(x[i+1]-x[i]) for i in range(n)]       #h从0到n-1，共n个
    if (f1a==0 and f1b==0):                 #采用2阶导数边界条件
        c=[h[j]/(h[j-1]+h[j]) for j in range (1,n-1)]       #c从1到n-2，共n-2个
        a=[h[j-1]/(h[j-1]+h[j]) for j in range (2,n)]    #a从2到n-1，共n-2个
        d=[6*(y[j-1]/(h[j-1]*(h[j-1]+h[j]))+y[j+1]/(h[j]*(h[j-1]+h[j]))-y[j]/(h[j-1]*h[j])) for j in range (1,n)]#d从1到n-1，共n-1个
        d[0]-=h[0]*f2a/(h[0]+h[1])      #补上两端点的二阶导信息
        d[-1]-=h[-1]*f2b/(h[-2]+h[-1])
        b=[2]*(n-1)         #对角元都为2，共n-1个
        M=TripleDiagnalMatrix(a,b,c,d)    #x为三弯矩方程的解，n-1维
        M=[f2a]+M+[f2b]         #利用自然边界条件扩充M0和Mn，n+1维
    else:       #采用一阶导数边界条件
        c=[h[j]/(h[j-1]+h[j]) for j in range (1,n)]       #c从1到n-1，共n-1个
        c=[1]+c
        a=[h[j-1]/(h[j-1]+h[j]) for j in range (1,n)]    #a从1到n-1，共n-1个
        a=a+[1]#a从1到n，共n个
        d=[6*(y[j-1]/(h[j-1]*(h[j-1]+h[j]))+y[j+1]/(h[j]*(h[j-1]+h[j]))-y[j]/(h[j-1]*h[j])) for j in range (1,n)]#d从1到n-1，共n-1个
        d=[6*((y[1]-y[0])/h[0]-f1a)/h[0]]+d+[6*(f1b-(y[n]-y[n-1])/h[-1])/h[-1]]#补充上两端点的一阶导信息，共(n+1)维
        b=[2]*(n+1)         #对角元都为2，共n+1个
        M=TripleDiagnalMatrix(a,b,c,d)    #x为三弯矩方程的解，n+1维
        del(b,a,c,d)    #删除解弯矩方程使用的中间变量
    #用于作业的第二小问
    print('第二小问：')
    for j in range(n):
        print('(','%1.5f' %(M[j]/(6*h[j])),')(',x[j+1],'-t)^3+(','%1.5f' %(M[j+1]/(6*h[j])),')(t-',x[j],')^3+(','%1.5f' %(y[j]/h[j]-M[j]*h[j]/6),')(',x[j+1],'-t)+(','%1.5f' %(y[j+1]/h[j]-M[j+1]*h[j]/6),')(t-',x[j],')    &t \in [',j,',',j+1,r']\\',sep='')
    def CubicSplineFunction(x_new):
        j=0
        if x_new!=x[0]:
            while (x_new>x[j]):         #找到所在区域
                j+=1
            j-=1
        return ((M[j]*((x[j+1]-x_new)**3)+M[j+1]*((x_new-x[j])**3))/(6*h[j])+(y[j]-M[j]*(h[j]**2)/6)*(x[j+1]-x_new)/h[j]+(y[j+1]-M[j+1]*(h[j]**2)/6)*(x_new-x[j])/h[j])
    return CubicSplineFunction

def Spline_Plot(n):
    '''作心型线，n为作图时的取点个数'''
    t=list(range(9))
    phi=[x*math.pi/4 for x in t]
    r_phi=[1-math.cos(x) for x in phi]
    x_phi_0=[(1-math.cos(x))*math.cos(x) for x in phi]
    y_phi_0=[(1-math.cos(x))*math.sin(x) for x in phi]
    x_Spline=CubicSpline(t,x_phi_0)
    y_Spline=CubicSpline(t,y_phi_0)

    '''#用于第一小题
    print("x_t", ' & '.join(['%1.4f'% x for x in x_phi_0 ]))
    print("y_t", ' & '.join(['%1.4f'% x for x in y_phi_0 ]))
    '''

    t=np.linspace(0,8,n)
    phi=[x*math.pi/4 for x in t]
    r_phi=[1-math.cos(x) for x in phi]
    x_phi=[(1-math.cos(x))*math.cos(x) for x in phi]
    y_phi=[(1-math.cos(x))*math.sin(x) for x in phi]
    x_Interpolation=[x_Spline(x) for x in t]
    y_Interpolation=[y_Spline(x) for x in t]

    #作图
    plt.plot(x_phi,y_phi,'r',label=r'$r(\phi)$')
    plt.plot(x_Interpolation,y_Interpolation,'g--',label=r'$(S_\Delta(X;t),S_\Delta(Y;t))$')
    plt.plot(x_phi_0,y_phi_0,'ko',label=r'$(x_t,y_t)$')
    plt.legend()
    plt.axis("equal")
    plt.title('Cardioid')
    plt.show()
    

Spline_Plot(100)#作图的取点数