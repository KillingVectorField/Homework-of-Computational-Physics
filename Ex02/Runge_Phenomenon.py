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

def LagrangeInterpolation(x,y):
    '''拉格朗日插值法构造'''
    n=len(x)
    def Lagrange(x_new):                    # Lagrange为一个n维向量，其分量为L_i(x)，是x_new的函数
        L_i=[1]*n
        for i in range(n):
            for j in list(range(0,i))+list(range(i+1,n)):
                L_i[i]=L_i[i]*(x_new-x[j])/(x[i]-x[j])
        return L_i
    def LagrangePolynomial(x_new):          #Lagrange插值多项式函数
        L=Lagrange(x_new)                   #L[i]即为L_i
        sum=0
        for i in range(n):
            sum+=y[i]*L[i]
        return sum
    return LagrangePolynomial               #返回LagrangePolynomial这个函数

def NewtonInterpolation(x,y):
    '''Newton 插值构造'''
    n=len(x)
    A=[[1 for j in range(n)] for i in range(n)]
    for i in range(1,n):
        for j in range (1,i+1):         #j是能取到i的
            A[i][j]=A[i][j-1]*(x[i]-x[j-1])
    diff=BackSubstitutionL(A,y)
    def NewtonPolynomial(x_new):
        sum=diff[0]
        for i in range(1,n):
            temp=diff[i]
            for j in range(i):
                temp=temp*(x_new-x[j])
            sum+=temp
        return sum
    return NewtonPolynomial

def ChebyshevPolynomial(n,x):
    '''第一类切比雪夫多项式，n为阶数'''
    '''用递推定义，阶数高时运算速度很慢
    if n==0:return 1
    elif n==1: return x
    else: 
        return 2*x*ChebyshevPolynomial(n-1,x)-ChebyshevPolynomial(n-2,x)'''
    return math.cos(n*math.acos(x))

def ChebyshevApproximation(n,y):
    '''切比雪夫近似'''
    c=[0]*n
    for j in range(n):
        for k in range(n):
            c[j]+=y[k]*math.cos(math.pi*j*(k+1.0/2)/n)
    def Chebyapprox(x):
        sum=c[0]/2
        for i in range (1,n):
            sum+=c[i]*ChebyshevPolynomial(i,x)
        sum*=2/n
        return sum
    return Chebyapprox

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
    def CubicSplineFunction(x_new):
        j=0
        if x_new!=x[0]:
            while (x_new>x[j]):         #找到所在区域
                j+=1
            j-=1
        return ((M[j]*((x[j+1]-x_new)**3)+M[j+1]*((x_new-x[j])**3))/(6*h[j])+(y[j]-M[j]*(h[j]**2)/6)*(x[j+1]-x_new)/h[j]+(y[j+1]-M[j+1]*(h[j]**2)/6)*(x_new-x[j])/h[j])
    return CubicSplineFunction

def Runge_Phenomenon(plot_points_number):
    #test=[[0.0,1.0,3.0],[1.0,3.0,2.0]]

    Runge_Uniform_x=np.linspace(-1,1,21)
    Runge_Uniform_y=1/(1+25* Runge_Uniform_x**2)
    Runge_Uniform_x_41=np.linspace(-1,1,41)
    Runge_Uniform_y_41=[1/(1+25* x**2) for x in Runge_Uniform_x_41]
    #print('Runge_Uniform_y_41:',Runge_Uniform_y_41)

    LagrangePoly=LagrangeInterpolation(Runge_Uniform_x,Runge_Uniform_y)

    NewtonPoly=NewtonInterpolation(Runge_Uniform_x,Runge_Uniform_y)

    Runge_Chebyshev_x=[math.cos(math.pi*(i+0.5)/20.0) for i in range(20)]#切比雪夫多项式的n个零点
    Runge_Chebyshev_y=[1/(1+25*(x**2)) for x in Runge_Chebyshev_x]
    #print('Runge_Chebyshev_x',Runge_Chebyshev_x)
    #print('Runge_Chebyshev_y',Runge_Chebyshev_y)
    ChebyshevApprox=ChebyshevApproximation(20,Runge_Chebyshev_y)
    #print('Chebishev:',[ChebyshevApprox(x) for x in Runge_Chebyshev_x])
    
    CubicSplineFunction=CubicSpline(Runge_Uniform_x,Runge_Uniform_y)#未设置端点处的一阶导或二阶导，默认采用自然边界条件
    #print('CubicSpline:',[CubicSplineFunction(x) for x in Runge_Uniform_x])
    
    Runge_Uniform_x_n=np.linspace(-1,1,plot_points_number)  #取n个点作散点图
    Runge_Uniform_y_n=1/(1+25* Runge_Uniform_x_n**2)

    '''#用于第一小问
    LagrangePoly_41=[LagrangePoly(x) for x in Runge_Uniform_x_41]
    print('Lagrange:',LagrangePoly_41)
    print("Lagrange's deviation:", [abs(LagrangePoly_41[i]-Runge_Uniform_y_41[i]) for i in range(41)])
    NewtonPoly_41=[NewtonPoly(x) for x in Runge_Uniform_x_41]
    print('Newton:',NewtonPoly_41)
    print("Newton's deviation:", [abs(NewtonPoly_41[i]-Runge_Uniform_y_41[i]) for i in range(41)])

    plt.subplot(121)
    plt.plot(Runge_Uniform_x_n,Runge_Uniform_y_n,color='k',linestyle='--',label=r'$f(x)$') 
    plt.plot(Runge_Uniform_x_n,[LagrangePoly(x) for x in Runge_Uniform_x_n],color='C1',label=r'$L_{20}(x)$')
    plt.legend()
    plt.axis([-1.05,1.05,-0.5,1.5])
    plt.subplot(122)
    plt.plot(Runge_Uniform_x_n,Runge_Uniform_y_n,color='k',linestyle='--',label=r'$f(x)$') 
    plt.plot(Runge_Uniform_x_n,[LagrangePoly(x) for x in Runge_Uniform_x_n],color='C1',label=r'$L_{20}(x)$')
    plt.legend()
    plt.axis([-1.05,1.05,-65,10])
    plt.suptitle("Runge Phenomenon from Interpolation by Polynomials")
    plt.show()
    '''

    '''#用于第二小问
    ChebyshevApprox_41=[ChebyshevApprox(x) for x in Runge_Uniform_x_41]
    print("ChebyshevApproximation:",ChebyshevApprox_41)
    print("Chebyshev's deviation:",[abs(ChebyshevApprox_41[i]-Runge_Uniform_y_41[i]) for i in range(41)])
    
    plt.plot(Runge_Uniform_x_n,Runge_Uniform_y_n,color='k',linestyle='--',label=r'$f(x)$') 
    plt.plot(Runge_Uniform_x_n,[ChebyshevApprox(x) for x in Runge_Uniform_x_n],color='C2',label='Chebyshev')
    plt.plot(Runge_Chebyshev_x,Runge_Chebyshev_y,'*',color='r',label='Chebyshev Zero Points')
    plt.legend()
    plt.axis([-1.05,1.05,-0.5,1.5])
    plt.title("Chebyshev Approximation")
    plt.show()
    '''

    '''#用于第三小问
    CubicSpline_41=[CubicSplineFunction(x) for x in Runge_Uniform_x_41]
    print("CubicSpline:",' & '.join(['%1.8f' % x for x in CubicSpline_41]))
    print("CubicSpline's deviation:",' & '.join(['%1.8f' % abs(CubicSpline_41[i]-Runge_Uniform_y_41[i]) for i in range(41)]))

    plt.plot(Runge_Uniform_x_n,Runge_Uniform_y_n,color='k',linestyle='--',label=r'$f(x)$') 
    plt.plot(Runge_Uniform_x_n,[CubicSplineFunction(x) for x in Runge_Uniform_x_n],color='C3',label=r'$S(x)$')
    plt.legend()
    plt.axis([-1.05,1.05,-0.5,1.5])
    plt.title("Cubic Spline Interpolation")
    plt.show()
    '''

    
    plt.plot(Runge_Uniform_x_n,Runge_Uniform_y_n,color='k',linestyle='--',label=r'$f(x)$') 
    plt.plot(Runge_Uniform_x_n,[LagrangePoly(x) for x in Runge_Uniform_x_n],color='C1',label=r'$L_{20}(x)$')
    plt.plot(Runge_Uniform_x_n,[ChebyshevApprox(x) for x in Runge_Uniform_x_n],color='C2',label=r'$C(x)$')
    plt.plot(Runge_Uniform_x_n,[CubicSplineFunction(x) for x in Runge_Uniform_x_n],color='C3',label=r'$S(x)$')
    plt.legend()
    plt.axis([-1.05,1.05,-0.5,1.5])
    plt.title('Runge Phenomenon')
    plt.show()
    
    

Runge_Phenomenon(101)#作图所用的点的数量
