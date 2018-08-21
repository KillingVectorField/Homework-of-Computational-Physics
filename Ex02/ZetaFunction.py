import numpy as np
import math
import matplotlib.pyplot as plt

MinFloat=10**(-40)
#MinFloat=1/float('inf')
y_00=1/math.sqrt(4*math.pi)         #常数系数
precision=10**(-12)                 #精度要求

def Bisection_Root_Finding(f,a,b,eps):
    A=f(a)
    B=f(b)
    if A*B>=0:
        if A==0:return a
        elif B==0:return b
        else:
            print('improper district')
            return
    while b-a>eps:
        c=(a+b)/2
        C=f(c)
        if C==0:return c
        elif A*C>0:
            a=c
            A=C
        else: 
            b=c
            B=C
    return (a+b)/2

def Composite_Simpson(f,a,b,n=1):
    '''复合辛普森公示。f为函数，a为积分下界，b为积分上界，n为分段的区间个数'''
    x=np.linspace(a,b,2*n+1)
    y=[f(x) for x in x]
    S=y[0]+y[-1]+4*y[1]
    for i in range(1,n):
        S+=2*y[2*i]+4*y[2*i+1]
    S*=(b-a)/(6*n)
    return S

def Composite_Trapezoidal(f,a,b,n=1):
    '''复合梯形公式。f为函数，a为积分下界，b为积分上界，n为区间分段数'''
    x=np.linspace(a,b,n+1)
    y=[f(x) for x in x]
    S=y[0]+y[-1]
    for i in range(1,n):
        S+=2*y[i]
    S*=(b-a)/(2*n)
    return S

def Romberg_Integrate(f,a,b,error=10**(-8),Nmax=100):
    '''Romberg外推法积分'''
    T=[[Composite_Trapezoidal(f,a,b,1)]]
    #print(T)
    for n in range(1,Nmax):
        T[0].append(Composite_Simpson(f,a,b,2**n))
        T.append([])
        for i in range(1,n+1):#利用第i-1列的信息，填充第i列最后一行的元素
            T[i].append((4**i*T[i-1][-1]-T[i-1][-2])/(4**i-1))
        #print(T)
        if T[-1][-1]==0 and T[-2][-1]==0:return 0
        elif abs((T[-1][-1]-T[-2][-1])/T[-1][-1])<error:        #精度达到要求
            #print('Romberg Iter Times:',n)
            break
    #print('Romberg Table:',T)
    return T[-1][-1]

def SortLattice(top):
    '''为三维整数格点排序，top为考虑的n^2的最大值'''
    Tmp=[]
    times=1     #用于统计重复度
    for i in range(top):
        for j in range(i+1):
            for k in range(j+1):
                tmp=i*i+j*j+k*k             #计算格点(i,j,k)对应的n^2的值
                if tmp<top:
                    if(i>j>k):times=6           #格点3!种
                    elif(i==j==k):times=1
                    else:times=3                #ijk中有两个相同，3种
                    if k>0:times*=8             #ijk都能取+或-
                    elif j>0:times*=4           #jk能取+或-
                    elif i>0:times*=2           #k能取+或-
                    Tmp.append((tmp,times))
                else:                           #已经取够了所需的格点
                    Tmp.sort(key=lambda x:(x[0],-x[1])) #按n^2从小到大排序，当n^2值相同时，按重复度降序排序
                    return Tmp

SortedLattice=SortLattice(100)                

def R3(q2):
    '''计算第三项R3的积分，可以用Taylor级数展开,q^2为参数'''
    Sum=0
    x=1
    i=1
    while (x*math.pi/2>precision):
        x=q2**i/((i-0.5)*math.factorial(i))
        Sum+=x
        i+=1
    return Sum*math.pi/2

def R1(q2):
    '''计算第一项的值'''
    R1=0
    for x in SortedLattice:
        if (x[0]!=q2):      #分母不为零
            s=x[1]*math.exp(q2-x[0])/(x[0]-q2)
        else:               #分母为零
            print('q2=',q2,', singularity to Infinity!')
            return float('inf')
        if (R1!=0) and (abs(s*y_00)<max(precision,precision*abs(R1))):         #达到了R1的余项可忽略的条件：绝对值小于precision或相对求和总值小于precision
            print('q2=',q2,', when n^2>=',x[0],', the rest of R1 can be neglected.')
            return R1*y_00
        else:       #继续求sum
            R1+=s   

def R4(q2):
    '''计算第四项的值'''
    R4=0
    for x in SortedLattice[1:]:           #根据公式，n=000不参与求和
        s=x[1]*Romberg_Integrate(lambda t:t**(-3/2)*math.exp(t*q2-math.pi**2*x[0]/t),MinFloat,1,10**(-14))
        #s=x[1]*Composite_Simpson(lambda t:t**(-3/2)*math.exp(t*q2-math.pi**2*x[0]/t),MinFloat,1,100000)
        if (R4!=0) and (abs(y_00*math.pi*s)<precision):         #R4剩余的项可忽略的条件：第四项小于10^{-12}
            print('q2=',q2,', when n^2>=',x[0],', the rest of R4 can be neglected.')
            break
        else:
            R4+=s
    R4*=y_00*math.pi
    return R4

def Zeta_Function(q2):return R1(q2)-math.pi+R3(q2)+R4(q2)

def Z_Find_Root(q2):
    '''用于带入二分求根公式'''
    return Zeta_Function(q2)-math.pi**(3/2)*(1+q2/4)


'''#用于试验求积公式检验其精度等等
print('用sin(x)/x在0到1上的积分检验：')
print('Composite_Simpson:',Composite_Simpson(lambda x:math.sin(x)/x,MinFloat,1,100),'Composite_Trapezoidal:',Composite_Trapezoidal(lambda x:math.sin(x)/x,MinFloat,1,100))
T=Romberg_Integrate(lambda x:math.sin(x)/x,MinFloat,1,10**(-14))
print('Romberg:',T)
print('Mathematica: 0.946083070367184')
print('用第四项当q^2取0.5，n^2取2时的积分检验：')
print('Composite_Simpson:',Composite_Simpson(lambda t:t**(-3/2)*math.exp(t*0.5-math.pi**2*2/t),MinFloat,1,100000))
T=Romberg_Integrate(lambda t:t**(-3/2)*math.exp(t*0.5-math.pi**2*2/t),MinFloat,1,10**(-14))
print('Romberg:',T)
print('Mathematica: 2.1334536626989812*^-10')'''

#用于第一小题
print('第一小题：')
q2=np.linspace(MinFloat,3,100)
Z=[Zeta_Function(x) for x in q2]
plt.plot(q2,Z) 
plt.axis([0,3,-30,30])
plt.title(r"$Z_{00}(q^2)$")
plt.show()


'''#R3的作图
R3=[R3(x) for x in q2]
plt.plot(q2,R3) 
plt.title(r"$R_3(q^2)$")
plt.show()
'''
'''#R4作图
R4=[R4(x) for x in q2]
plt.plot(q2,R4) 
plt.title(r"$R_4(q^2)$")
plt.show()'''
#R1作图
'''
R1=[R1(x) for x in q2]
plt.plot(q2,R1) 
plt.axis([0,3,-50,50])
plt.title(r"$R_1(q^2)$")
plt.show()
'''

#print(Bisection_Root_Finding(R1,2.7,2.9,10**(-10)))#当第一项求和趋于0时需要计算多少项

#第二小题求解方程
Solution=Bisection_Root_Finding(Z_Find_Root,0.7,0.9,10**(-15))
print('第二小题：')
print('Solution:',Solution)

