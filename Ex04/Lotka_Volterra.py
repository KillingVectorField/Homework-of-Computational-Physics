import numpy as np
from matplotlib import pyplot as plt
(alpha,beta,gamma,delta)=(2/3,4/3,1,1)

def RK4(f,y_0,t_final,step=10**5,t_0=0):
    h=(t_final-t_0)/(step-1)
    t= np.linspace(t_0,t_final,step)#均匀步长，t从t[0]到t[n-1]
    y=[y_0]
    for n in range(len(t)-1):
        k1=h*f(t[n],y[n])
        k2=h*f(t[n]+h/2,y[n]+k1/2)
        k3=h*f(t[n]+h/2,y[n]+k2/2)
        k4=h*f(t[n]+h,y[n]+k3)
        y.append(y[n]+k1/6+k2/3+k3/3+k4/6)
    return (t,np.array(y))

def Lotka_Volterra(t,y):
    return np.array([alpha*y[0]-beta*y[0]*y[1],delta*y[0]*y[1]-gamma*y[1]])

def plot_in_turn(t):
    for init in np.array([[0.8,0.8],[1.0,1.0],[1.2,1.2],[1.4,1.4],[1.6,1.6]]):
        Result=RK4(Lotka_Volterra,init,t)
        plt.subplot(121)
        plt.plot(Result[0],Result[1][:,0],label='prey')
        plt.plot(Result[0],Result[1][:,1],label='predator')
        plt.xlabel(r't')
        plt.ylabel(r'x or y')
        plt.legend()
        plt.subplot(122)
        plt.plot(Result[1][:,0],Result[1][:,1])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axis('equal')
        plt.suptitle(r'$x(0)=%.1f,y(0)=%.1f$'%(init[0],init[1]))
        plt.show()

def plot_together(t):
    for init in np.array([[0.8,0.8],[1.0,1.0],[1.2,1.2],[1.4,1.4],[1.6,1.6]]):
        Result=RK4(Lotka_Volterra,init,t)
        plt.plot(Result[1][:,0],Result[1][:,1],label=r'$(%.1f,%.1f)$'%(init[0],init[1]))
    plt.plot([gamma/delta],[alpha/beta],'x',label=r'$(%.1f,%.1f)$'%(gamma/delta,alpha/beta))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.axis('equal')
    plt.show()

plot_together(12)
plot_in_turn(12)