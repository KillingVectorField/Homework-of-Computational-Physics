import numpy as np
from matplotlib import pyplot as plt
from scipy import stats #需要使用chisquare函数计算p值

#载入数据
f=open (r'correlation-function.dat','r')
data_str = f.read()
f.close()
data=np.empty(200*64)
tmp=[]
index=data_str.find('\t')
while index >=0:
    tmp.append(index)
    index=data_str.find('\t',index+1)
for i in range(len(tmp)-1):
    if i%2!=0:
        data[int(i/2)]=float(data_str[tmp[i]+1:tmp[i+1]])
data=data.reshape((200,64))
for j in range(1,32):#对称化
    data[:,j]=(data[:,j]+data[:,64-j])/2
data=data[:,:33]#只要前33个时间片
data_mean=np.mean(data,axis=0)       #求出每个时间片t的200次测量的均值
data_std=np.std(data,axis=0,ddof=1)
mean_std=data_std/np.sqrt(data.shape[0])
relative_error=abs(mean_std/data_mean)       #信噪比作为t的函数
def output_question_a():
    '''输出第1小题的结果'''
    print('C(t)的平均值：',data_mean)
    print('C(t)平均值的误差ΔC(t)：',mean_std)
    print(r'相对误差（*100%）：',100*relative_error)
    '''
    #用于Latex输出表格
    upper=r'$t$'
    lower=r'$\Delta C(t)/\bar{C}(t)$'
    for i in range(len(relative_error)):
        upper+='&%d'%i
        lower+='&%.2f'%(100*relative_error[i])
        if (i+1)%11==0:
            upper+='\n'+r'$t$'
            lower+='\n'+r'$\Delta C(t)/\bar{C}(t)$'
    print(upper)
    print(lower)
    '''
    #作图显示信噪比随时间的变化
    plt.plot(100*relative_error,'o')
    plt.xlabel(r'$t$')
    plt.ylabel(r'$\Delta C(t)/\bar{C}(t)\times100\%$')
    plt.title(r'Relative Error of $\bar{C}(t)$')
    plt.show()

def m_eff(data,definition):
    '''求有效质量'''
    if definition=='b':
        return np.array([np.log(data[i]/data[i+1]) for i in range(len(data)-1)])
    elif definition=='d':
        return np.array([np.arccosh((data[t+1]+data[t-1])/(2*data[t])) for t in range(1,len(data)-1)])

def output_qusetion(definition):
    if definition=='b':
        start=0  
        Exagg=40
    elif definition=='d':
        start=1
        Exagg=20
    m_eff_mean=m_eff(data_mean,definition)#用全样本的平均值带入计算
    #用Jackknife重抽样方法，估计m_eff的误差
    def Jackknife(data):
        resampled_m_eff=[]
        for i in range(data.shape[0]):
            resampled_data=np.vstack((data[:i],data[i+1:]))
            resampled_data_mean=np.mean(resampled_data,axis=0)
            resampled_m_eff.append(m_eff(resampled_data_mean,definition))
        return np.array(resampled_m_eff)

    resampled_m_eff=Jackknife(data)
    #print(np.mean(resampled_m_eff,axis=0))#由于meff函数是非线性的，所以并不等于用全样本平均值算出的m_eff
    mean_resampled_m_eff=np.mean(resampled_m_eff,axis=0)
    #根据讲义(7.45)式，估计Δm_eff
    dev_m_eff=np.sqrt(data.shape[0]-1)*np.std(resampled_m_eff,axis=0,ddof=0)#讲义(7.45)式
    if definition=='b':
        plt.bar(np.arange(len(dev_m_eff)),dev_m_eff)
    elif definition=='d':
        plt.bar(np.arange(1,len(dev_m_eff)+1),dev_m_eff)
    plt.xlabel(r'$t$')
    plt.ylabel(r'$\Delta m_{eff}$')
    plt.title(r'$\Delta m_{eff}$ of $m_{eff}$ defined in ('+definition+')')
    plt.show()

    def MLE_m(meff,dev_m):
        '''由推导出的求导公式，求第3小题中，使chi-squared最小的m'''
        return np.sum(meff/dev_m**2)/np.sum(1/dev_m**2)

    def Chi_Fitting(meff,dev_m,leaststep=3):
        record=(0,1,0,10**20,1)#分别记录[t_min,t_max,m,chi2,p_value]
        for t_min in range(len(meff)-leaststep):#i从0到len(meff)-leaststep-1
            for t_max in range(t_min+leaststep,len(meff)):
                m=MLE_m(meff[t_min:t_max+1],dev_m[t_min:t_max+1])#t_min=i,t_max=j-1，共t_max-t_min+1个
                chi2=np.sum(((meff[t_min:t_max+1]-m)/dev_m[t_min:t_max+1])**2)
                p_value=stats.chi2.cdf(chi2,df=t_max-t_min)
                if chi2/(t_max-t_min)<record[3]/(record[1]-record[0]):#以最小的chi2/d.o.f为标准拟合
                #if p_value<record[-1]:#以最小的p_value为标准拟合
                    record=(t_min,t_max,m,chi2,p_value)
        return dict(t_min=record[0]+start,t_max=record[1]+start,m=record[2],chi2=record[3],p_value=record[4])

    Result=Chi_Fitting(m_eff_mean,dev_m_eff)
    print('m defined in ('+definition+')',Result,sep='\n')
    #用区间内Δm的（平方）平均值估计拟合出的m的误差
    m_dev=np.sqrt(np.mean(dev_m_eff[Result['t_min']:Result['t_max']]**2))
    print('拟合出的m的误差估计：',m_dev)
    #用\Delta \chi2构造拟合出的m的（1 sigma）置信区间
    a=np.sum(1/dev_m_eff[Result['t_min']-start:Result['t_max']+1-start]**2)
    b=-2*np.sum(m_eff_mean[Result['t_min']-start:Result['t_max']+1-start]/(dev_m_eff[Result['t_min']-start:Result['t_max']+1-start]**2))
    c=np.sum((m_eff_mean[Result['t_min']-start:Result['t_max']+1-start]/dev_m_eff[Result['t_min']-start:Result['t_max']+1-start])**2)-(Result['chi2']+1)
    (m_lower_bound,m_upper_bound)=((-b-np.sqrt(b**2-4*a*c))/(2*a),(-b+np.sqrt(b**2-4*a*c))/(2*a))
    print(r'1 sigma confidence interval for fitted m defined in (%s): [%f, %f]'%(definition,m_lower_bound,m_upper_bound))
    
    plt.plot(np.arange(start,len(mean_resampled_m_eff)+start),m_eff_mean,'.',label=r'$m_{eff}$')
    plt.errorbar(np.arange(start,start+len(mean_resampled_m_eff)),mean_resampled_m_eff,yerr=dev_m_eff*Exagg,fmt='none',ecolor='r',label=r'$\Delta m_{eff}\times%d$'%Exagg)#误差放大了Exagg倍作图
    plt.plot([Result['t_min'],Result['t_max']],[Result['m']]*2,'k',label=r'fitted value $m=%.5f$'%Result['m'])
    plt.errorbar([1/2*(Result['t_min']+Result['t_max'])],[Result['m']],yerr=m_dev*Exagg,fmt='none',ecolor='k',label=r'estimated error of m $\times%d$'%Exagg)#误差放大了Exagg倍作图
    plt.xlabel(r'$t$')
    plt.ylabel(r'$m_{eff}$')
    plt.title(r'$m_{eff}$ defined in ('+definition+')')
    plt.errorbar
    plt.legend()
    plt.show()
    #放大到平台区作图
    plt.plot(np.arange(start,len(mean_resampled_m_eff)+start),m_eff_mean,'.',label=r'$m_{eff}$')
    plt.errorbar(np.arange(start,start+len(mean_resampled_m_eff)),mean_resampled_m_eff,yerr=dev_m_eff,fmt='none',ecolor='r',label=r'$\Delta m_{eff}$')
    plt.plot([Result['t_min'],Result['t_max']],[Result['m']]*2,'k',label=r'fitted value $m=%.5f$'%Result['m'])
    plt.errorbar([1/2*(Result['t_min']+Result['t_max'])],[Result['m']],yerr=m_dev,fmt='none',ecolor='k',label=r'estimated error of $m$')
    plt.plot([Result['t_min'],Result['t_max']],[m_upper_bound]*2,color='g',label=r"1 $\sigma$ confidence interval of fitted $m$: [%.5f, %.5f]"%(m_lower_bound,m_upper_bound))
    plt.plot([Result['t_min'],Result['t_max']],[m_lower_bound]*2,color='g')
    if definition=='b':
        plt.axis([9.5,28.5,1.145,1.18])
    elif definition=='d':
        plt.axis([12.5,31.5,1.145,1.17])
    plt.xlabel(r'$t$')
    plt.ylabel(r'$m_{eff}$')
    plt.title(r'$m_{eff}$ defined in ('+definition+')')
    plt.errorbar
    plt.legend()
    plt.show()
    
'''
#构造协方差矩阵
#C=np.array([np.array([np.sum((data[:,t]-data_mean[t])*(data[:,t_]-data_mean[t_])) for t_ in range(data.shape[1])]) for t in range(data.shape[1])])/(data.shape[0]-1)
C=np.cov(data,ddof=1,rowvar=False)
def correlation_coef(t,t_):
    return C[t,t_]/np.sqrt(C[t,t]*C[t_,t_])
print(correlation_coef(3,4),correlation_coef(3,5))
'''

def Bootstrap(data,N):
    '''用于第五小题'''
    r_34=np.empty(N)
    r_35=np.empty(N)
    for i in range(N):
        Bootstrap_sample=np.array([data[i] for i in np.random.randint(low=0,high=data.shape[0],size=data.shape[0])])
        C=np.cov(Bootstrap_sample,ddof=1,rowvar=False)#计算协方差矩阵
        '''当然也可以用自己的代码计算协方差矩阵'''
        #C=np.array([np.array([np.sum((data[:,t]-data_mean[t])*(data[:,t_]-data_mean[t_])) for t_ in range(data.shape[1])]) for t in range(data.shape[1])])/(data.shape[0]-1)
        r_34[i]=C[0,1]/np.sqrt(C[0,0]*C[1,1])
        r_35[i]=C[0,2]/np.sqrt(C[0,0]*C[2,2])
    plt.subplot(121)
    plt.hist(r_34)
    plt.xlabel(r'$\rho_{3,4}$')
    plt.title(r'Bootstrap of $\rho_{3,4}$')
    plt.subplot(122)
    plt.hist(r_35)
    plt.xlabel(r'$\rho_{3,5}$')
    plt.title(r'Bootstrap of $\rho_{3,5}$')
    plt.show()
    r_34.sort()
    r_35.sort()
    return (dict(r_34_mean=np.mean(r_34),r_34_std=np.std(r_34,ddof=1),r_34_interval68=(r_34[int(len(r_34)*0.16)-1],r_34[int(len(r_34)*0.84)])),
            dict(r_35_mean=np.mean(r_35),r_35_std=np.std(r_35,ddof=1),r_35_interval68=(r_35[int(len(r_35)*0.16)-1],r_35[int(len(r_35)*0.84)])))

def output_question_e():
    Bootstrap_Result=Bootstrap(data[:,3:6],N=1000)#第五小题的任务只和t=3,4,5的时间片有关
    print(Bootstrap_Result[0])
    print(Bootstrap_Result[1])

output_question_a()
output_qusetion('b')
output_qusetion('d')
output_question_e()

