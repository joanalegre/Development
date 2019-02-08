# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 10:03:29 2019

@author: Joana
"""

#%% NEW ONE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import itertools as it
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
#%% FUNCTION OF WELFARE GAINS:
def welfare(A,nhu):
    
    #IDIOSYNCRATIC STOCHASTIC PART: 
    
    
    et = np.exp(np.random.normal(0,0.2,(n,T)))*np.exp(-0.2/2) #Idiosyncratic shock.
    zt = np.exp(np.random.normal(0,0.2,n))*np.exp(-0.2/2) #Permanent Consumption.
    
    #TOTAL with seasonality and idiosyncratic risk:
    if nhu == 1:
        B = A.T
        B = B[0]
        lowseason = np.zeros((n,M,T))
        for i,m,t in it.product(range(n), range(M), range(T)):
            lowseason[i,m,t] = np.log(zt[i]*B[m]*et[i,t])*pow(beta,m+12*t)
          
        B = A.T
        B = B[1]
        mediumseason = np.zeros((n,M,T))
        for i,m,t in it.product(range(n), range(M), range(T)):
            mediumseason[i,m,t] = np.log(zt[i]*B[m]*et[i,t])*pow(beta,m+12*t)
        B = A.T
        B = B[2]
        highseason = np.zeros((n,M,T))
        for i,m,t in it.product(range(n), range(M), range(T)):
            highseason[i,m,t] = np.log(zt[i]*B[m]*et[i,t])*pow(beta,m+12*t)
        
        #TOTAL without seasonality:
        '''To eliminate seasonality I assum exp(gm) = 1 '''
        
        lowseason_nosea = np.zeros((n,M,T))
        for i,m,t in it.product(range(n), range(M), range(T)):
            lowseason_nosea[i,m,t] = np.log(zt[i]*et[i,t])*pow(beta,m+12*t)
    
        mediumseason_nosea = np.zeros((n,M,T))
        for i,m,t in it.product(range(n), range(M), range(T)):
            mediumseason_nosea[i,m,t] = np.log(zt[i]*et[i,t])*pow(beta,m+12*t)
    
        highseason_nosea = np.zeros((n,M,T))
        for i,m,t in it.product(range(n), range(M), range(T)):
            highseason_nosea[i,m,t] = np.log(zt[i]*et[i,t])*pow(beta,m+12*t)
        
        #TOTAL without idiosuncratic risk:  
        '''To eliminate idiosyncratic risk I set the mean of the different 40 years. '''
        B = A.T
        B = B[0]
    
        lowseason_noido = np.zeros((n,M,T))
        for i,m,t in it.product(range(n), range(M), range(T)):
            lowseason_noido[i,m,t] = np.log(zt[i]*B[m]*np.mean(et[i]))*pow(beta,m+12*t)
            
        B = A.T
        B = B[1]
        mediumseason_noido = np.zeros((n,M,T))
        for i,m,t in it.product(range(n), range(M), range(T)):
            mediumseason_noido[i,m,t] = np.log(zt[i]*B[m]*np.mean(et[i]))*pow(beta,m+12*t)
        
        B = A.T
        B = B[2]
        highseason_noido = np.zeros((n,M,T))
        for i,m,t in it.product(range(n), range(M), range(T)):
            highseason_noido[i,m,t] = np.log(zt[i]*B[m]*np.mean(et[i]))*pow(beta,m+12*t)
        
        #COMPUTING INDIVIDUAL WELFARE:
        
        Welfarelowseason = np.ones(n)
        Welfaremediumseason = np.ones(n)
        Welfarehighseason = np.ones(n)
        Welfarelowseason_nosea = np.ones(n)
        Welfaremediumseason_nosea  = np.ones(n)
        Welfarehighseason_nosea = np.ones(n)
        Welfarelowseason_noido = np.ones(n)
        Welfaremediumseason_noido = np.ones(n)
        Welfarehighseason_noido = np.ones(n)
        
        for i in range(n):       
            Welfarelowseason[i] = np.sum(lowseason[i])
            Welfaremediumseason[i] = np.sum(mediumseason[i])
            Welfarehighseason[i] = np.sum(highseason[i])
            Welfarelowseason_nosea[i] = np.sum(lowseason_nosea[i])
            Welfaremediumseason_nosea[i] =  np.sum(mediumseason_nosea[i])
            Welfarehighseason_nosea[i] = np.sum(highseason_nosea[i])
            Welfarelowseason_noido[i] = np.sum(lowseason_noido[i])
            Welfaremediumseason_noido[i] = np.sum(mediumseason_noido[i])
            Welfarehighseason_noido[i] = np.sum(highseason_noido[i])
        
        Welfare_individual = np.array([Welfarelowseason, Welfaremediumseason, Welfarehighseason, Welfarelowseason_nosea, Welfaremediumseason_nosea, Welfarehighseason_nosea, Welfarelowseason_noido,Welfaremediumseason_noido, Welfarehighseason_noido]).T
    
            
        
        #COMPUTE AGGREGATED WELFARE:
        Welfare_aggregate = np.sum(Welfare_individual, axis = 0) 
        
        C = Welfare_aggregate.reshape(3,3)
        table2 = pd.DataFrame(C, index = ['Everything', 'No seasonal', 'No Idiosyncratic'], columns = ['low', 'medium', 'high'])
        
        
        #WELFARE GAINS BOTH INDIVIDUAL AND AGGREGATED:
        β1=np.zeros((T))
        for t in range(T):
            β1[t]=pow(β,1/12)**(t*12)
        total_β1=np.sum(β1)  
        
        β2=np.zeros((M))
        for m in range(M):
            β2[m]=pow(β,1/12)**(m-1)
        total_β2=np.sum(β2)
        
        #Due to seasonality
        Welfare_individual = Welfare_individual.T
        g_h_seasonal=np.exp(( Welfare_individual[5]-Welfare_individual[2] )/(total_β2*total_β1)) - 1
        g_m_seasonal=np.exp(( Welfare_individual[4]-Welfare_individual[1])/(total_β2*total_β1)) - 1
        g_l_seasonal=np.exp(( Welfare_individual[3]-Welfare_individual[0])/(total_β2*total_β1)) - 1
        
        g_total_h_seasonal=np.exp(abs(Welfare_aggregate[2]-Welfare_aggregate[5])/(total_β2*total_β1)) - 1
        g_total_m_seasonal=np.exp(abs(Welfare_aggregate[1]-Welfare_aggregate[4])/(total_β2*total_β1)) - 1
        g_total_l_seasonal=np.exp((abs(Welfare_aggregate[0]-Welfare_aggregate[3]))/(total_β2*total_β1)) - 1
        
        #Due to Idiosyncratic non-seasonal:
        g_h_Idio=np.exp(((Welfare_individual[8]-Welfare_individual[2]))/(total_β2*total_β1)) - 1
        g_m_Idio=np.exp(((Welfare_individual[7]- Welfare_individual[1]))/(total_β2*total_β1)) - 1
        g_l_Idio=np.exp(((Welfare_individual[6]- Welfare_individual[0]))/(total_β2*total_β1)) - 1
        
        g_total_h_Idio=np.exp((abs(Welfare_aggregate[2]-Welfare_aggregate[8]))/(total_β2*total_β1)) - 1
        g_total_m_Idio=np.exp((abs(Welfare_aggregate[1]-Welfare_aggregate[7]))/(total_β2*total_β1)) - 1
        g_total_l_Idio=np.exp((abs(Welfare_aggregate[0]-Welfare_aggregate[6]))/(total_β2*total_β1)) - 1
        gan_ido = np.array([g_l_seasonal, g_m_seasonal, g_h_seasonal, g_l_Idio, g_m_Idio, g_h_Idio])
        gan = np.array([[g_total_l_seasonal, g_total_m_seasonal, g_total_h_seasonal],[g_total_l_Idio, g_total_m_Idio, g_total_h_Idio]])
        table3 = pd.DataFrame(gan, index = ['seasonal', 'Idiosyncratic'], columns = ['low', 'medium', 'high'])
       
    
    else:
        B = A.T
        B = B[0]
        lowseason = np.zeros((n,M,T))
        for i,m,t in it.product(range(n), range(M), range(T)):
            lowseason[i,m,t] = pow(zt[i]*B[m]*et[i,t],1-nhu)*pow(beta,m+12*t)/(1-nhu)
          
        B = A.T
        B = B[1]
        mediumseason = np.zeros((n,M,T))
        for i,m,t in it.product(range(n), range(M), range(T)):
            mediumseason[i,m,t] = pow(zt[i]*B[m]*et[i,t],1-nhu)*pow(beta,m+12*t)/(1-nhu)
        B = A.T
        B = B[2]
        highseason = np.zeros((n,M,T))
        for i,m,t in it.product(range(n), range(M), range(T)):
            highseason[i,m,t] = pow(zt[i]*B[m]*et[i,t],1-nhu)*pow(beta,m+12*t)/(1-nhu)
        
        #TOTAL without seasonality:
        '''To eliminate seasonality I assum exp(gm) = 1 '''
        
        lowseason_nosea = np.zeros((n,M,T))
        for i,m,t in it.product(range(n), range(M), range(T)):
            lowseason_nosea[i,m,t] = pow(zt[i]*et[i,t],1-nhu)*pow(beta,m+12*t)/(1-nhu)
        
        mediumseason_nosea = np.zeros((n,M,T))
        for i,m,t in it.product(range(n), range(M), range(T)):
            mediumseason_nosea[i,m,t] = pow(zt[i]*et[i,t],1-nhu)*pow(beta,m+12*t)/(1-nhu)
        
        highseason_nosea = np.zeros((n,M,T))
        for i,m,t in it.product(range(n), range(M), range(T)):
            highseason_nosea[i,m,t] = pow(zt[i]*et[i,t],1-nhu)*pow(beta,m+12*t)/(1-nhu)
        
        #TOTAL without idiosuncratic risk:  
        '''To eliminate idiosyncratic risk I set the mean of the different 40 years. '''
        B = A.T
        B = B[0]
        
        lowseason_noido = np.zeros((n,M,T))
        for i,m,t in it.product(range(n), range(M), range(T)):
            
            lowseason_noido[i,m,t] = pow(zt[i]*B[m]*np.mean(et[i]),1-nhu)*pow(beta,m+12*t)/(1-nhu)
            
        B = A.T
        B = B[1]
        mediumseason_noido = np.zeros((n,M,T))
        for i,m,t in it.product(range(n), range(M), range(T)):
            mediumseason_noido[i,m,t] = pow(zt[i]*B[m]*np.mean(et[i]),1-nhu)*pow(beta,m+12*t)/(1-nhu)
        
        B = A.T
        B = B[2]
        highseason_noido = np.zeros((n,M,T))
        for i,m,t in it.product(range(n), range(M), range(T)):
            highseason_noido[i,m,t] = pow(zt[i]*B[m]*np.mean(et[i]),1-nhu)*pow(beta,m+12*t)/(1-nhu)
        
        #COMPUTING INDIVIDUAL WELFARE:
        
        Welfarelowseason = np.ones(n)
        Welfaremediumseason = np.ones(n)
        Welfarehighseason = np.ones(n)
        Welfarelowseason_nosea = np.ones(n)
        Welfaremediumseason_nosea  = np.ones(n)
        Welfarehighseason_nosea = np.ones(n)
        Welfarelowseason_noido = np.ones(n)
        Welfaremediumseason_noido = np.ones(n)
        Welfarehighseason_noido = np.ones(n)
        
        for i in range(n):       
            Welfarelowseason[i] = np.sum(lowseason[i])
            Welfaremediumseason[i] = np.sum(mediumseason[i])
            Welfarehighseason[i] = np.sum(highseason[i])
            Welfarelowseason_nosea[i] = np.sum(lowseason_nosea[i])
            Welfaremediumseason_nosea[i] =  np.sum(mediumseason_nosea[i])
            Welfarehighseason_nosea[i] = np.sum(highseason_nosea[i])
            Welfarelowseason_noido[i] = np.sum(lowseason_noido[i])
            Welfaremediumseason_noido[i] = np.sum(mediumseason_noido[i])
            Welfarehighseason_noido[i] = np.sum(highseason_noido[i])
        
        Welfare_individual = np.array([Welfarelowseason, Welfaremediumseason, Welfarehighseason, Welfarelowseason_nosea, Welfaremediumseason_nosea, Welfarehighseason_nosea, Welfarelowseason_noido,Welfaremediumseason_noido, Welfarehighseason_noido]).T
        
            
        
        #COMPUTE AGGREGATED WELFARE:
        Welfare_aggregate = np.sum(Welfare_individual, axis = 0) 
        
        C = Welfare_aggregate.reshape(3,3)
        table2 = pd.DataFrame(C, index = ['Everything', 'No seasonal', 'No Idiosyncratic'], columns = ['low', 'medium', 'high'])
        
        
        #WELFARE GAINS BOTH INDIVIDUAL AND AGGREGATED:
        β1=np.zeros((T))
        for t in range(T):
            β1[t]=pow(β,1/12)**(t*12)
        total_β1=np.sum(β1)  
        
        β2=np.zeros((M))
        for m in range(M):
            β2[m]=pow(β,1/12)**(m-1)
        total_β2=np.sum(β2)
        
        #Due to seasonality
        Welfare_individual = Welfare_individual.T
        g_h_seasonal=np.exp(( Welfare_individual[5]-Welfare_individual[2] )/(total_β2*total_β1)) - 1
        g_m_seasonal=np.exp(( Welfare_individual[4]-Welfare_individual[1])/(total_β2*total_β1)) - 1
        g_l_seasonal=np.exp(( Welfare_individual[3]-Welfare_individual[0])/(total_β2*total_β1)) - 1
        
        g_total_h_seasonal=np.exp(abs(Welfare_aggregate[2]-Welfare_aggregate[5])/(total_β2*total_β1)) - 1
        g_total_m_seasonal=np.exp(abs(Welfare_aggregate[1]-Welfare_aggregate[4])/(total_β2*total_β1)) - 1
        g_total_l_seasonal=np.exp((abs(Welfare_aggregate[0]-Welfare_aggregate[3]))/(total_β2*total_β1)) - 1
        
        #Due to Idiosyncratic non-seasonal:
        g_h_Idio=np.exp(((Welfare_individual[8]-Welfare_individual[2]))/(total_β2*total_β1)) - 1
        g_m_Idio=np.exp(((Welfare_individual[7]- Welfare_individual[1]))/(total_β2*total_β1)) - 1
        g_l_Idio=np.exp(((Welfare_individual[6]- Welfare_individual[0]))/(total_β2*total_β1)) - 1
        
        g_total_h_Idio=np.exp((abs(Welfare_aggregate[2]-Welfare_aggregate[8]))/(total_β2*total_β1)) - 1
        g_total_m_Idio=np.exp((abs(Welfare_aggregate[1]-Welfare_aggregate[7]))/(total_β2*total_β1)) - 1
        g_total_l_Idio=np.exp((abs(Welfare_aggregate[0]-Welfare_aggregate[6]))/(total_β2*total_β1)) - 1
        gan_ido = np.array([g_l_seasonal, g_m_seasonal, g_h_seasonal, g_l_Idio, g_m_Idio, g_h_Idio])
        gan = np.array([[g_total_l_seasonal, g_total_m_seasonal, g_total_h_seasonal],[g_total_l_Idio, g_total_m_Idio, g_total_h_Idio]])
        table3 = pd.DataFrame(gan, index = ['seasonal', 'Idiosyncratic'], columns = ['low', 'medium', 'high'])
           
    return table2, table3, gan_ido

#%% QUESTION 1.1
np.random.seed( 10 )
random.seed(10)

#Parameters
β=0.99
σ=0.2
n=1000
T=40
M=12

beta = pow(0.99,1/12)


A = np.array([[-0.073, -0.185, 0.071, 0.066, 0.045, 0.029, 0.018, 0.018, 0.018, 0.001, -0.017, -0.041], [-0.147, -0.370, 0.141,  0.131, 0.090, 0.058, 0.036, 0.036, 0.036, 0.002, -0.033,  -0.082], [-0.293, -0.739, 0.282, 0.262, 0.180, 0.116, 0.072, 0.072, 0.072,0.004,-0.066,-0.164]])
A = np.exp(A)
#A = np.array([[0.932,0.845,1.076,1.07,1.07,1.047,1.03,1.018,1.018,1.001,0.984,0.961],[0.863,0.661,1.151,1.140,1.094,1.060,1.037,1.037,1.037,1.002,0.968,0.921],[0.727,0.381,1.303,1.280,1.188,1.119,1.073,1.073,1.073,1.004,0.935,0.843]])
A = A.T

table1 = pd.DataFrame(A, index = ['january', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'], columns = ['low', 'medium', 'high'])



table2, table3, gan_ido = welfare(A, 1)
print(table1.to_latex())
print(table2.to_latex())
print(table3.to_latex())

'''Explanation table 2 and table 3:
    In table 2 we can see the values of total welfare for low, medium and high degree of seasonality
    in the cases of having both seasonal and non seasonal risk (Everything), in the case of having
    only non-seasonal risk (No seasonal) and in the case of having only seasonal risk (No Idiosyncratic)
    As expected when we have no seasonal risk welfare is equal in low, medium an high.
    When we have everything we are worse off w.r.t no seasonal and Idiosyncratic.
    Also, it is interesting see that the more volatility has the economy, the worse
    is the welfare in all cases. 
    In table 3 we can see the welfare gains of removing seasonal and non-seasonal risk.
    We can see that removing non-seasonal risk is always better, nevertheless, the higher
    is the degree of seasonality the lower is the welfare gap between removing non seasonal
    and seasonal risk.
    Another feature of the table is that we can see how the higher is the degree of seasonality
    the higher is the gain of welfare.'''
    


bins=50 #Adjust the number of bins
plt.hist(gan_ido,bins, alpha=0.5)
plt.title('Non Idosyncratic risk gain distribution for eta=1')
plt.show()

#%% For nhu = 2 and nhu = 4:


table4, table5, gan_ido_nhu2 = welfare(A,2)

table6, table7, gan_ido_nhu4 = welfare(A,4)


print(table4.to_latex())
print(table5.to_latex())
print(table6.to_latex())
print(table7.to_latex())
'''Table4 explanation: We can see how the higher is the degree of seasonality the lower
is the welfare as in table 2.
Table 5 explanation: The big difference with table 3 is that now when high degree
of seasonality we have that the gain of removing seasonal risk is higher than removing non-seasonal risk'''

'''Table6 explanation: As before the more degree of seasonality the worse. Nevrtheless, notice
that the decreament of welfare is really high in comparation with the previous cases.
Table7 explanation: As a result of what we were talking in table6 we have that total gains
are much higher removing seasonal risk than non-seasonal risk.

In overall we can say, that it is not only the nature of the season and the variance of the idiosyncratic
shock what determines welfare, but also adversion to risk is a big factor which we have
to take into account. Since depending on the adversion of risk we have that the optimal
policy ( removing seasonal risk or non-seasonal risk) change. '''

plt.figure
plt.subplots_adjust(top=0.8, bottom=0.2, left=0.3, right=2, wspace=0.5)
plt.suptitle('Welfare gains distribution for eta=2 and 3')
bins=50 #Adjust the number of bins
plt.subplot(1,2,1)
plt.hist(gan_ido_nhu2,bins, alpha=0.5)
plt.title('Non Idosyncratic risk gain eta=2, distribution')

plt.subplot(1,2,2)
plt.hist(gan_ido_nhu4,bins, alpha=0.5)
plt.title('Non Idosyncratic risk gain eta=4, distribution')
plt.show()


#%% QUESTION 1.2:

var = np.array([[0.043,  0.034, 0.145,  0.142,  0.137,  0.137,  0.119,  0.102,  0.094,  0.094,  0.085,  0.068], [0.085,0.068,0.290,0.283,0.273,0.273,0.239, 0.205,  0.188,0.188, 0.171,0.137], [0.171, 0.137,0.580, 0.567, 0.546, 0.546,0.478,0.410,0.376,0.376,0.341,0.273]])
 
var = np.exp((-0.5*var))
var = np.exp(np.random.normal(0,0.2,(12)))*var
var = var.T
A = A*var

table8, table9, gan_ido = welfare(A, 1)
print(table8.to_latex())
print(table9.to_latex())


display(table8, table9, table3)

table10, table11, gan_ido_nhu2 = welfare(A,2)

table12, table13, gan_ido_nhu4 = welfare(A,4)

display (table10, table11, table12, table13)

print(table10.to_latex())
print(table11.to_latex())

print(table12.to_latex())
print(table13.to_latex())
'''Explanation table  8-table13:
    In this case we can see easily that removing seasonality is genereting a high improvement
    The more volatil is the economy the worse, as usual.
    Table 11 says exactly the same, removing idiosyncratic shock will generate a lower
    welfare than removing seasonality.
    In the following cases the more we increase nhu the higher is the gain of removing seasonality
    Nevertheless, in the case of nhu=2 in the low degree of seasonality we have that 
    removing non-seasonal risk is better. This change when degree of seasonality increases.'''


plt.figure
plt.subplots_adjust(top=0.8, bottom=0.2, left=0.3, right=2, wspace=0.5)
plt.suptitle('Welfare gains distribution for eta=2 and 3')
bins=50 #Adjust the number of bins
plt.subplot(1,3,1)
plt.hist(gan_ido_nhu4,bins, alpha=0.5)
plt.title('Non Idosyncratic risk gain eta=1, distribution')

plt.subplot(1,3,2)
plt.hist(gan_ido_nhu2,bins, alpha=0.5)
plt.title('Non Idosyncratic risk gain eta=2, distribution')

plt.subplot(1,3,3)
plt.hist(gan_ido_nhu4,bins, alpha=0.5)
plt.title('Non Idosyncratic risk gain eta=4, distribution')
plt.show()

#%% QUESTION 2:
import numpy as np
import matplotlib.pyplot as plt
import random
import itertools as it
import pandas as pd 
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

np.random.seed( 10 )
random.seed(10)

#Parameters
β=0.99
σ=0.2
σ_l=0.3
n=1000
T=40
M=12


#1.- Processes for LABOUR
#Permanent level of labour
log_u=np.random.normal(0,σ_l,n)
z_l = np.zeros((n))
for i in range(n):
    z_l[i]=np.exp(log_u[i])*np.exp(-σ_l/2)
    
#Idiosyncratic non-stationary stochastic component
log_e=np.random.normal(0,σ,(n,T))
l= np.zeros((n,T))
for i,t in it.product(range(n),range(T)):
    l[i,t]=np.exp(log_e[i,t])*np.exp(-σ_l/2)


#Seasonal risk (only one degree of seasonality)
#Deterministic component (low degree of seasonality of new table1)
g_l=[0.932,0.845, 1.076,1.070,1.047,1.030,1.018,1.018,1.018,1.001,0.984,0.961 ]

#Stochastic (low of table2)
s_l=[ 0.043,  0.034, 0.145,  0.142,  0.137,  0.137,  0.119,  0.102,  0.094,  0.094,  0.085,  0.068]

#Build the vectors
eps_l=np.zeros((M))
for m in range(M):
    eps_l[m]=np.random.normal(0,s_l[m],1)

s_labour=np.zeros((M))
for m in range(M):
    s_labour[m]=np.exp(-s_l[m]/2)*np.exp(eps_l[m])



#2.- Processes for CONSUMPTION (same as 1.2.)
#Permanent level of consumption
log_u=np.random.normal(0,σ,n)
z = np.zeros((n))
for i in range(n):
    z[i]=np.exp(log_u[i])*np.exp(-σ/2)

#Idiosyncratic non-stationary stochastic component
log_e=np.random.normal(0,σ,(n,T))
c = np.zeros((n,T))
for i,t in it.product(range(n),range(T)):
    c[i,t]=np.exp(log_e[i,t])*np.exp(-σ/2)
    

#Seasonal risk
#Deterministic component
g_middle=[-0.147, -0.370, 0.141,  0.131, 0.090, 0.058, 0.036, 0.036, 0.036, 0.002, -0.033,  -0.082]
g_high=[-0.293, -0.739, 0.282, 0.262, 0.180, 0.116, 0.072, 0.072, 0.072,0.004,-0.066,-0.164]
g_low=[-0.073, -0.185, 0.071, 0.066, 0.045, 0.029, 0.018, 0.018, 0.018, 0.001, -0.017, -0.041]

#Stochastic component
s_m=[0.085,0.068,0.290,0.283,0.273,0.273,0.239, 0.205,  0.188,0.188, 0.171,0.137]
s_h=[0.171, 0.137,0.580, 0.567, 0.546, 0.546,0.478,0.410,0.376,0.376,0.341,0.273]
s_l=[ 0.043,  0.034, 0.145,  0.142,  0.137,  0.137,  0.119,  0.102,  0.094,  0.094,  0.085,  0.068]

#Middle degree
eps_m=np.zeros((M))
for m in range(M):
    eps_m[m]=np.random.normal(0,s_m[m],1)

s_m=np.zeros((M))
for m in range(M):
    s_m[m]=np.exp(-s_m[m]/2)*np.exp(eps_m[m])
    
#High degree
eps_h=np.zeros((M))
for m in range(M):
    eps_h[m]=np.random.normal(0,s_h[m],1)

s_h=np.zeros((M))
for m in range(M):
    s_h[m]=np.exp(-s_h[m]/2)*np.exp(eps_h[m])

#Low degree
eps_l=np.zeros((M))
for m in range(M):
    eps_l[m]=np.random.normal(0,s_l[m],1)

s_l=np.zeros((M))
for m in range(M):
    s_l[m]=np.exp(-s_l[m]/2)*np.exp(eps_l[m])




#3.- UTILITY
#FROM CONSUMPTION
c_mt_middle = np.zeros((n,M,T))
for i,m,t in it.product(range(n),range(12),range(T)):
    c_mt_middle[i,m,t]=np.log(z[i]*np.exp(g_middle[m])*s_m[m]*c[i,t])

c_mt_high = np.zeros((n,M,T))
for i,m,t in it.product(range(n),range(12),range(T)):
    c_mt_high[i,m,t]=np.log(z[i]*np.exp(g_high[m])*s_h[m]*c[i,t])   

c_mt_low = np.zeros((n,M,T))
for i,m,t in it.product(range(n),range(12),range(T)):
    c_mt_low[i,m,t]=np.log(z[i]*np.exp(g_low[m])*s_l[m]*c[i,t])  



#From labour
#Dirty calibration
labour=np.zeros((n,M,T))
consumption=np.zeros((n,M,T))
for i,m,t in it.product(range(n),range(12),range(T)):
    labour[i,m,t]=z_l[i]*s_labour[m]*l[i,t]
    consumption[i,m,t]=z[i]*np.exp(g_middle[m])*s_m[m]*c[i,t]   
κ=1/(np.mean(labour)*np.mean(consumption))


#Desutility from labour
u_l=np.zeros((n,M,T))
for i,m,t in it.product(range(n),range(12),range(T)):
    u_l[i,m,t]=-κ*pow(labour[i,m,t],2)/2
    
#Total utility
u=c_mt_middle+u_l




#4.- LIFETIME WELFARE
w=np.zeros((n,M,T))
β1=np.zeros((T))
for t in range(T):
    β1[t]=pow(β,1/12)**(t*12)
total_β1=np.sum(β1)  
  
β2=np.zeros((M))
for m in range(M):
    β2[m]=pow(β,1/12)**(m-1)
total_β2=np.sum(β2)

a=np.zeros((n,T))   
welfare=np.zeros((n))         
for i in range(n):
    for t in range(T):
        for m in range(M):
            a[i,t]=np.sum(β2[m]*u[i,m,t])
            
        welfare[i]=np.sum(β1[t]*a[i,t])

total_welfare=np.sum(welfare)




#5.- WELFARE GAINS
#5.1. REMOVE SEASONALITY
c_mt_middle_n= np.zeros((n,T))
for i,t in it.product(range(n),range(T)):
    c_mt_middle_n[i,t]=np.log(z[i]*c[i,t])

u_l_n=np.zeros((n,T))
for i,t in it.product(range(n),range(T)):
    u_l_n[i,t]=-κ*z_l[i]*l[i,t]
    
#Total utility
u_n=c_mt_middle_n+u_l_n

#Welfare
a_n=np.zeros((n,T))   
welfare_n=np.zeros((n))         
for i in range(n):
    for t in range(T):
        for m in range(M):
            a_n[i,t]=np.sum(β2[m]*u_n[i,t])
            
        welfare_n[i]=np.sum(β1[t]*a_n[i,t])

total_welfare_n=np.sum(welfare_n)



#5.2. DECOMPOSITION OF WELFARE EFFECTS
#Welfare W(c*, h) (w_nl)
#Total utility
#u_nl=c_mt_middle_n+u_l

#Step 1. Welfare: no-seasonality in consumption, but seasonality in labour
a_nl=np.zeros((n,T))   
welfare_nl=np.zeros((n))         
for i in range(n):
    for t in range(T):
        for m in range(M):
            a_nl[i,t]=np.sum(β2[m]*c_mt_middle_n[i,t]+u_l[i,m,t])
            
        welfare_nl[i]=np.sum(β1[t]*a_nl[i,t])

total_welfare_nl=np.sum(welfare_nl)


#Welfare gain step 1
g_c=np.exp((welfare_nl-welfare)/(total_β2*total_β1)) - 1
g_total_c=np.exp((total_welfare_nl-total_welfare)/(total_β2*total_β1))-1



#Step 2. 
g_l=np.exp((welfare_n-welfare_nl)/(total_β2*total_β1)) - 1
g_total_l=np.exp((total_welfare_n-total_welfare_nl)/(total_β2*total_β1))-1


#Welfare gain without decomposition
g=np.exp((welfare_n-welfare)/(total_β2*total_β1)) - 1
g_total=np.exp((total_welfare_n-total_welfare)/(total_β2*total_β1))-1


array1=np.array([[round(total_welfare,2)],[round(total_welfare_nl,2)], [round(total_welfare_n,2)],[round(g_total_c,2)],[round(g_total_l,2)], [round(g_total,2)]])
table5=pd.DataFrame(array1, index = ['Welfare', 'Welfare Non-Seasonality C','Welfare Non-Seasonality','Welfare gains C', 'Welfare gains L', 'Total Welfare Gains'], columns = [''])
table5
print(table5.to_latex())

#%%From labour
#Dirty calibration
κ=0.05    
labour=np.zeros((n,M,T))
consumption=np.zeros((n,M,T))
for i,m,t in it.product(range(n),range(12),range(T)):
    labour[i,m,t]=z_l[i]*s_labour[m]*l[i,t]
    consumption[i,m,t]=z[i]*np.exp(g_middle[m])*s_m[m]*c[i,t]   
#κ=1/(np.mean(labour)*np.mean(consumption))


#Desutility from labour
u_l=np.zeros((n,M,T))
for i,m,t in it.product(range(n),range(12),range(T)):
    u_l[i,m,t]=-κ*pow(labour[i,m,t],2)/2
    
#Total utility
u=c_mt_middle+u_l




#4.- LIFETIME WELFARE
w=np.zeros((n,M,T))
β1=np.zeros((T))
for t in range(T):
    β1[t]=pow(β,1/12)**(t*12)
total_β1=np.sum(β1)  
  
β2=np.zeros((M))
for m in range(M):
    β2[m]=pow(β,1/12)**(m-1)
total_β2=np.sum(β2)

a=np.zeros((n,T))   
welfare=np.zeros((n))         
for i in range(n):
    for t in range(T):
        for m in range(M):
            a[i,t]=np.sum(β2[m]*u[i,m,t])
            
        welfare[i]=np.sum(β1[t]*a[i,t])

total_welfare=np.sum(welfare)




#5.- WELFARE GAINS
#5.1. REMOVE SEASONALITY
c_mt_middle_n= np.zeros((n,T))
for i,t in it.product(range(n),range(T)):
    c_mt_middle_n[i,t]=np.log(z[i]*c[i,t])

u_l_n=np.zeros((n,T))
for i,t in it.product(range(n),range(T)):
    u_l_n[i,t]=-κ*z_l[i]*l[i,t]
    
#Total utility
u_n=c_mt_middle_n+u_l_n

#Welfare
a_n=np.zeros((n,T))   
welfare_n=np.zeros((n))         
for i in range(n):
    for t in range(T):
        for m in range(M):
            a_n[i,t]=np.sum(β2[m]*u_n[i,t])
            
        welfare_n[i]=np.sum(β1[t]*a_n[i,t])

total_welfare_n=np.sum(welfare_n)



#5.2. DECOMPOSITION OF WELFARE EFFECTS
#Welfare W(c*, h) (w_nl)
#Total utility
#u_nl=c_mt_middle_n+u_l

#Step 1. Welfare: no-seasonality in consumption, but seasonality in labour
a_nl=np.zeros((n,T))   
welfare_nl=np.zeros((n))         
for i in range(n):
    for t in range(T):
        for m in range(M):
            a_nl[i,t]=np.sum(β2[m]*c_mt_middle_n[i,t]+u_l[i,m,t])
            
        welfare_nl[i]=np.sum(β1[t]*a_nl[i,t])

total_welfare_nl=np.sum(welfare_nl)


#Welfare gain step 1
g_c=np.exp((welfare_nl-welfare)/(total_β2*total_β1)) - 1
g_total_c=np.exp((total_welfare_nl-total_welfare)/(total_β2*total_β1))-1



#Step 2. 
g_l=np.exp((welfare_n-welfare_nl)/(total_β2*total_β1)) - 1
g_total_l=np.exp((total_welfare_n-total_welfare_nl)/(total_β2*total_β1))-1


#Welfare gain without decomposition
g=np.exp((welfare_n-welfare)/(total_β2*total_β1)) - 1
g_total=np.exp((total_welfare_n-total_welfare)/(total_β2*total_β1))-1


array1=np.array([[round(total_welfare,2)],[round(total_welfare_nl,2)], [round(total_welfare_n,2)],[round(g_total_c,2)],[round(g_total_l,2)], [round(g_total,2)]])
table5=pd.DataFrame(array1, index = ['Welfare', 'Welfare Non-Seasonality C','Welfare Non-Seasonality','Welfare gains C', 'Welfare gains L', 'Total Welfare Gains'], columns = [''])
table5

plt.figure
plt.subplots_adjust(top=0.8, bottom=0.2, left=0.3, right=2, wspace=0.5)
plt.suptitle('Welfare gains distribution')
bins=50 #Adjust the number of bins
plt.subplot(1,3,1)
plt.hist(g_c,bins, alpha=0.5)
plt.title('Non-seasonality in consumption')

plt.subplot(1,3,2)
plt.hist(g_l,bins, alpha=0.5)
plt.title('Non-seasonality in labour')


plt.subplot(1,3,3)
plt.hist(g,bins, alpha=0.5)
plt.title('Non-seasonality')
plt.show()

#Plot decomposition of welfare gains
plt.figure
plt.subplots_adjust(top=0.8, bottom=0.2, left=0.3, right=2, wspace=0.5)
plt.suptitle('Welfare gains distribution')
bins=50 #Adjust the number of bins
plt.subplot(1,3,1)
plt.hist(g_c,bins, alpha=0.5)
plt.title('Non-seasonality in consumption')

plt.subplot(1,3,2)
plt.hist(g_l,bins, alpha=0.5)
plt.title('Non-seasonality in labour')


plt.subplot(1,3,3)
plt.hist(g,bins, alpha=0.5)
plt.title('Non-seasonality')
plt.show()
    
#Plot decomposition of welfare gains
plt.figure
plt.subplots_adjust(top=0.8, bottom=0.2, left=0.3, right=2, wspace=0.5)
plt.suptitle('Welfare gains distribution')
bins=50 #Adjust the number of bins
plt.subplot(1,3,1)
plt.hist(g_c,bins, alpha=0.5)
plt.title('Non-seasonality in consumption')

plt.subplot(1,3,2)
plt.hist(g_l,bins, alpha=0.5)
plt.title('Non-seasonality in labour')


plt.subplot(1,3,3)
plt.hist(g,bins, alpha=0.5)
plt.title('Non-seasonality')
plt.show()

# NEGATIVE SEASONAL CORRELATION 
#Seasonal risk (only one degree of seasonality)
#Deterministic component (low degree of seasonality of new table1 REARRANGED)
g_l=[1.07	, 1.076,	0.845,0.932, 0.961,0.984,1.001,1.018,1.018,1.018,1.047,1.07]
        
#Stochastic (low of table2)
s_l=[0.142,	0.145,	0.034,	0.043,	0.068,	0.085,	0.094,	0.094,	0.102,	0.119,	0.137,	0.137]
    
#Same code as before
#Build the vectors
eps_l=np.zeros((M))
for m in range(M):
    eps_l[m]=np.random.normal(0,s_l[m],1)

s_labour=np.zeros((M))
for m in range(M):
    s_labour[m]=np.exp(-s_l[m]/2)*np.exp(eps_l[m])

     
#Labour utility
u_l=np.zeros((n,M,T))
for i,m,t in it.product(range(n),range(12),range(T)):
    u_l[i,m,t]=-κ*z_l[i]*s_labour[m]*l[i,t]
    
#Total utility
u=c_mt_middle+u_l


#Welfare
w=np.zeros((n,M,T))
β1=np.zeros((T))
for t in range(T):
    β1[t]=pow(β,1/12)**(t*12)
total_β1=np.sum(β1)  
  
β2=np.zeros((M))
for m in range(M):
    β2[m]=pow(β,1/12)**(m-1)
total_β2=np.sum(β2)

a=np.zeros((n,T))   
welfare=np.zeros((n))         
for i in range(n):
    for t in range(T):
        for m in range(M):
            a[i,t]=np.sum(β2[m]*u[i,m,t])
            
        welfare[i]=np.sum(β1[t]*a[i,t])

total_welfare=np.sum(welfare)

#DECOMPOSITION OF WELFARE EFFECTS
#Welfare W(c*, h) (w_nl)
#Total utility
#u_nl=c_mt_middle_n+u_l

#Step 1. Welfare: no-seasonality in consumption, but seasonality in labour
a_nl=np.zeros((n,T))   
welfare_nl=np.zeros((n))         
for i in range(n):
    for t in range(T):
        for m in range(M):
            a_nl[i,t]=np.sum(β2[m]*c_mt_middle_n[i,t]+u_l[i,m,t])
            
        welfare_nl[i]=np.sum(β1[t]*a_nl[i,t])

total_welfare_nl=np.sum(welfare_nl)


#Welfare gain step 1
g_c=np.exp((welfare_nl-welfare)/(total_β2*total_β1)) - 1
g_total_c=np.exp((total_welfare_nl-total_welfare)/(total_β2*total_β1))-1



#Step 2. 
g_l=np.exp((welfare_n-welfare_nl)/(total_β2*total_β1)) - 1
g_total_l=np.exp((total_welfare_n-total_welfare_nl)/(total_β2*total_β1))-1

#Welfare gain without decomposition
g=np.exp((welfare_n-welfare)/(total_β2*total_β1)) - 1
g_total=np.exp((total_welfare_n-total_welfare)/(total_β2*total_β1))-1


array1=np.array([[round(total_welfare,2)],[round(total_welfare_nl,2)], [round(total_welfare_n,2)],[round(g_total_c,2)],[round(g_total_l,2)], [round(g_total,2)]])
table6=pd.DataFrame(array1, index = ['Welfare', 'Welfare Non-Seasonality C','Welfare Non-Seasonality','Welfare gains C', 'Welfare gains L', 'Total Welfare Gains'], columns = [''])
table6

plt.figure
plt.subplots_adjust(top=0.8, bottom=0.2, left=0.3, right=2, wspace=0.5)
plt.suptitle('Welfare gains distribution')
bins=50 #Adjust the number of bins
plt.subplot(1,3,1)
plt.hist(g_c,bins, alpha=0.5)
plt.title('Non-seasonality in consumption')

plt.subplot(1,3,2)
plt.hist(g_l,bins, alpha=0.5)
plt.title('Non-seasonality in labour')

plt.subplot(1,3,3)
plt.hist(g,bins, alpha=0.5)
plt.title('Non-seasonality')
plt.show()

print(table5.to_latex())
print(table6.to_latex())
