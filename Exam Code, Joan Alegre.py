#%% FINAL EXAM:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import itertools as it
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


#%%
np.random.seed( 13 )
random.seed(13)

## PARAMETERS:
Gamma = 0.5
mean = [1, 1]
var = [[1.416,0], [0, 0.799]]

## 1.1, genetaring logarithms of productivity and capital:
A = np.random.multivariate_normal(mean,var,10000)
lninputs = pd.DataFrame(A, columns = ['lnsi', 'lnki'])

ax = lninputs.plot.hist(bins=1000, alpha=0.5, title ='logarithms, cov=0, gamma=0.5')


inputs = np.exp(lninputs)
inputs.columns = ['si','ki']
ay = inputs.plot.hist(bins=1000, alpha=0.5, title ='Levels, cov=0, gamma=0.5')

## 1.2, computing yi:
yi = inputs['si']*pow(inputs['ki'],Gamma)
yi = pd.DataFrame(yi, columns = ['yi'])

## 1.3, computing efficient ki:
K = sum(inputs['ki'])

#Computing Z from individual productivity.
Z = sum(pow(inputs['si'],1/(1-Gamma)))
inputs['zi'] = pow(inputs['si'],1/(1-Gamma))
inputs['efficientki'] = inputs['zi']*(K/Z) 

#1.4, Comparing ki against eficient ki.
efi = inputs[['ki','efficientki']]


plt.figure
plt.subplots_adjust(top=0.9, bottom=0, left=0.3, right=1.5, wspace=0.5)
plt.suptitle('Efficient vs actual capìtal, cov=0, gamma=0.5')
bins=200 #Adjust the number of bins

plt.subplot(1,2,1)
plt.hist(efi['ki'], bins, alpha=0.5, range=[0, 20], label='actual ki')
plt.hist(efi['efficientki'], bins, alpha=0.5, range=[0, 20], label='efficient ki')
plt.xlabel('capital')
#pyplot.hist(wu, bins, alpha=0.5, label='Wealth')
plt.legend(loc='upper right')

plt.subplot(1,2,2)
plt.hist(efi['efficientki'], bins, alpha=0.5, range = [60,1500], label='efficient ki')
plt.xlabel('capital')
#pyplot.hist(wu, bins, alpha=0.5, label='Wealth')
plt.legend(loc='upper right')

plt.show()

'''We have some observations that are reallyextreme draws from the multinomial, some people
have more than 200 unities of productivity si, some of them more than thousands, this is
purely fruit of statistic probabilities, but this is the main reason of why a lot of people
is recieving almost 0 capital in the efficient escenario. It is mainly because of the extreme
draws of productivity. '''

#1.5, Computing output gains:
yi['yeff'] =  inputs['si']*pow(inputs['efficientki'],Gamma)
Yeff = sum(yi['yeff'])
Y = sum(yi['yi'])
gains = Yeff/Y


#1.6 REMAKING EVERYTHING FOR COVARIANCES 0:
mean = [1, 1]
var = [[1.416,0.25], [0.25, 0.799]]

## 1.1, genetaring logarithms of productivity and capital:
A = np.random.multivariate_normal(mean,var,10000)
lninputs = pd.DataFrame(A, columns = ['lnsi', 'lnki'])

ax = lninputs.plot.hist(bins=1000, alpha=0.5, title ='logarithms, cov=0.25, gamma=0.5')


inputs = np.exp(lninputs)
inputs.columns = ['si','ki']
ay = inputs.plot.hist(bins=1000, alpha=0.5, title ='Levels, cov=0.25, gamma=0.5')

## 1.2, computing yi:
yi = inputs['si']*pow(inputs['ki'],Gamma)
yi = pd.DataFrame(yi, columns = ['yi'])

## 1.3, computing efficient ki:
K = sum(inputs['ki'])

#Computing Z from individual productivity.
Z = sum(pow(inputs['si'],1/(1-Gamma)))
inputs['zi'] = pow(inputs['si'],1/(1-Gamma))
inputs['efficientki'] = inputs['zi']*(K/Z) 

#1.4, Comparing ki against eficient ki.
efi = inputs[['ki','efficientki']]

plt.figure
plt.subplots_adjust(top=0.9, bottom=0, left=0.3, right=1.5, wspace=0.5)
plt.suptitle('Efficient vs actual capìtal covariance 0.25, gamma=0.5')
bins=200 #Adjust the number of bins

plt.subplot(1,2,1)
plt.hist(efi['ki'], bins, alpha=0.5, range=[0, 20], label='actual ki')
plt.hist(efi['efficientki'], bins, alpha=0.5, range=[0, 20], label='efficient ki')
plt.xlabel('capital')
#pyplot.hist(wu, bins, alpha=0.5, label='Wealth')
plt.legend(loc='upper right')

plt.subplot(1,2,2)
plt.hist(efi['efficientki'], bins, alpha=0.5, range = [60,1500], label='efficient ki')
plt.xlabel('capital')
#pyplot.hist(wu, bins, alpha=0.5, label='Wealth')
plt.legend(loc='upper right')

plt.show()

#1.5, Computing output gains:
yi['yeff'] =  inputs['si']*pow(inputs['efficientki'],Gamma)
Yeff = sum(yi['yeff'])
Y = sum(yi['yi'])
gains = Yeff/Y

#%% QUESTION 2, REMAKING EVERYTHING FOR GAMMA 0.8:

np.random.seed( 13 )
random.seed(13)

## PARAMETERS:
Gamma = 0.8
mean = [1, 1]
var = [[1.416,0], [0, 0.799]]

## 2.1, genetaring logarithms of productivity and capital:
A = np.random.multivariate_normal(mean,var,10000)
lninputs = pd.DataFrame(A, columns = ['lnsi', 'lnki'])

ax = lninputs.plot.hist(bins=1000, alpha=0.5, title ='logarithms, cov=0, gamma=0.8')


inputs = np.exp(lninputs)
inputs.columns = ['si','ki']
ay = inputs.plot.hist(bins=1000, alpha=0.5, title ='Levels, cov=0, gamma=0.8')

## 2.2, computing yi:
yi = inputs['si']*pow(inputs['ki'],Gamma)
yi = pd.DataFrame(yi, columns = ['yi'])

## 2.3, computing efficient ki:
K = sum(inputs['ki'])

#Computing Z from individual productivity.
Z = sum(pow(inputs['si'],1/(1-Gamma)))
inputs['zi'] = pow(inputs['si'],1/(1-Gamma))
inputs['efficientki'] = inputs['zi']*(K/Z) 

#2.4, Comparing ki against eficient ki.
efi = inputs[['ki','efficientki']]

plt.figure
plt.subplots_adjust(top=0.9, bottom=0, left=0.3, right=1.5, wspace=0.5)
plt.suptitle('Efficient vs actual capìtal gamma 0.8')
bins=200 #Adjust the number of bins

plt.subplot(1,2,1)
plt.hist(efi['ki'], bins, alpha=0.5, range=[0, 20], label='actual ki')
plt.hist(efi['efficientki'], bins, alpha=0.5, range=[0, 20], label='efficient ki')
plt.xlabel('capital')
#pyplot.hist(wu, bins, alpha=0.5, label='Wealth')
plt.legend(loc='upper right')

plt.subplot(1,2,2)
plt.hist(efi['efficientki'], bins, alpha=0.5, range = [60,1500], label='efficient ki')
plt.xlabel('capital')
#pyplot.hist(wu, bins, alpha=0.5, label='Wealth')
plt.legend(loc='upper right')

plt.show()

#2.5, Computing output gains:
yi['yeff'] =  inputs['si']*pow(inputs['efficientki'],Gamma)
Yeff = sum(yi['yeff'])
Y = sum(yi['yi'])
gains = Yeff/Y


#2.6 REMAKING EVERYTHING FOR COVARIANCES 0.25:
mean = [1, 1]
var = [[1.416,0.75], [0.75, 0.799]]

## 2.1, genetaring logarithms of productivity and capital:
A = np.random.multivariate_normal(mean,var,10000)
lninputs = pd.DataFrame(A, columns = ['lnsi', 'lnki'])

ax = lninputs.plot.hist(bins=1000, alpha=0.5, title ='logarithms, cov=0.25, gamma=0.8')


inputs = np.exp(lninputs)
inputs.columns = ['si','ki']
ay = inputs.plot.hist(bins=1000, alpha=0.5, title ='Levels, cov=0.25, gamma=0.8')

## 2.2, computing yi:
yi = inputs['si']*pow(inputs['ki'],Gamma)
yi = pd.DataFrame(yi, columns = ['yi'])

## 2.3, computing efficient ki:
K = sum(inputs['ki'])

#Computing Z from individual productivity.
Z = sum(pow(inputs['si'],1/(1-Gamma)))
inputs['zi'] = pow(inputs['si'],1/(1-Gamma))
inputs['efficientki'] = inputs['zi']*(K/Z) 

#2.4, Comparing ki against eficient ki.
efi = inputs[['ki','efficientki']]

plt.figure
plt.subplots_adjust(top=0.9, bottom=0, left=0.3, right=1.5, wspace=0.5)
plt.suptitle('Efficient vs actual capìtal covariance 0.25, Gamma 0.8')
bins=200 #Adjust the number of bins

plt.subplot(1,2,1)
plt.hist(efi['ki'], bins, alpha=0.5, range=[0, 20], label='actual ki')
plt.hist(efi['efficientki'], bins, alpha=0.5, range=[0, 20], label='efficient ki')
plt.xlabel('capital')
plt.legend(loc='upper right')

plt.subplot(1,2,2)
plt.hist(efi['efficientki'], bins, alpha=0.5, range = [60,1500], label='efficient ki')
plt.xlabel('capital')
plt.legend(loc='upper right')

plt.show()

#2.5, Computing output gains:
yi['yeff'] =  inputs['si']*pow(inputs['efficientki'],Gamma)
Yeff = sum(yi['yeff'])
Y = sum(yi['yi'])
gains = Yeff/Y

#%% QUESTION 3, RANDOM SAMPLE:
np.random.seed( 13 )
random.seed(13)

## PARAMETERS:
Gamma = 0.5
mean = [1, 1]
var = [[1.416,0], [0, 0.799]]

##  genetaring logarithms of productivity and capital:
A = np.random.multivariate_normal(mean,var,10000)
lninputs = pd.DataFrame(A, columns = ['lnsi', 'lnki'])

input1 = np.exp(lninputs)
input1.columns = ['si','ki']

## computing yi:
yi1 = input1['si']*pow(input1['ki'],Gamma)
yi1 = pd.DataFrame(yi, columns = ['yi'])

# Sampling:
gains = np.zeros(100)

for i in range(100):
    inputs = input1.sample(10)
    ##computing efficient ki:
    K = sum(inputs['ki'])
    
    yi = inputs['si']*pow(inputs['ki'],Gamma)
    yi = pd.DataFrame(yi, columns = ['yi'])
    
    #Computing Z from individual productivity.
    Z = sum(pow(inputs['si'],1/(1-Gamma)))
    inputs['zi'] = pow(inputs['si'],1/(1-Gamma))
    inputs['efficientki'] = inputs['zi']*(K/Z) 
    
    #Comparing ki against eficient ki.
    efi = inputs[['ki','efficientki']]
    
    #Computing output gains:
    yi['yeff'] =  inputs['si']*pow(inputs['efficientki'],Gamma)
    Yeff = sum(yi['yeff'])
    Y = sum(yi['yi'])
    gains[i] = Yeff/Y

plt.figure
plt.subplots_adjust(top=0.9, bottom=0, left=0.3, right=1.5, wspace=0.5)
plt.suptitle('Distribution gains, sampling cov=0, Gamma=0.5')
bins=50 #Adjust the number of bins

plt.subplot(1,1,1)
plt.hist(gains, bins, alpha=0.5, label='Gain')
#pyplot.hist(wu, bins, alpha=0.5, label='Wealth')
plt.legend(loc='upper right')

plt.show()

#%% QUESTION 4: Endogenous prpoductivity.

np.random.seed( 13 )
random.seed(13)


## PARAMETERS:
alpha= 0.5
Gamma = 0.5
mean = [1, 1]
var = [[1.416,0], [0, 0.799]]

## 1.1, genetaring logarithms of productivity and capital:
A = np.random.multivariate_normal(mean,var,10000)
lninputs = pd.DataFrame(A, columns = ['lnsi', 'lnki'])

ax = lninputs.plot.hist(bins=1000, alpha=0.5, title ='logarithms, cov=0, gamma=0.5')


inputs = np.exp(lninputs)
inputs.columns = ['si','ki']
ay = inputs.plot.hist(bins=1000, alpha=0.5, title ='Levels, cov=0, gamma=0.5')

## 1.2, computing yi:
yi = inputs['si']*pow(inputs['ki'],Gamma)
yi = pd.DataFrame(yi, columns = ['yi'])

## 1.3, computing efficient ki:
K = sum(inputs['ki'])

#Computing Z from individual productivity.
'''The only thing we change with respect to the other problem is the computation of Z
Z now is what in the pdf we recall. '''

inputs['zi'] = pow((1-alpha)/inputs['si']+Gamma*inputs['si'],(1/(1-Gamma)))
Z = sum(pow((1-alpha)/inputs['si']+Gamma*inputs['si'],(1/(1-Gamma))))

inputs['efficientki'] = inputs['zi']*(K/Z) 


#1.4, Comparing ki against eficient ki.

efi = inputs[['ki','efficientki']]


plt.figure
plt.subplots_adjust(top=0.9, bottom=0, left=0.3, right=1.5, wspace=0.5)
plt.suptitle('Efficient vs actual capìtal, cov=0, gamma=0.5')
bins=200 #Adjust the number of bins

plt.subplot(1,2,1)
plt.hist(efi['ki'], bins, alpha=0.5, range=[0, 20], label='actual ki')
plt.hist(efi['efficientki'], bins, alpha=0.5, range=[0, 20], label='efficient ki')
plt.xlabel('capital')
#pyplot.hist(wu, bins, alpha=0.5, label='Wealth')
plt.legend(loc='upper right')

plt.subplot(1,2,2)
plt.hist(efi['efficientki'], bins, alpha=0.5, range = [60,1500], label='efficient ki')
plt.xlabel('capital')
#pyplot.hist(wu, bins, alpha=0.5, label='Wealth')
plt.legend(loc='upper right')

plt.show()

'''We have some observations that are reallyextreme draws from the multinomial, some people
have more than 200 unities of productivity si, some of them more than thousands, this is
purely fruit of statistic probabilities, but this is the main reason of why a lot of people
is recieving almost 0 capital in the efficient escenario. It is mainly because of the extreme
draws of productivity. '''

#1.5, Computing output gains:
yi['yeff'] =  inputs['si']*pow(inputs['efficientki'],Gamma)
Yeff = sum(yi['yeff'])
Y = sum(yi['yi'])
gains = Yeff/Y




