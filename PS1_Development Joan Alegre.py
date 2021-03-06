
import pandas as pd
import numpy as np
import os
os.chdir('C:/Users/Joana/Desktop/Cole/18-19/2.Development/ps1/data')

pd.options.display.float_format = '{:,.2f}'.format

dollars = 2586.89    #https://data.worldbank.org/indicator/PA.NUS.FCRF
#%%
#   QUESTION 1.1:
data = pd.read_csv("ps1_data.csv")
data['inctotal'] = data['inctotal'].fillna(0)
urban = data.loc[data["urban"].isin([1]),["HHID","ctotal","total_W","inctotal"]]
urban.columns = ['hh','urban consumption','urban wealth', 'urban income']
rural = data.loc[data["urban"].isin([0]),["HHID","ctotal","total_W","inctotal"]]
rural.columns = ['hh','rural consumption','rural wealth', 'rural income']
#I do not do only mean, I make the main descriptive statistics with 'describe'.
#Notice that I drop all the Household with NaN consumption, I can accept that for
# income and wealth, but not for consumption.

curban = urban[["urban consumption"]]
curban = curban.dropna()
wurban = urban[["urban wealth"]]
iurban = urban[["urban income"]]
sum_curban = curban.describe()
sum_wurban = wurban.describe()
sum_iurban = iurban.describe()
print(sum_curban.to_latex())
print(sum_wurban.to_latex())
print(sum_iurban.to_latex())


crural = rural[["rural consumption"]]
crural = crural.dropna()
wrural = rural[["rural wealth"]]
irural = rural[["rural income"]]
sum_crural = crural.describe()
sum_wrural = wrural.describe()
sum_irural = irural.describe()
print(sum_crural.to_latex())
print(sum_wrural.to_latex())
print(sum_irural.to_latex())
#%%
# QUESTION 1.2:
#Comparation:
# IMPORTANT NOTE: WE NEED TO SEEE IF SCALES ARE CORRECT, SINCE INCOME IS 
# IN HUNDRED THOUSAND AND CONSUMPTION IN THOUSANDS.

#BINS:
curban.hist(bins = 20)
wurban.hist(bins= 20)
iurban.hist(bins= 20)
crural.hist(bins= 20)
wrural.hist(bins= 20)
irural.hist(bins= 20)

# VARIANCES:
###RURAL.
#Since logarithm of 0 is menus infinite, I convert these values to 0. Variance will be undervaluated.
lncrural = np.log(crural)
varcrural =  np.var(lncrural)
irural = irural.replace(0, 1)
lnirural = np.log(irural)
varirural =  np.var(lnirural)
wrural = wrural.replace(0, 1)
lnwrural = np.log(wrural)
varwrural =  np.var(lnwrural)
### Urban.

lncurban = np.log(curban)
varcurban = np.var(lncurban)
iurban = iurban.replace(0, 1)
lniurban = np.log(iurban)
iurban = lniurban.fillna(0)
variurban = np.var(lniurban)
wurban = wurban.replace(0, 1)
lnwurban = np.log(wurban)
wurban = lniurban.fillna(0)
varwurban = np.var(lnwurban)
#%%
# QUESTION 1.3
#I remove 0 consumption people:
urban = urban.dropna()
rural = rural.dropna()
ax = rural.plot.hist(bins=12, alpha=0.5)
ax = urban.plot.hist(bins=12, alpha=0.5)
#%%
#QUESTION 1.4: LIFECICLE CIW:
import matplotlib.pyplot as plt
age = np.linspace(16,100,10000)
A = data.groupby(by = "age")[["ctotal","total_W","inctotal"]].mean()
A.reset_index(inplace=True)
consumcoef = np.polynomial.polynomial.polyfit(A['age'],A['ctotal'], 1)
con = consumcoef[0]+consumcoef[1]*age # +consumcoef[2]*pow(age,2)
wealthcoef = np.polynomial.polynomial.polyfit(A['age'],A['total_W'], 1)
wealth = wealthcoef[0]+wealthcoef[1]*age # +wealthcoef[2]*pow(age,2)
incomecoef = np.polynomial.polynomial.polyfit(A['age'],A['inctotal'], 1)
income = incomecoef[0]+incomecoef[1]*age # +incomecoef[2]*pow(age,2)

plt.plot(age,con, label = 'consumption over lifecicle')
plt.ylabel('consumption', size = 20)
plt.xlabel('Age', size = 20)
plt.legend()
plt.show()


plt.plot(age,wealth, label = 'wealth over lifecicle')
plt.ylabel('wealth', size = 20)
plt.xlabel('Age', size = 20)
plt.legend()
plt.show()


plt.plot(age,income, label = 'income over lifecicle')
plt.ylabel('income', size = 20)
plt.xlabel('Age', size = 20)
plt.legend()
plt.show()
#%%
#1.5 #RANKING INCOME AND ANALYZING BEHAVIOR OF WEALTH AND CONSUMPTION.
#I will drop income = 0:
#PROBLEM WITH INCOME NEGATIVE, LOW INCOME PART MAKE NO SENSE AT ALL.

data['inctotal'] = data['inctotal'].replace(0,float('NaN'))
data = data[["HHID","inctotal","total_W","ctotal"]]
data = data.dropna()
N = round(len(data)/4)
incomelow = data.sort_values(['inctotal'], ascending=True).head(N) #With this I take the 25% largest income.
#incomelow = incomelow[["HHID","inctotal"]]
incomehight = data.sort_values(['inctotal'], ascending=False).head(N)

plt.scatter(incomelow['inctotal'],incomelow['total_W'], label = 'Wealth')
plt.scatter(incomelow['inctotal'],incomelow['ctotal'], label = 'consumption')
plt.title('Low income')
plt.ylabel('wealth and Consumption', size = 10)
plt.xlabel('Income', size = 20)
plt.legend()
plt.show()


plt.scatter(incomehight['inctotal'],incomehight['total_W'], label = 'Wealth')
plt.scatter(incomehight['inctotal'],incomehight['ctotal'], label = 'consumption')
plt.title('High income')
plt.ylabel('wealth and Consumption', size = 10)
plt.xlabel('Income', size = 20)
plt.legend()
plt.show() 
#Question 2:
#%% LABOUR DATA:
'''METHOD: I took for extensive labour number of people from the household that worked
paid or not paid, for somebody, running a business, on the family farm, I sum every
individual. For example, if one is working in farm and business will count as 2.'''

os.chdir('C:/Users/Joana/Desktop/Cole/18-19/2.Development/ps1/data2/UGA_2013_UNPS_v01_M_STATA8')

'''1 if urban, 0 if rural '''
data = pd.read_stata("gsec1.dta")
dataurban = data[["HHID","urban"]]
dataurban = dataurban.replace(['Urban', 'Rural'], 
                     [1, 0])

'''in Gender, 1 if women, 0 if men '''
data = pd.read_stata("gsec2.dta")
dataage = data[["HHID","h2q8", "h2q3"]]
dataage.columns = [["HHID","Age", "Gender"]]
dataage = dataage.replace(['Female', 'Male'], 
                     [1, 0])
dataage.columns = ['HHID','Age', 'Gender']

data = dataage.merge(dataurban, on="HHID", how="left")

'''Less than primary is 0, primary is 1, and more or equal than secundary is 2 .
Don't know is understood as less than primary.'''
school = pd.read_stata("gsec3.dta")
school = school[["HHID","h3q3"]].dropna()
school.columns = ["HHID","school"]
school = school.replace(['No formal education (*OLD*)', 'Less than primary (*OLD*)', 
                           'Some schooling but not Completed P.1','DK','Completed P.1','Completed P.2'
                           ,'Completed P.3', 'Completed P.4', 'Completed P.5', 'Completed P.6','Completed P.7'
                           ,'Completed J.1', 'Completed J.2', 'Completed J.3', '	Completed primary (*OLD*)'
                           ,'Completed S.1', 'Completed S.2', 'Completed S.3', 'Completed S.4'
                           ,'Completed S.5', 'Completed S.6', 'Completed Post primary Specialized training or Certificate'
                           ,'Completed Post secondary Specialized training or diploma', 'Completed Degree and above'
                           , 'Some secondary', 'Some primary', 'Never attended school', 'Completed O-level (*OLD*)'
                           ,'Completed A-level (*OLD*)', 'Completed University (*OLD*)', 'Don\'t know (*OLD*)', 'Completed primary (*OLD*)', '	Other (Specify) (*OLD*)'], 
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 0, 2, 2, 2, 0, 1, 0])

data1 = data.merge(school, on="HHID", how="left")

'''METHOD: I took for intensive labour hours worked the whole week, 
I add it, and I multiply by 52 '''

data = pd.read_stata("gsec8_1.dta")
data = data.merge(data1, on="HHID", how="left") 
data = data[["HHID","h8q5","h8q7","h8q9","h8q13", "urban"]]
data.columns = ["hh", "paid", "busines", "non-paid", "farm", "urban"]
data = data.replace(['No', 'Yes', 'nan'], 
                     [0, 1, float('NaN')]).dropna()

dataurban = data.loc[data["urban"].isin([1]),["hh", "paid", "busines", "non-paid", "farm"]]
datarural = data.loc[data["urban"].isin([0]),["hh", "paid", "busines", "non-paid", "farm"]]


dataurban = dataurban.groupby(by = "hh")[["paid", "busines", "non-paid", "farm"]].sum()
extensivelaboururban = dataurban.sum(axis=1)
datarural = datarural.groupby(by = "hh")[["paid", "busines", "non-paid", "farm"]].sum()
extensivelabourrural = datarural.sum(axis=1)

data = pd.read_stata("gsec8_1.dta")
data = data.merge(data1, on="HHID", how="left") 
data = data[["HHID","h8q36a","h8q36b","h8q36c","h8q36d","h8q36e","h8q36f","h8q36g", "urban"]].dropna()
dataurban = data.loc[data["urban"].isin([1]),["HHID","h8q36a","h8q36b","h8q36c","h8q36d","h8q36e","h8q36f","h8q36g"]]
datarural = data.loc[data["urban"].isin([0]),["HHID","h8q36a","h8q36b","h8q36c","h8q36d","h8q36e","h8q36f","h8q36g"]]
dataurban = dataurban.groupby(by = "HHID")[["h8q36a","h8q36b","h8q36c","h8q36d","h8q36e","h8q36f","h8q36g"]].sum()
datarural = datarural.groupby(by = "HHID")[["h8q36a","h8q36b","h8q36c","h8q36d","h8q36e","h8q36f","h8q36g"]].sum()
intensivelaboururban = dataurban.sum(axis=1)
intensivelabourrural = datarural.sum(axis=1)

#%% QUESTION 2.1:
#2.1.1 Comparation for rural and urban areas of extensive and intensive margin:

sum_extensivelaboururban = extensivelaboururban.describe()
sum_extensivelabourrural = extensivelabourrural.describe()
sum_intensivelaboururban = intensivelaboururban.describe()
sum_intensivelabourrural = intensivelabourrural.describe()
print(sum_extensivelaboururban.to_latex())
print(sum_extensivelabourrural.to_latex())
print(sum_intensivelaboururban.to_latex())
print(sum_intensivelabourrural.to_latex())

# 2.1.2 Histograms:
#BINS:
extensivelaboururban.hist(bins = 20)
extensivelabourrural.hist(bins= 20)
intensivelaboururban.hist(bins= 20)
intensivelabourrural.hist(bins= 20)
labels = ["extensivelaboururban","extensivelabourrural", "intensivelaboururban", "intensivelabourrural"]
plt.legend(labels)

#VARIANCES OF LOGARITHMS
extensivelaboururban = extensivelaboururban.replace(0, 1)
lnextensivelaboururban = np.log(extensivelaboururban)
varextensivelaboururban=  np.var(lnextensivelaboururban)
extensivelabourrural = extensivelabourrural.replace(0, 1)
lnextensivelabourrural = np.log(extensivelabourrural)
varextensivelabourrural =  np.var(lnextensivelabourrural)
intensivelaboururban = intensivelaboururban.replace(0, 1)
lnintensivelaboururban = np.log(intensivelaboururban)
varintensivelaboururban =  np.var(lnintensivelaboururban)
intensivelabourrural = intensivelabourrural.replace(0, 1)
lnintensivelabourrural= np.log(intensivelabourrural)
varintensivelabourrural =  np.var(lnintensivelabourrural)

# 2.1.4 LIFECICLE
data = pd.read_stata("gsec8_1.dta")
data = data.merge(data1, on="HHID", how="left") 
data = data[["HHID","h8q36a","h8q36b","h8q36c","h8q36d","h8q36e","h8q36f","h8q36g", "urban", "Age"]].dropna()
dataurban = data.loc[data["urban"].isin([1]),["HHID","h8q36a","h8q36b","h8q36c","h8q36d","h8q36e","h8q36f","h8q36g","Age"]]
datarural = data.loc[data["urban"].isin([0]),["HHID","h8q36a","h8q36b","h8q36c","h8q36d","h8q36e","h8q36f","h8q36g", "Age"]]
daturban = dataurban.groupby(by = "HHID")[["h8q36a","h8q36b","h8q36c","h8q36d","h8q36e","h8q36f","h8q36g"]].sum()
datural = datarural.groupby(by = "HHID")[["h8q36a","h8q36b","h8q36c","h8q36d","h8q36e","h8q36f","h8q36g"]].sum()
daturban = daturban.reset_index()
datural = datural.reset_index()
dataurban = daturban.merge(dataurban["Age"], on = "HHID", how = "left")
datarural = datural.merge(datural["Age"], on = "HHID", how = "left")
intensivelaboururban = dataurban.sum(axis=1)
intensivelabourrural = datarural.sum(axis=1)


#%%
#QUESTION 3
#3.1.Consumption, wealth and labor supply agains total income by zone. 
ps1_data =pd.read_csv("data10.csv")
list(ps1_data)
ps1_data=ps1_data[["HHID","urban","year","month","index","inflation","ctotal","total_W","inctotal","wage_total","w_share","worker","ocupation","sex","age","father_educ","father_ocup","writeread","classeduc","region","familysize","wave","region2","region3", "region4","female"]]
ps1_data.to_csv("ps1_data.csv", index=False)

#Region 2
data_2 = ps1_data[['HHID','urban','year','ctotal','inctotal','total_W','worker','region2']]
data_2 = data_2[data_2.region2 != 0]
 
ax = data_2.plot.scatter(x='inctotal', y='ctotal', color='DarkBlue', label='Consumption');
ax1 = data_2.plot.scatter(x='inctotal', y='total_W', color='DarkGreen', label='Wealth');
ax2 = data_2.plot.scatter(x='inctotal', y='worker', color='Yellow', label='Labor');

#Region 3
data_3 = ps1_data[['HHID','urban','year','ctotal','inctotal','total_W','worker','region3']]
data_3 = data_3[data_3.region3 != 0]
 
ax = data_3.plot.scatter(x='inctotal', y='ctotal', color='DarkBlue', label='Consumption');
ax1 = data_3.plot.scatter(x='inctotal', y='total_W', color='DarkGreen', label='Wealth');
ax2 = data_3.plot.scatter(x='inctotal', y='worker', color='Yellow', label='Labor');

#Region 4 
data_4 = ps1_data[['HHID','urban','year','ctotal','inctotal','total_W','worker','region4']]
data_4 = data_4[data_4.region4 != 0]
data_4=data_4.drop([642]) #It looks like a fake household!

ax = data_4.plot.scatter(x='inctotal', y='ctotal', color='DarkBlue', label='Consumption');
ax1 = data_4.plot.scatter(x='inctotal', y='total_W', color='DarkGreen', label='Wealth');
ax2 = data_4.plot.scatter(x='inctotal', y='worker', color='Yellow', label='Labor');

#3.2. Inequality
ax=data_2.hist('ctotal')
ax1=data_2.hist('inctotal')
ax2=data_2.hist('total_W')
ax3=data_2.hist('worker') #It does not look like a labor supply! We need something like number of hours worked...

ax=data_3.hist('ctotal')
ax1=data_3.hist('inctotal')
ax2=data_3.hist('total_W')

ax=data_4.hist('ctotal')
ax1=data_4.hist('inctotal')
ax2=data_4.hist('total_W')

#3.3. Covariance. It is a number! We can get the covariance matrix. Again, the problem with labor supply. 
data_2=data_2[["ctotal","inctotal","total_W","worker"]]
cov2=data_2.cov()
print(cov2.to_latex())

data_3=data_3[["ctotal","inctotal","total_W","worker"]]
cov3=data_3.cov()
print(cov3.to_latex())

data_4=data_4[["ctotal","inctotal","total_W","worker"]]
cov4=data_4.cov()
print(cov4.to_latex())

#3.4. Individual vs. regional income
#Substract per pc consumption from individual consumption
data_2["c_deviation"]=data_2["ctotal"]-data_2["ctotal"].mean()
data_3["c_deviation"]=data_3["ctotal"]-data_3["ctotal"].mean()
data_4["c_deviation"]=data_4["ctotal"]-data_4["ctotal"].mean()
plt.plot(data_2["inctotal"],data_2["c_deviation"])
plt.plot(data_3["inctotal"],data_3["c_deviation"])
plt.plot(data_4["inctotal"],data_4["c_deviation"])

#Income
data_2["i_deviation"]=data_2["inctotal"]-data_2["inctotal"].mean()
data_3["i_deviation"]=data_3["inctotal"]-data_3["inctotal"].mean()
data_4["i_deviation"]=data_4["inctotal"]-data_4["inctotal"].mean()
plt.plot(data_2["inctotal"],data_2["i_deviation"])
plt.plot(data_3["inctotal"],data_3["i_deviation"])
plt.plot(data_4["inctotal"],data_4["i_deviation"])

#Wealth
data_2["W_deviation"]=data_2["total_W"]-data_2["total_W"].mean()
data_3["W_deviation"]=data_3["total_W"]-data_3["total_W"].mean()
data_4["W_deviation"]=data_4["total_W"]-data_4["total_W"].mean()
plt.plot(data_2["inctotal"],data_2["W_deviation"])
plt.plot(data_3["inctotal"],data_3["W_deviation"])
plt.plot(data_4["inctotal"],data_4["W_deviation"])








