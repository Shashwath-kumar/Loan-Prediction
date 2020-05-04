#!/usr/bin/env python
# coding: utf-8

# In[219]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn import preprocessing
import scipy.stats as stats
from statistics import mean,stdev
from statsmodels.graphics.gofplots import qqplot
import math


# In[220]:



data =pd.read_csv('train.csv')
data.shape


# In[221]:


data.head(7)


# In[222]:


data.isnull().sum()


# In[223]:


data.loc[23:31,:]


# In[224]:


data['Gender'].fillna('Unknown',inplace =True)
data['Married'].fillna('Unknown',inplace = True)
data['Dependents'].fillna(data['Dependents'].mode()[0],inplace = True)
data['Self_Employed'].fillna(data['Self_Employed'].mode()[0],inplace = True)
data['Credit_History'].fillna(0.5,inplace = True)
data['LoanAmount'].fillna(data.LoanAmount.mean(),inplace = True)                  #data.LoanAmount.mean() = 128
data['Loan_Amount_Term'].fillna(360,inplace = True) 


# In[225]:


data =data.drop('Loan_ID',axis =1)


# In[226]:


data.loc[23:31,:]


# In[227]:


replace_dict = {'Y':1, 'N':0}
data.Loan_Status = data.Loan_Status.replace(replace_dict)


# In[228]:


data['TotalIncome'] = data['ApplicantIncome'] + data['CoapplicantIncome']


# In[229]:


data.describe()


# In[230]:


plt.figure(0)
sns.distplot(data.TotalIncome)
plt.figure(1)
sns.distplot(data.LoanAmount)
plt.show()


# In[231]:


#data.LoanAmount = (data.LoanAmount - data.LoanAmount.min())/(data.LoanAmount.max() - data.LoanAmount.min())


# In[232]:


#data.TotalIncome = (data.TotalIncome - data.TotalIncome.min())/(data.TotalIncome.max() - data.TotalIncome.min())


# In[233]:


data.LoanAmount = preprocessing.scale(np.log(data.LoanAmount))


# In[234]:


data['TotalIncome'] = preprocessing.scale(np.log(np.log(data.TotalIncome)))


# In[235]:


data.TotalIncome.mean()


# In[236]:



sns.distplot(data.TotalIncome) #kde = False, color = 'b', hist_kws={'alpha': 0.9})


# In[237]:


qqplot(data.TotalIncome,dist ='norm',line='45')
plt.show()


# In[238]:


corr = data.select_dtypes(include = ['float64', 'int64']).loc[:,['ApplicantIncome','CoapplicantIncome','LoanAmount','TotalIncome']].corr()
plt.figure(figsize=(7, 7))
sns.heatmap(corr, vmax=1, square=True)
plt.show()


# In[239]:


data.corr(method = 'pearson')


# In[256]:



plt.figure(figsize = (12, 6))
sns.boxplot(y = 'Property_Area', x = 'LoanAmount',  data = data, orient = 'h')


# In[241]:


plt.pie(data.Education.value_counts(),labels=['Graduate','Undergrad'],autopct='%1.2f%%')
#plt.xlabel('Education')
plt.show()


# In[242]:


gb = data.groupby(by=["Education", "Loan_Status"])
gbs = gb.size()
gbs


# In[243]:


plt.figure(0)
plt.pie([gbs[0],gbs[1]],autopct='%1.2f%%')
plt.xlabel('Graduate')
plt.figure(1)
plt.pie([gbs[2],gbs[3]],autopct='%1.2f%%')
plt.xlabel('Not Graduate')

plt.show()


# In[244]:


plt.pie(data.Credit_History.value_counts(),labels=['Yes','No','Unknown'],autopct='%1.2f%%')
#plt.xlabel('Credit History')
plt.show()


# In[254]:


gb = data.groupby(by=["Credit_History", "Loan_Status"])
gbs = gb.size().tolist()

plt.figure(0)
plt.pie([gbs[1],gbs[0]],autopct='%1.2f%%')
plt.xlabel('Good Credit History')
plt.figure(1)
plt.pie([gbs[5],gbs[4]],autopct='%1.2f%%')
plt.xlabel('Bad Credit History')

plt.show()


# In[246]:


sns.distplot(data.LoanAmount)


# In[247]:


plt.scatter(data.TotalIncome,data.LoanAmount,color ='b')


# In[248]:


def estimate_coef(x, y): 
    n = np.size(x) 
  
    m_x, m_y = np.mean(x), np.mean(y) 
  
    numa = np.sum(y*x) - n*m_y*m_x 
    numb = np.sum(x*x) - n*m_x*m_x 
  
    b_1 = numa / numb 
    b_0 = m_y - b_1*m_x 
  
    return(b_0, b_1) 

def plot_regression_line(x, y, b): 
    plt.scatter(x, y, color = "b", 
               marker = "o", s = 30) 
  
    y_pred = b[0] + b[1]*x 
  
    plt.plot(x, y_pred, color = "r") 
    plt.xlabel('Total Income') 
    plt.ylabel('Loan Amount') 
  
    plt.show() 


# In[249]:


b = estimate_coef(data.TotalIncome,data.LoanAmount)

plot_regression_line(data.TotalIncome,data.LoanAmount,b)
b


# In[250]:


from scipy.stats import pearsonr
data1 = data.Loan_Status
data2 = data.LoanAmount
stat, p = pearsonr(data1, data2)
print('stat=',stat,', p=',p)
if p > 0.05:
    print('Failed to reject H0')
else:
    print('Reject H0')


# In[251]:


from scipy.stats import pearsonr
data1 = data.CoapplicantIncome
data2 = data.LoanAmount
stat, p = pearsonr(data1, data2)
print('stat=',stat,', p=',p)
if p > 0.05:
    print('Failed to reject H0')
else:
    print('Reject H0')


# In[252]:


data.Credit_History.value_counts()


# In[253]:


from scipy.stats import pearsonr
data1 = data.Credit_History
data2 = data.Loan_Status
stat, p = pearsonr(data1, data2)
print('stat=',stat,', p=',p)
if p > 0.05:
    print('Failed to reject H0')
else:
    print('Reject H0')


# In[ ]:




