#!/usr/bin/env python
# coding: utf-8

# In[12]:


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# In[16]:


credit= pd.read_csv("testcpns.csv")
credit


# In[17]:


credit.describe


# In[22]:


credit.head


# In[23]:


sns.FacetGrid(credit, hue="diterima", height=7) \
   .map(plt.scatter, "ipk", "toefl") \
   .add_legend()
     


# In[24]:


sns.FacetGrid(credit, hue="diterima", height=7) \
   .map(plt.scatter, "pengalaman_kerja", "toefl") \
   .add_legend()


# In[25]:


sns.FacetGrid(credit, hue="diterima", height=7) \
   .map(plt.scatter, "pengalaman_kerja", "ipk") \
   .add_legend()
     


# In[26]:


import matplotlib.pyplot as plt 
import seaborn as sns

# Using pairplot we'll visualize the data for correlation
sns.pairplot(credit, x_vars=[ 'ipk','pengalaman_kerja','toefl'], 
             y_vars='diterima', height=4, aspect=1, kind='scatter')
plt.show()


# In[19]:


from sklearn import metrics 


# In[ ]:





# In[21]:


X = credit[['toefl', 'ipk','pengalaman_kerja']]
y = credit['diterima']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)

logistic_regression= LogisticRegression()
logistic_regression.fit(X_train,y_train)
y_pred=logistic_regression.predict(X_test)

confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(confusion_matrix, annot=True)

print('Accuracy: ',metrics.accuracy_score(y_test, y_pred))
plt.show()


# In[27]:


candidate = {'toefl': [590,740,680,610,710],
              'ipk': [2,3.07,3.03,2.03,3],
              'pengalaman_kerja': [3,4,6,1,5]
              }
credit = pd.DataFrame(candidate,columns= ['toefl', 'ipk','pengalaman_kerja',])
y_pred=logistic_regression.predict(credit)
print (credit)
print (y_pred)


# In[28]:


pip install tabulate


# In[29]:


from tabulate import tabulate


# In[30]:


candidate = {'toefl': [590,740,680,610,710],
              'ipk': [2,3.07,3.03,2.03,3],
              'pengalaman_kerja': [3,4,6,1,5]
              }
credit = pd.DataFrame(candidate,columns= ['toefl', 'ipk','pengalaman_kerja',])
y_pred=logistic_regression.predict(credit)
print(tabulate(credit.head(14),  headers='keys', tablefmt='fancy_grid'))
print (y_pred)


# In[31]:


print(tabulate(X_test.head(14),  headers='keys', tablefmt='fancy_grid')) 
print (y_pred)


# In[ ]:




