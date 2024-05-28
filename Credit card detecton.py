#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


# In[2]:


df=pd.read_csv(r"D:\data sets\creditcard.csv")
df


# In[3]:


df.info()


# In[4]:


#count the missing values
df.isnull().sum()


# In[5]:


#distribution of legit transaction & fradulunt transaction
df["Class"].value_counts()


# 0-----normal transaction
# 1-----Fradulant transaction
# 

# In[6]:


#separating normal transaction and fradalant transaction for analysis
legit=df[df["Class"]==0]
fraud=df[df["Class"]==1]


# In[7]:


print(legit.shape)
print(fraud.shape)


# In[8]:


#statistical process of data
legit["Amount"].describe()


# In[9]:


fraud["Amount"].describe()


# In[10]:


#campare the values for both transaction
df.groupby("Class").mean()


# # Under sampling
Build a sample dataset containing similar distribution of normal transactions and fradulent transactionNumber of fraudelent transactions=492
# In[11]:


legit_sample=legit.sample(n=492)


# In[12]:


#Concatinating two data frames
new_data=pd.concat([legit_sample,fraud],axis=0)


# In[13]:


new_data


# In[14]:


new_data.head()


# In[15]:


new_data.tail()


# In[16]:


new_data["Class"].value_counts()


# In[17]:


new_data.groupby("Class").mean()


# In[18]:


#Splitting the data into Features & Target
x=new_data.drop(columns="Class",axis=1)
y=new_data["Class"]


# In[19]:


print(x)


# In[20]:


print(y)


# In[21]:


print(x.shape,x_train.shape,y_train.shape)


# In[ ]:


#split the data into training data and testing data
x_train,x_test,y_train,x_test=train_test_split(x,y,test_size=0.2,stratify=y,random_state=2)


# In[ ]:


#LogisticRegression
model=LogisticRegression()


# In[ ]:


#trainig logisting regression model with training data
model.fit(x_train,y_train)


# In[ ]:


#accuracy on training data
x_train_prediction=model.predict(x_train)
training_data_accuracy=accuracy_score(x_train_prediction,y_train)


# In[ ]:




