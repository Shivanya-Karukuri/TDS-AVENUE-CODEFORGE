#!/usr/bin/env python
# coding: utf-8

# In[54]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv(r"D:\data sets\tested.csv")


# In[55]:


df


# In[56]:


df.isnull().sum()


# In[57]:


df.describe()
sns.heatmap(df.corr(),cmap='YlGnBu')


# In[58]:


df=df.drop(columns="Cabin",axis=1)
df


# In[59]:


df["Age"].fillna(df["Age"].mean(),inplace=True)


# In[61]:


df.isnull().sum()


# In[62]:


print(df["Embarked"].mode())


# In[65]:


df["Fare"].fillna(df["Fare"].mean(),inplace=True)


# In[66]:


df.isnull().sum()


# In[67]:


df.describe()


# In[68]:


df["Survived"].value_counts()


# In[69]:


sns.countplot("Survived",data=df)


# In[70]:


df["Sex"].value_counts()


# In[71]:


sns.countplot("Sex",data=df)
plt.show()


# In[72]:


sns.countplot("Sex",hue="Survived",data=df)


# In[73]:


sns.countplot("Pclass",data=df)


# In[74]:


sns.countplot("Pclass",hue="Survived",data=df)


# In[75]:


df["Sex"].value_counts()


# In[76]:


df.replace({"Sex":{"male":0,"female":1},"Embarked":{"Q":0,"S":1,"C":2}},inplace=True)


# In[77]:


x=df.drop(["PassengerId","Name","Ticket","Survived"],axis=1)
y=df["Survived"]


# In[78]:


print(x)


# In[79]:


print(y)


# In[80]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)


# In[81]:


print(x_train.shape,x_test.shape,y_train.shape)


# In[82]:


from sklearn.linear_model import LogisticRegression


# In[83]:


model=LogisticRegression()


# In[84]:


model.fit(x_train,y_train)


# In[86]:


from sklearn.metrics import accuracy_score


# In[87]:


x_train_predict=model.predict(x_train)


# In[88]:


x_train_predict


# In[90]:


training_data_accuracy=accuracy_score(y_train,x_train_predict)


# In[91]:


training_data_accuracy


# In[93]:


x_test_predict=model.predict(x_test)


# In[94]:


x_test_predict


# In[97]:


testing_data_accuracy=accuracy_score(y_test,x_test_predict)


# In[98]:


testing_data_accuracy


# In[ ]:




