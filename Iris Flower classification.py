#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_shttp://localhost:8888/notebooks/Iris%20Flower%20classification.ipynb#plit
import seaborn as sns


# In[14]:


df=pd.read_csv(r"D:\data sets\IRIS.csv")
df


# In[15]:


df.info()


# In[16]:


df.isnull().sum()


# In[17]:


sns.pairplot(hue="species",data=df)


# In[18]:


df["species"].value_counts()


# In[12]:


data=df.values
x=data[:,0:4]
y=data[:,4]
print(x)
#print(y)


# In[22]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[28]:


from sklearn.svm import SVC
model_svm=SVC()
model_svm.fit(x_train,y_train)


# In[29]:


prediction1=model_svm.predict(x_test)


# In[31]:


prediction1


# In[44]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,prediction1)*100)
for i in range(len(prediction1)):
    print(y_test[i],prediction1[i])


# In[37]:


#Logistic Regression
from sklearn.linear_model import LogisticRegression


# In[38]:


model_LR=LogisticRegression()
model_LR.fit(x_train,y_train)


# In[42]:


from sklearn.metrics import accuracy_score

# Calculate and print the accuracy score
accuracy = accuracy_score(y_test, prediction2)
print("Accuracy: {:.2f}%".format(accuracy * 100))


# In[45]:


from sklearn.tree import DecisionTreeClassifier
model_DTC=DecisionTreeClassifier()
model_DTC.fit(x_train,y_train)


# In[47]:


prediction3=model_svm.predict(x_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,prediction3))


# In[50]:


#classification report
from sklearn.metrics import classification_report
print(classification_report(y_test,prediction2))


# In[ ]:




