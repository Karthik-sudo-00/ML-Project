#!/usr/bin/env python
# coding: utf-8

# In[27]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[28]:


pwd


# In[29]:


df1 = pd.read_csv(r"/Users/karthikreddy/Downloads/ds_salaries (1).csv")


# In[30]:


df1.head


# In[31]:


df1.shape


# In[32]:


df1.isnull().sum()


# In[33]:


df1.drop_duplicates()


# In[64]:


df1.dtypes


# In[70]:


from sklearn.preprocessing import LabelEncoder


# In[71]:


m1 = LabelEncoder()
print(m1)


# In[73]:


df1["experience_level"] = m1.fit_transform(df1["experience_level"])
df1["employment_type"] = m1.fit_transform(df1["employment_type"])
df1["job_title"] = m1.fit_transform(df1["job_title"])
df1["salary_currency"] = m1.fit_transform(df1["salary_currency"])
df1["employee_residence"] = m1.fit_transform(df1["employee_residence"])
df1["company_location"] = m1.fit_transform(df1["company_location"])
df1["company_size"] = m1.fit_transform(df1["company_size"])


# In[74]:


df1.head()


# In[75]:


x1 = df1.drop(['salary'],axis=1)
y1 = df1['salary']
print(x1.shape)
print(y1.shape)


# In[76]:


from sklearn.model_selection import train_test_split


# In[77]:


print(x1.shape)
print(0.3*3755)


# In[78]:


x_train,x_test,y_train,y_test = train_test_split(x1,y1,test_size = 0.3)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)



# In[79]:


from sklearn.linear_model import LinearRegression


# In[80]:


m2 = LinearRegression()


# In[81]:


m2.fit(x_train,y_train)


# In[82]:


print('Train Score', m2.score(x_train,y_train))
print('Test Score', m2.score(x_test,y_test))


# In[83]:


ypred_m2 = m2.predict(x_test)


# In[94]:


from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score


# In[95]:


def eval_model(ytest,ypred):
    mse = mean_squared_error(ytest,ypred)
    rmse = np.sqrt(mean_squared_error(ytest,ypred))
    mae = mean_absolute_error(ytest,ypred)
    r2 = r2_score(ytest,ypred)
    print('MSE',mse)
    print('RMSE',rmse)
    print('MAE',mae)
    print('R2_SCORE',r2)


# In[96]:


eval_model(y_test,ypred_m2)


# In[ ]:




