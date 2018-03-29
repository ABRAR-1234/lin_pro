
# coding: utf-8

# #LINEAR REGRESSION MODEL
# 

# ##IMPORT LIBRARIES

# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plot
import pandas as pd
import numpy as np
from sklearn import datasets,linear_model
from sklearn.metrics import mean_squared_error,r2_score


# #LOAD DATASET

# In[7]:


diabetes = datasets.load_diabetes()
print(diabetes)


# In[8]:


diabetes_X = diabetes.data[:,np.newaxis,2]
print(diabetes_X )


# SPLIT TRAINING SET,TEST SET 

# In[9]:


diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]


# In[10]:


diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]


# #INCLUDE LINEAR MODEL FOR LINEAR REGRESSION

# In[12]:


regr = linear_model.LinearRegression()


# #TO FIT TRAINING SETS

# In[13]:


regr.fit(diabetes_X_train, diabetes_y_train)


# In[21]:


print('Coefficients: \n', regr.coef_)


# In[15]:


diabetes_y_pred = regr.predict(diabetes_X_test)
print(diabetes_y_pred)


# In[16]:


print("Mean squared error: %.2f"
      % mean_squared_error(diabetes_y_test, diabetes_y_pred))


# In[17]:


print('Variance score: %.2f' % r2_score(diabetes_y_test, diabetes_y_pred))


# #PLOT THE TEST,TRAINING IN LINEAR 

# In[18]:


plot.scatter(diabetes_X_test, diabetes_y_test,  color='black')
plot.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)
plot.xticks(())
plot.yticks(())
plot.show()

