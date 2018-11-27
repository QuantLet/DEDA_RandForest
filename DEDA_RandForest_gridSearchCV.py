
# coding: utf-8

# # Random Forest

# Based on:
# - https://www.datacamp.com/community/tutorials/random-forests-classifier-python
# - https://towardsdatascience.com/random-forest-in-python-24d0893d51c0
# - https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
# - https://mathematica.stackexchange.com/questions/98794/how-to-visualize-a-random-forest-classifier

# In[6]:


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
from sklearn.datasets import load_breast_cancer,load_iris,load_wine
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pydot
get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
target = pd.Series(data.target)
print(data.DESCR)


# In[8]:


X_train, X_test, y_train, y_test = train_test_split(
    df, 
    target,
    test_size=0.33, # ratio of data that is used for testing
    random_state=42,
    stratify = target # Keeps the ratio of the labels in train and test the same as in the initial data
)


# ## Grid Search with CV
# Defining paramter grid for GridSearchCV and setting up RandomForestClassifier.

# In[11]:


param_grid = {
    'max_depth': np.arange(1,11,3),
    'n_estimators': [5,15,20,30,40,50,75,100],
    'max_features': np.arange(5,15,5)
}
rf = RandomForestClassifier() #bootstrap=True,oob_score=True)
grid_search = GridSearchCV(
    estimator=rf, # RandomForestClassifier to be optimized
    param_grid=param_grid, # parameter grid
    cv=4, # cross validation split
    n_jobs=-1, # setting for parallization, -1: use all processors
    verbose=1,
    iid=True, # see documentation
    refit=True # Refit estimator using best found parameters
)
grid_search.fit(X_train,y_train)


# Best parameter setting in the grid:

# In[14]:


print(grid_search.best_params_)


# In[16]:



rf_ = grid_search.best_estimator_
print('The train accuracy: %.4f'%rf_.score(X_train,y_train))
print('The test accuracy: %.4f'%rf_.score(X_test,y_test))
pd.DataFrame(confusion_matrix(y_test, rf_.predict(X_test)), 
             index=data.target_names, columns=data.target_names)

