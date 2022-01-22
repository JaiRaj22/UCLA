#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data = pd.read_csv('admission_predict.csv')


# In[3]:


data.head()


# In[4]:


data.info()


# In[5]:


data.isnull().any()


# In[6]:


data = data.rename(columns={'GRE Score':'GRE', 'TOEFL Score':'TOEFL'})


# In[7]:


fig = plt.hist(data['GRE'], rwidth=0.7)
plt.title("distribution of gre scores")
plt.xlabel('gre scores')
plt.ylabel('count')
plt.show()


# In[8]:


fig = plt.hist(data['TOEFL'],rwidth=0.7)
plt.xlabel('TOEFL scores')
plt.ylabel('count')
plt.show()


# In[9]:


fig = plt.hist(data['University Rating'], rwidth=0.7)
plt.title('Distribution of University Rating')
plt.xlabel('University Rating')
plt.ylabel('Count')
plt.show()


# In[10]:


data.drop('Serial No.', axis='columns', inplace=True)


# In[11]:


data


# In[15]:


x = data.iloc[:,:-1].values
y = data.iloc[:,-1].values


# In[16]:


x


# In[17]:


y


# In[18]:


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor


# In[20]:


def find_best_model(X, y):
    models = {
        'linear_regression': {
            'model': LinearRegression(),
            'parameters': {
                'normalize': [True,False]
            }
        },
        'svr': {
            'model': SVR(),
            'parameters': {
                'gamma': ['auto','scale']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'parameters': {
                'criterion': ['mse', 'friedman_mse'],
                'splitter': ['best', 'random']
            }
        },
        
        'random_forest': {
            'model': RandomForestRegressor(criterion='mse'),
            'parameters': {
                'n_estimators': [5,10,15,20]
            }
        },
        
        'knn': {
            'model': KNeighborsRegressor(algorithm='auto'),
            'parameters': {
                'n_neighbors': [2,5,10,20]
            }
        }
    }
    
    scores = []
    for model_name, model_params in models.items():
        gs = GridSearchCV(model_params['model'], model_params['parameters'], cv=5, return_train_score=False)
        gs.fit(x, y)
        scores.append({
            'model': model_name,
            'best_parameters': gs.best_params_,
            'score': gs.best_score_
        })
        
    return pd.DataFrame(scores, columns=['model','best_parameters','score'])
        
find_best_model(x, y)


# In[21]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(LinearRegression(normalize=True), x, y, cv=5)
print(scores)


# In[26]:


avg = round(sum(scores)*100/len(scores),3)
print(avg,"%")


# In[39]:


from sklearn.model_selection import train_test_split
x_train, x_test,y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=7)


# In[40]:


model = LinearRegression(normalize=True)
model.fit(x_train, y_train)
model.score(x_test, y_test)


# In[42]:


print(model.predict([[337, 118, 4, 4.5, 4.5, 9.65, 0]]))


# In[44]:


import pickle


# In[45]:


pickle.dump(model, open('ucla.pkl','wb'))


# In[ ]:




