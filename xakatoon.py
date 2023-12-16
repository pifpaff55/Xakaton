#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import json
import numpy as np


# In[4]:


data_val = pd.read_json(r"D:\downloads\val.json").transpose()
data_val.head(5)


# In[5]:


data_test = pd.read_json(r"D:\downloads\test.json").transpose()
data_test.head(5)


# In[ ]:


data_train = pd.read_json(r"D:\downloads\train.json").transpose()
data_train.head(5)


# In[ ]:


with open('general_categories_mapping.json', 'r', encoding='utf-8') as f:
    general_categories_mapping = json.load(f)
data_general_categories_mapping = pd.DataFrame(general_categories_mapping.items())
data_general_categories_mapping.rename(columns={0: "article",1: "name"}, inplace=True)
data_general_categories_mapping.head(5)


# In[ ]:


#pd.options.mode.chained_assignment = None


# In[ ]:


data_val.loc[data_val['target'] == 'female', 'target'] = 0
data_val.loc[data_val['target'] == 'male', 'target'] = 1


# In[ ]:


data_train.loc[data_train['target'] == 'female', 'target'] = 0
data_train.loc[data_train['target'] == 'male', 'target'] = 1


# In[ ]:


data_val.describe()


# In[ ]:


data = data_val.copy()


# In[ ]:


data = data.reset_index()


# In[ ]:



data.rename(columns={"index": "ID"}, inplace=True)


# In[ ]:


data.head(5)


# In[ ]:


features_name = []
for i in data['features'].map(lambda x: x.keys()):
 for j in i:
    if j not in features_name:
        features_name.append(j)
print(features_name)


# In[ ]:



for i in features_name:
    data.loc[:, i] = np.nan


# In[ ]:


for j in range(data.shape[0]):
    for i in data.loc[j,'features'].keys():
        data[i][j] = data.loc[j,'features'][i]


# In[ ]:


data = data.drop(columns = 'features')


# In[ ]:


data.head(5)


# In[ ]:




