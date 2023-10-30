#!/usr/bin/env python
# coding: utf-8

# # Iris Flower Classification

# #### importing libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# #### Loading the dataset

# In[3]:


df = pd.read_csv('Iris.csv')
df.head()


# In[4]:


#delete id column from the data
df = df.drop(columns=['Id'])
df.head()


# In[5]:


df.describe()


# In[6]:


# Dataset info
df.info()


# In[7]:


# display the number of sample of each class
df['Species'].value_counts()


# #### Preprocessing the dataset

# In[8]:


# checking null value
df.isnull().sum()


# #### Exploratory Data Analysis (EDA)

# In[10]:


df['SepalWidthCm'].hist() #hisogram
plt.show()


# In[11]:


df['SepalLengthCm'].hist() #hisogram
plt.show()


# In[12]:


df['PetalWidthCm'].hist() #hisogram
plt.show()


# In[13]:


df['PetalLengthCm'].hist() #hisogram
plt.show()


# In[22]:


# creating a scatterplot
colors = ['red', 'green', 'blue']
species = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

for i in range(3):
    x = df[df['Species'] == species[i]]
    #print(x)
    plt.scatter(x['SepalLengthCm'], x['SepalWidthCm'], c = colors[i], label=species[i])
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.legend()
plt.show()


# In[23]:


for i in range(3):
    x = df[df['Species'] == species[i]]
    #print(x)
    plt.scatter(x['PetalLengthCm'], x['PetalWidthCm'], c = colors[i], label=species[i])
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.legend()
plt.show()


# In[24]:


for i in range(3):
    x = df[df['Species'] == species[i]]
    #print(x)
    plt.scatter(x['SepalLengthCm'], x['PetalLengthCm'], c = colors[i], label=species[i])
plt.xlabel('Sepal Length')
plt.ylabel('Petal Length')
plt.legend()
plt.show()


# In[25]:


for i in range(3):
    x = df[df['Species'] == species[i]]
    #print(x)
    plt.scatter(x['SepalWidthCm'], x['PetalWidthCm'], c = colors[i], label=species[i])
plt.xlabel('Sepal Width')
plt.ylabel('Petal Width')
plt.legend()
plt.show()


# #### Coorelation Matrix

# In[26]:


# showing correlation coefficients between variables
df.corr()


# In[27]:


# heatmap
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='mako')
plt.show()


# #### Training a Model

# In[28]:


from sklearn.model_selection import train_test_split
X = df.drop(columns=['Species'])
Y = df['Species']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30)


# #### Logistic Regression

# In[29]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


# In[30]:


# training model
model.fit(X_train, y_train)


# In[37]:


# model accuracy
print("Accuracy : ", model.score(X_test, y_test) * 100)


# In[38]:


model.predict([[6.0, 2.2, 4.0, 1.0]])


# #### KNN Model

# In[32]:


from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()


# In[33]:


# training model
model.fit(X_train, y_train)


# In[34]:


# model accuracy
print("Accuracy : ", model.score(X_test, y_test) * 100)


# In[35]:


model.predict([[6.0, 2.2, 4.0, 1.0]])


# You can find this project on <a href="https://github.com/Vyas-Rishabh/Iris-Flower-Classification---Oasis-Infobyte"><b>GitHub.</b></a>
