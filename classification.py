#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Description: This program detects breast cancer from the given set of data
#Category: Classification


# In[2]:


#Import the libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


#load the data

df = pd.read_csv('cancer_data.csv')


# In[4]:


df.head()


# In[5]:


# M --> Malignant, which is harmful, hence cancerous
# B --> Benign, which is not harmful, hence non-cancerous

# In the column names,
# _se --> std. error
# _worst
# Unnamed 32: has empty values


# In[6]:


# Count the number of rows and columns in the dataset
# Each row represents a patient


# In[7]:


df.shape


# In[8]:


# Since last column provides no information, we may have less than 33 features with valuable datapoints

# Count of no. of empty (NaN, NA) values in each column

df.isna().sum()


# In[9]:


# Column Unnamed: 32 is not required
# Hence dropping all columns with all missing values, here only 1 such column

df = df.dropna(axis = 1)

# axis = 0 drop all rows having NA


# In[10]:


df.head(7)


# In[11]:


df.shape # now there are only 32 column!


# In[12]:


# Cleaning the data before classification/detection of cancerous cell

# Getting a count of M-cells and B-cells

df['diagnosis'].value_counts()


# In[13]:


# Visualize the count

sns.countplot(df['diagnosis'], label = 'count')


# In[14]:


# Looking at the datatypes to see which columns need to be transformed (maybe into a number value)

df.dtypes # diagnosis is object datatype --> categorical values


# In[15]:


# Encode the categorical data values

from sklearn.preprocessing import LabelEncoder
LabelEncoder_Y = LabelEncoder()
LabelEncoder_Y.fit_transform(df.iloc[:,1].values) # ':' is for all rows; converts all 'M's =1 and 'B's =0

# df.iloc[:,1].values --> displays 'M' and 'B' values


# In[16]:


# Now to put the data back into my dataset

df.iloc[:,1] = LabelEncoder_Y.fit_transform(df.iloc[:,1].values)


# In[17]:


df.head()


# In[18]:


# Create a pair plot

sns.pairplot(df.iloc[:,1:6]) # does not include index '6'


# In[19]:


sns.pairplot(df.iloc[:,1:6], hue ='diagnosis')


# In[20]:


# Get the correlation of the columns i.e. how a column can influence another column

df.iloc[:,1:12].corr() # Ex. 'perimeter_mean' column has more inluence on 'diagnosis' column

# Value 0 means no influence on the other column


# In[21]:


# To visualize the correlation

sns.heatmap(df.iloc[:,1:12].corr()) # difficult to get exactness, need to put values as well


# In[22]:


# To make it easier to read

plt.figure(figsize=(10,10)) # resizing the figure for better view
sns.heatmap(df.iloc[:,1:12].corr(), annot = True, fmt = '.0%')


# In[23]:


# Split the dataset into independent (X) and dependent (Y) datasets

X = df.iloc[:,2:31].values # 'values' make it an ARRAY; 'X' tells abt features that can detect if the patient has cancer or not
Y = df.iloc[:,1].values    # it tells if the patient gas cancer or not; it has the diagnosis column

type(X)


# In[24]:


type(df) # Datatypes of 'X' and 'df' are different. We changed it into arrays because of the parameters we will
         # be taking in for our model


# In[25]:


# Split the dataset into 75% training and 25% testing

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test =train_test_split(X, Y, test_size = 0.25, random_state = 0)


# In[26]:


# Scale the data (Feature Scaling)
from sklearn.preprocessing import StandardScaler()
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# In[27]:


X_train


# In[28]:


# Create a function for the models to detect cancer

def models(X_train, Y_train):
    
    # Logistic Regression model
    
    from sklearn.linear_model import LogisticRegression
    log = LogisticRegression(random_state=0)
    log.fit(X_train, Y_train)
    
    # Decision Tree
    
    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier(criterion='entropy', random_state=0)
    tree.fit(X_train, Y_train)
    
    # Random Forest
    
    from sklearn.ensemble import RandomForestClassifier
    forest = RandomForestClassifier(n_estimators = 10, random_state=0, criterion='entropy')
    forest.fit(X_train, Y_train)
    
    # Print the model accuracy on the training data
    
    print('[0] Logistic Regression Training Accuracy:', log.score(X_train, Y_train))
    print('[1] Decision Tree Classifier Training Accuracy:', tree.score(X_train, Y_train))
    print('[2] Random Forest Classifier Training Accuracy:', forest.score(X_train, Y_train))
    
    return log, tree, forest


# In[29]:


# Getting all of the models

model = models(X_train, Y_train)


# In[30]:


# Test model accuracy on test data on confusion matrix

from sklearn.metrics import confusion_matrix

for i in range(len(model)):
    print('Model ',i)
    cm = confusion_matrix(Y_test, model[i].predict(X_test))

    TP = cm[0][0]
    TN = cm[1][1]
    FN = cm[1][0]
    FP = cm[0][1]

    # [0,0] True positives (TP)
    # [0,1] False positives (FP)
    # [1,0] False negatives (FN)
    # [1,1] True negatives (TN)

    print(cm) # printing the confusion matrix
    print('Testing Accuracy = ', (TP + TN)/(TP+TN+FP+FN))
    print()


# In[31]:


# Show another way to get metrics of the models

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

for i in range(len(model)):
    print('Model ',i)

    print(classification_report(Y_test, model[i].predict(X_test)))
    print(accuracy_score(Y_test, model[i].predict(X_test)))
    print()


# In[32]:


# Print the prediction of Random Forest Classifier Model

pred = model[2].predict(X_test) # it'll decide whether the patient does or does not have cancer
print(pred)    # values that the model predicts
print()
print(Y_test)  # values present in the test dataset

# columns (say, 11 and last) has different predictions


# In[ ]:




