#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


# In[5]:


dataset = pd.read_excel("HousePricePredictionData.xlsx")

# Printing first 5 records of the dataset
print(dataset.head(5))




obj = (dataset.dtypes == 'object')
object_cols = list(obj[obj].index)
print("Categorical variables:",len(object_cols))


int_ = (dataset.dtypes == 'int')
num_cols = list(int_[int_].index)
# print("Integer variables:",len(num_cols))

fl = (dataset.dtypes == 'float')
fl_cols = list(fl[fl].index)
# print("Float variables:",len(fl_cols))


# In[18]:


plt.figure(figsize=(12, 6))

# sns.heatmap(dataset.corr(),
#             cmap = 'BrBG',
#             fmt = '.2f',
#             linewidths = 2,
#             annot = True)
#
#
#
unique_values = []

for col in object_cols:
  unique_values.append(dataset[col].unique().size)

plt.figure(figsize=(10,6))
plt.title('No. Unique values of Categorical Features')
plt.xticks(rotation=90)
sns.barplot(x=object_cols,y=unique_values)


# In[16]:


plt.figure(figsize=(18, 36))
plt.title('Categorical Features: Distribution')
plt.xticks(rotation=90)
index = 1


for col in object_cols:
    y = dataset[col].value_counts()
    plt.subplot(11, 4, index)
    plt.xticks(rotation=90)
    sns.barplot(x=list(y.index), y=y)
    index += 1




dataset['SalePrice'] = dataset['SalePrice'].fillna(
  dataset['SalePrice'].mean())




new_dataset = dataset.dropna()
new_dataset.isnull().sum()




from sklearn.preprocessing import OneHotEncoder

s = (new_dataset.dtypes == 'object')
object_cols = list(s[s].index) #boolean indexing



print("Categorical variables:")
print(object_cols)
print('No. of. categorical features: ',
      len(object_cols))


# In[22]:


OH_encoder = OneHotEncoder(sparse=False) # creating an object of OH_encoder

OH_cols = pd.DataFrame(OH_encoder.fit_transform(new_dataset[object_cols]))

OH_cols.index = new_dataset.index

OH_cols.columns = OH_encoder.get_feature_names_out()

print(OH_cols)

df_final = new_dataset.drop(object_cols, axis=1)


df_final = pd.concat([df_final, OH_cols], axis=1)

print(df_final)



from sklearn.model_selection import train_test_split

X = df_final.drop(['SalePrice'], axis=1)
Y = df_final['SalePrice']

# Split the training set into
# training and validation set
X_train, X_valid, Y_train, Y_valid = train_test_split(
    X, Y, train_size=0.8, test_size=0.2, random_state=0)


# In[24]:


# Using SVM for the data

from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import mean_absolute_percentage_error

model_SVR = svm.SVR()
model_SVR.fit(X_train,Y_train)
Y_pred = model_SVR.predict(X_valid)

print(mean_absolute_percentage_error(Y_valid, Y_pred))


# In[25]:


# Using Random Forest for the Data

from sklearn.ensemble import RandomForestRegressor

model_RFR = RandomForestRegressor(n_estimators=10)
model_RFR.fit(X_train, Y_train)
Y_pred = model_RFR.predict(X_valid)

mean_absolute_percentage_error(Y_valid, Y_pred)


# In[26]:


# Using Linear regression for the data

# predicts final independant value based on a number of dependant values

from sklearn.linear_model import LinearRegression

model_LR = LinearRegression()
model_LR.fit(X_train, Y_train)
Y_pred = model_LR.predict(X_valid)

print(mean_absolute_percentage_error(Y_valid, Y_pred))

