#%% [markdown]
# # Santander Data Science Frameworks


from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from IPython.display import display
import matplotlib.patches as patch
import matplotlib.pyplot as plt
from sklearn.svm import NuSVR
from scipy.stats import norm
from sklearn import svm
import lightgbm as lgb
import xgboost as xgb
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
import time
import glob
import sys
import os
import gc

#%% [markdown]
# # 
train=pd.read_csv('/home/rahul/Desktop/Link to rahul_environment/Projects/Machine_Learning Projects/Santander/train.csv')


#%% [markdown]
train.head(10)
train.shape

#%% [markdown]
train.columns

#%% [markdown]
print(len(train.columns))

#%% [markdown]
print(train.info())

#%% [markdown]
train.describe()

#%% [markdown]
train[train.columns[2:]].mean().plot('hist')
plt.savefig('meanfrequency')
plt.title('Mean frequency')
plt.show

#%% [markdown]
train['target'].value_counts().plot.pie()

#%% [markdown]
sns.countplot(x='target',data=train)

#%% [markdown]
train["var_0"].hist()
plt.savefig('var_0fig')
plt.show()
train["var_81"].hist()
plt.savefig('var_81fig')
plt.show()

#%% [markdown]
sns.distplot(train['target'])


#%% [markdown]
sns.violinplot(x='target',data=train,y='var_81')

#%% [markdown]
# # For checking the null value is present in the columsn or not
train.isnull().sum()

#%% [markdown]
# # Binary Classification
train['target'].unique()

#%% [markdown]
train.head(20)

#%% [markdown]
def check(df,target):
    check=[]
    print('size of the data',df.shape[0])
    for i in [0,1]:
        print('for target',format(i))
        print(df[target].value_counts()[i]/df.shape[0]*100,'%')



#%% [markdown]
check(train,'target')

#%% [markdown]
# # RandomForestClassifier

random=RandomForestClassifier(train)

#%% [markdown]
# # Decision Tree Classifier
decision=DecisionTreeClassifier(train)
