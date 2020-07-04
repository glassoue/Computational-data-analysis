#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 16:23:20 2020

@author: ghassen97
"""

# To embed plots in the notebooks
import matplotlib.pyplot as plt

import numpy as np # numpy library
import scipy . linalg as lng # linear algebra from scipy library
from scipy . spatial import distance # load distance function
from sklearn import preprocessing as preproc # load preprocessing function

# seaborn can be used to "prettify" default matplotlib plots by importing and setting as default
import seaborn as sns
sns.set() # Set searborn as default

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, Lars


import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from random import sample
from random import choices
from sklearn import preprocessing
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import enet_path
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
#from labellines import labelLine, labelLines
from sklearn.linear_model import LassoCV
from sklearn.linear_model import lars_path
from sklearn.linear_model import lasso_path
import seaborn as sns
from scipy import stats
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import random as sparse_random
from sklearn.random_projection import sparse_random_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.ensemble import BaggingRegressor
from sklearn.datasets import make_regression
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import cross_val_score

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from scipy import stats
from sklearn import model_selection
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import scipy.linalg as lng # linear algebra from scipy library
import sklearn.linear_model as lm




#%% 
'''Load Dataset '''
diabetPath = '/home/ghassen97/Desktop/S8/computional analysis/case/Case Data/case1Data.txt'
df = pd.read_csv('/home/ghassen97/Desktop/S8/computional analysis/case/Case Data/case1Data.txt')

''' split numerical and categorical data '''
# Feature columns holding numerical data
columns_num = df.columns[1:96]

# Feature columns holding categorical data
columns_cat = df.columns[96:]

'''Handling data issues '''

def get_most_common_value(column):
    return df[column].value_counts().head(1).index.values[0]

def replace_nan(x, replace_value):
    if pd.isnull(x):
        return replace_value
    else:
        return x

#%% 
''' Change NaN with mean '''
df = df.replace(' NaN', np.nan)
# Categorical nan's: replace the missing entries with the most frequent one.
for i in range(len(columns_cat)):
    most_common_value = get_most_common_value(columns_cat[i])
    df[columns_cat[i]] = df[columns_cat[i]].map(lambda x: replace_nan(x, most_common_value))
#%% explaining onehotencoding
before_1hot = df.iloc[:,100]
before_1hot = before_1hot[:25]
#%%
'''1 out of K  '''
#Use one-hot encoding to transform categorical data
one_hot_encodings = {}
for i in range(len(columns_cat)):
    df_dummies = pd.get_dummies(df[columns_cat[i]], prefix = columns_cat[i])
    one_hot_encodings[columns_cat[i]] = df_dummies

# Drop original categorical columns
df = df.drop(columns=list(columns_cat))

# Append the one-hot encodings of the categorical columns
for _, value in one_hot_encodings.items():
    df = pd.concat([df, value], axis=1)
#%% explaining onehotencoding
after_1hot = df.iloc[:,116:]
#after_1hot = after_1hot []
#%%
''' numpy array data '''
data=df.to_numpy() 
X=data[:,1:]
Y=data[:,0]
y=data[:,0]
n,p=X.shape

#%% Essay with ridge from week2 ex1 
# here NaN are replaced by mean value
[n, p] = np.shape(X)


lambda_values = np.logspace(-100, 100, 10000)

K = 5
kf = model_selection.KFold(n_splits=K,shuffle=True)
split_X = kf.split(X)

MSE = np.zeros((len(lambda_values),K))
betas = np.zeros((len(lambda_values), p))

i = 0
for train_index, test_index in kf.split(X):
    #print(i)
    # extract training and test set for current CV fold
    X_train, y_train = X[train_index,:], y[train_index]
    X_test, y_test = X[test_index,:], y[test_index]
    
    
    #centralize data
    # use mean from trin dataset
    X_train_mean = np.mean(X_train)
    y_train_mean = np.mean(y_train)
    
    X_train = X_train - X_train_mean
    y_train = y_train - y_train_mean
    
    X_test = X_test - X_train_mean # have to use train mean
    y_test = y_test - y_train_mean
    
    # L2 normalization
    X_train = X_train / np.sqrt(np.sum(X_train**2, axis = 1))[:,None]
    X_test = X_test / np.sqrt(np.sum(X_test**2, axis = 1))[:,None]
    
    for j in range(len(lambda_values)):
        ridge = lm.Ridge(alpha = lambda_values[j])
        ridge = ridge.fit(X_train, y_train)
        y_test_est = ridge.predict(X_test)
        mse = np.mean((y_test - y_test_est) ** 2)
        MSE[j,i] = mse
        betas[j,:] = ridge.coef_
        #print(betas)
        #print(lambda_values[i])
    
    i = i+1 # to increase index of MSE matrix

meanMSE = np.mean(MSE, axis = 1)
jOpt = np.argsort(meanMSE)[0]
lambda_OP = lambda_values[jOpt]
RMSE= np.sqrt (meanMSE[jOpt])

# Remember excact solution depends on a random indexing, so results may vary
# I reuse the plot with all the betas from 1 a) and add a line for the optimal value of lambda
plt.figure(figsize = (10,10))
plt.semilogx(lambda_values,betas)
plt.xlabel('lambdas')
plt.ylabel('betas')
plt.suptitle(f"Optimal lambda: {lambda_OP}", fontsize=20)
plt.semilogx([lambda_OP, lambda_OP], [np.min(betas), np.max(betas)], marker = ".")
plt.show()

