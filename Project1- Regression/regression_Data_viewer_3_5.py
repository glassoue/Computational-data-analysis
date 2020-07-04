import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
#from fancyimpute import KNN
from funcer import encode_ordinal
import random
from sklearn.ensemble import AdaBoostRegressor

np.random.seed(2)

dat_read = pd.read_csv('/home/ghassen97/Desktop/S8/computional analysis/case/Case Data/case1Data.txt')
dat = dat_read
dat_ordinal = dat.copy()
dat_random = dat.copy()
dat = dat_read.replace(' NaN', np.nan)
H = list(dat.columns.values)


cat_cols = H[96:101]
for columns in cat_cols:
    encode_ordinal(dat_ordinal[columns])
    dat_random[columns].fillna(lambda x: random.choice(dat_random[dat_random[columns] != np.nan][columns]), inplace=True)

#imputer = KNN(k=5)
#dat_KNN = pd.DataFrame((imputer.fit_transform(dat_ordinal)), columns=H)

#dat = dat_random

def model_validation(dat, method, param, t_s):
    H = list(dat.columns.values)
    X = dat[H[1:101]]
    y = dat[H[0]]

    numeric_features = H[1:96]
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())])

    categorial_features = H[96:101]
    categorial_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer()),
        ('onehot', OneHotEncoder()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorial_transformer, categorial_features)
             ])

    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('method', method)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=t_s)

    grid_search = GridSearchCV(clf, param, cv=5, scoring='neg_root_mean_squared_error', verbose=True)
    grid_search.fit(X_train, y_train)
    model = grid_search.best_estimator_
    model_fit = model.fit(X_train, y_train)

    y_h = model_fit.predict(X_test)

    EPE = np.sum(np.power((y_test - y_h), 2))/(t_s*100)

    print(EPE)
    print(grid_search.best_score_)

    plt.plot(y_h, 'r')
    plt.plot(y_test.to_numpy(), 'b')
    plt.show()

    return grid_search

test_size = 0.2
alpha = np.logspace(-4, 0.5, num=100)


param_las = {
 'preprocessor__cat__imputer__strategy': ['most_frequent'],
 'method__alpha': alpha,
 'method__max_iter': [1200000],
 'method__fit_intercept': [True]
 }

print('Lasso')
model_validation(dat, Lasso(), param_las, t_s=test_size)
#model_validation(dat_rmove, Lasso(), param_las, t_s=test_size)


param_els = {
 'preprocessor__cat__imputer__strategy': ['most_frequent'],
 'method__alpha': alpha,
 'method__l1_ratio': [1/5, 3/5],
 'method__max_iter': [1200000],
 'method__fit_intercept': [True]
 }

print('ElasticNet')
model_validation(dat, ElasticNet(), param_els, t_s=test_size)
#model_validation(dat_rmove, ElasticNet(), param_els, t_s=test_size)

'''
param_ADA = {
# 'preprocessor__cat__imputer__strategy': ['most_frequent'],
 'method__n_estimators': [50, 100],
 'method__learning_rate': [0.01, 0.05, 0.1, 0.5],
 'method__loss': ['linear', 'square', 'exponential']
 }

print('AdaBoostRegressor')
model_validation(dat, AdaBoostRegressor(), param_ADA, t_s=test_size)
#model_validation(dat_rmove, AdaBoostRegressor(), param_ADA, t_s=test_size)


param_GDB = {
# 'preprocessor__cat__imputer__strategy': ['most_frequent'],
 'method__n_estimators': [50, 100],
 'method__learning_rate': [0.01, 0.05, 0.1, 0.5],
 'method__loss': ['ls', 'lad']
 }


print('GradientBoostingRegressor')
model_validation(dat, GradientBoostingRegressor(), param_GDB, t_s=test_size)
#model_validation(dat_rmove, GradientBoostingRegressor(), param_GDB, t_s=test_size)

'''
depth = np.linspace(1, 100, dtype=int)
param_RF = {
 'preprocessor__cat__imputer__strategy': ['most_frequent'],
 'method__max_depth': [51],
 }


print('RandomForest')
mod_RF = model_validation(dat, RandomForestRegressor(), param_RF, t_s=test_size)


C = np.logspace(-4, 0.5, num=50)
epsilon = np.logspace(-4, 0, num=50)

param_RF = {
 'preprocessor__cat__imputer__strategy': ['most_frequent'],
 'method__C': [0.0460537825582],
 'method__epsilon': [0.26826957952],
 'method__kernel': ['linear']
 }


print('SVR')
mod_SVR = model_validation(dat, SVR(), param_RF, t_s=test_size)

#%%
''' Standardize '''
scaler = StandardScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

''' Regression '''
# Use numpy array from now on
data = df.values[:100]
new_data = df.values[100:]
X = data[:, 1:]
y = data[:, 0]
X_new = new_data[:, 1:]

model = AdaBoostRegressor(n_estimators=50, learning_rate=0.1)

fitter = model.fit(X_train, y_train) # Use all the data from case1Data

predict = fitter.predict(X_new) # Use data from Case1Data.txt







