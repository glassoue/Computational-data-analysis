import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso
#from missingpy import KNNImputer
#from fancyimpute import KNN
#from fancyimpute import KNN, NuclearNormMinimization, SoftImpute, BiScaler
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import ElasticNet

np.random.seed(1)

dat = pd.read_csv('/home/ghassen97/Desktop/S8/computional analysis/case/Case Data/case1Data.txt')
dat = dat.replace(' NaN', np.nan)
dat_new = pd.read_csv('/home/ghassen97/Desktop/S8/computional analysis/case/Case Data/case1Data_Xnew.txt')
dat_mew = dat_new.replace(' NaN', np.nan)
test_size = 0.3

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
        #('imputer', SimpleImputer(strategy='most_frequent')),
        ('imputer', SimpleImputer()),
         #('imputer', KNNImputer()),
        ('onehot', OneHotEncoder())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorial_transformer, categorial_features)])

    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('method', method)])



    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=t_s)
    grid_search = GridSearchCV(clf, param, cv=5)

    X_train[H[1:96]] = StandardScaler().fit_transform(X_train[H[1:96]])
    X_test[H[1:96]] = StandardScaler().fit_transform(X_test[H[1:96]])

    y_train = y_train - np.mean(y_train)
    y_test = y_test - np.mean(y_test)


    grid_search.fit(X_train, y_train)
    model = grid_search.best_estimator_.fit(X_train, y_train)

    Y_h = model.predict(X_test)
    print(Y_h)
    print(y_test.to_numpy())
    EPE = np.sum(np.power((y_test.to_numpy() - Y_h), 2))/(t_s*100)
    print('EPE',EPE)

    plt.plot(Y_h, 'r')
    plt.plot(y_test.to_numpy(), 'b')
    plt.show()



alpha = np.logspace(-4, 0.5, num=100)
param_las = {
 'preprocessor__cat__imputer__strategy': ['most_frequent'],
 'method__alpha': alpha,
 'method__max_iter': [1200000],
 'method__fit_intercept': [False]
 }

dat_rmove = dat.dropna()
model_validation(dat_rmove, Lasso(), param_las, t_s=test_size)

param_els = {
 'preprocessor__cat__imputer__strategy': ['most_frequent'],
 'method__alpha': alpha,
 'method__l1_ratio': [1/5, 3/5],
 'method__max_iter': [1200000],
 'method__fit_intercept': [False]
 }

#model_validation(dat, ElasticNet(), param_els, t_s=test_size)

param_ADA = {
 'preprocessor__cat__imputer__strategy': ['most_frequent'],
 'method__n_estimators': [50, 100],
 'method__learning_rate': [0.01, 0.05, 0.1, 0.5],
 'method__loss': ['linear', 'square', 'exponential']
 }

#model_validation(dat, AdaBoostRegressor(), param_ADA, t_s=test_size)

