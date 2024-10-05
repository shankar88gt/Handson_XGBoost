# Xgboost gblinear experiments

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor, XGBClassifier, XGBRFClassifier, XGBRFRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.linear_model import Lasso,Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error as MSE
from sklearn.datasets import load_diabetes

# Load the data
X,y = load_diabetes(return_X_y=True)
Kfold = KFold(n_splits=5,shuffle=True,random_state=2)

def regression_model(model):
    scores = cross_val_score(model,X,y,scoring='neg_mean_squared_error',cv=Kfold)
    rmse = (-scores)**0.5
    return rmse.mean()

# Xgboost gblinear
score = regression_model(XGBRegressor(booster='gblinear'))
print('XGB gblinear',score)

#Linear Regression
score = regression_model(LinearRegression())
print('Linear Reg',score)

#Lasso
score = regression_model(Lasso())
print('Lasso Reg',score)

#Ridge
score = regression_model(Ridge())
print('Ridge Reg',score)

#Xgboost gbtree
score = regression_model(XGBRegressor(booster='gbtree'))
print('XGB gbtree',score)

# comparing all the results. the error is minimum on the gblinear compared to other methods. a hint that the linear solution is sufficient


"""
gblinear Hyperparameters
    reg_lambda
        L2 Regularization used by Ridge
    reg_alpha
        L1 Regularization used by Lasso
    updater
        This is the algo that xgboost uses to build linear model during each round of boosting
        Shotgun uses hogwild parallelism with coordinate descent
            co-ordinate descent is a machine learning term defined as minimizing the error by finding the gradient one coordinate at a time
    feature_selector
        determines how the weights are selected with following options
            1) cyclic - cycles thru features iteratively
            2) shuffle - cyclic with random feature shuffling in each round
            3) random
            4) greedy - time consuming; selects with greatest gradient magnitude
            5) thrifty - approx greeedy, reorders features according to weight changes
                shotgun:cyclic,suffle
                coord_descent:random,greedy,thrifty
    top_k
        is the number os features that greedy & thrifty select from during coordinate descent
"""

# gblinear grid Search
def gridsearch(params,X,y,kfold,reg=XGBRegressor(booster='gblinear'),):
    grid_reg = GridSearchCV(reg,params,scoring='neg_mean_squared_error',cv=kfold)
    grid_reg.fit(X,y)
    best_params = grid_reg.best_params_
    print('Best params:', best_params)
    best_score = np.sqrt(-grid_reg.best_score_)
    print('Best Score:', best_score)

# iteration 1
gridsearch(params={'reg_alpha':[0.001,0.01,0.1,0.5,1,5]},X=X,y=y,kfold=Kfold)

# iteration 2
gridsearch(params={'reg_lambda':[0.001,0.01,0.1,0.5,1,5]},X=X,y=y,kfold=Kfold)

# iteration 3
gridsearch(params={'feature_selector':['shuffle']},X=X,y=y,kfold=Kfold)

# iteration 4
gridsearch(params={'feature_selector':['random','greedy','thrifty'], 'updater':['coord_descent']},X=X,y=y,kfold=Kfold)

# iteration 5
gridsearch(params={'feature_selector':['random','greedy','thrifty'], 'updater':['coord_descent'], 'top_k':[3,5,7,9]},X=X,y=y,kfold=Kfold)

# slight improvement in the error 
# gblinear is a compelling option but it should be used when u have a reason to believe that linear model may perform better than tree based model
# gblinear did outperform Linear regression by a slight margin
# strong option when datasets are strong and linear
# An option for classification datasets as well


