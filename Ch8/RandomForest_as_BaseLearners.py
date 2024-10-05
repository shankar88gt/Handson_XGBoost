#  Two strategies to implement random forest in Xgboost
#  as Baselearners
#  as XGBoost RF as indiviul models


# AS BASE Learners - STrategy 1
# there is not an option to set the booster as RF but instead use the default boosters such as gbtree, dart etc and use the hyperparamer num_prallel_tree > 1
# the round will no longer contain one tree but instead a forest in each round


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

# Xgboost Random Forest
#Xgboost gbtree
score = regression_model(XGBRegressor(booster='gbtree',num_parallel_tree=25))
print('XGB gbtree with RF25',score)

# Xgboost Random Forest
#Xgboost gbtree
score = regression_model(XGBRegressor(booster='gbtree',num_parallel_tree=5))
print('XGB gbtree with RF5',score)

# Analysing the results
# The score is ok. its nearly the same as boosting a single gbtree. the reson is that gradient boosting is designed to learn from the mistakes of previous trees. 
# by starting with a robust RF. ther is little to be learned; the gains are minimal at best
# when building RF low values of num_parallel_tree are better


# Strategy 2
#RF as XGBoost Models
# XGBRFRegressor & XGBRFClassifier
# in 2020 these were the defaults since this is still in the experimentation stage

# n_estimators
# use n_estimators and not num_parallel_trees - you are not Gradient boosting but bagging trees in one round as is the case with traditional Random forest

#learning_rate
# this is generally designed for models that learn not RF as they consist of one round of trees. default is set to 1 so dont modify

#Subsample, colsample_by_node
# scikit learns defauls are 1
# xgboost defaults are at 0.8 - less prone to overfitting

# Xgboost Random Forest as standalone
#Xgboost gbtree
score = regression_model(XGBRFRegressor(objective='reg:squarederror'))
print('XGB gbtree with RF IndvModel',score)

# Scikit learn RF
score = regression_model(RandomForestRegressor())
print('XGB gbtree with Scikit RF',score)

# conclusion
# Can be tried
# boosting is designed to learn from weak models and not strong models
#  random forest as base learners shd be used sparingly
# XGBoost RF can be used as an alternatives to scikit learns
# XGBoost & SCikit are close in terms of performance although XGBRF only slightly better
