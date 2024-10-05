# Xgboost DART experiments

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
    print('RMSE Mean:',rmse.mean())

#basic DART Model
regression_model(XGBRegressor(booster='dart',objective='reg:squarederror'))

# base model dart give the same result as gbtree
df = pd.read_csv('/Users/shankarmanoharan/VSCode/Handson_XGBoost/Ch8/census_cleaned.csv')
X = df.iloc[:,:-1]
y = df.iloc[:,-1]

def classification_model(model):
    scores = cross_val_score(model,X,y,scoring='accuracy',cv=Kfold)
    print('mean Accuracy Score:',scores.mean())

# gbtree
classification_model(XGBClassifier(booster='gbtree'))

#Dart
classification_model(XGBClassifier(booster='dart'))

#Results analysis
# dart gives the exact same result as gbtree. its unclear whether trees have been dropped or the dropping has no effect. gbtree is faster compared to DART

# Compare how DART compares to gblinear
classification_model(XGBClassifier(booster='gblinear'))
classification_model(LogisticRegression(max_iter=1000))

# gblinear has a clear edge over simple logistic regression


# DART Hyperparameters
#    https://xgboost.readthedocs.io/en/latest/parameter.html#additional-parameters-for-dart-booster-booster-dart

"""
The following are hyperparameters that are unique to DART
    Sample Type
        uniform : drops trees uniformly
        Weighted : drops trees proportion to their weights
    normalize_type
        Tree : new tree have the same weight as dropped Trees
        Forest : new trees have the same weight as sum of dropped Trees
    rate_drop
        allows user to set exactly how many trees are dropped percentage-wise
        Range [0.1,1.0]
    one_drop
        when set to 1, ensured at least one tree is always dropped during boosting round
        [0,1]
        used to ensure drops
    skip_drop
        gives the proabability of skipping the dropping entirly
"""

#Dart - one drop 
classification_model(XGBClassifier(booster='dart', one_drop=1))
# slight improvement

#lets switch to Regression for fast execution as we are trying diff params

# Sample Type
regression_model(XGBRegressor(booster='dart',objective='reg:squarederror',sample_type='weighted'))

# normalize Type
regression_model(XGBRegressor(booster='dart',objective='reg:squarederror',normalize_type='forest'))

# one drop
regression_model(XGBRegressor(booster='dart',objective='reg:squarederror',one_drop=1))


# grid search - iteration 1
grid_search(params={'rate_drop':[ 0.01,0.1,0.2,0.4] },reg=XGBRegressor(booster='dart',objective='reg:squarederror'.one_drop=1))

# grid search - iteration 2
grid_search(params={'skip_drop':[ 0.01,0.1,0.2,0.4] },reg=XGBRegressor(booster='dart',objective='reg:squarederror'.one_drop=1))

# Analysing Results
#   DART provides compelling option within XGboost framework
#   Since dart accepts all gbtree hyperparameters; its easy to change the learner from gbtree to dart
#   dart is definetly worth trying if using Xgboost

