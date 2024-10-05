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
# dart gives the exact same result as gbtree. its unclear whether trees have been dropped or the dropping has no effect

# Compare how DART compares to gblinear
classification_model(XGBClassifier(booster='gblinear'))
classification_model(LogisticRegression(max_iter=1000))
