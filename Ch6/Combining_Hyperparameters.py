# combining all hyperparametrs techniques tried so far to find the best model
# approach one at a time and then keep appending

import pandas as pd
df = pd.read_csv('/Users/shankarmanoharan/VSCode/Handson_XGBoost/Ch6/heart_disease.csv')
#print(df.info())

from xgboost import XGBClassifier

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

model = XGBClassifier(booster='gbtree',objective='binary:logistic',random_state=2)

from sklearn.model_selection import cross_val_score
import numpy as np
scores = cross_val_score(model,X,y,cv=5)
print('Base Accuracy',np.round(scores,2))
print('Base Accuracy mean', scores.mean())

#using stratified Kfold
#Iteration 1
from HyperParametrs_Grid_RandmCV_Combined import grid_search
from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=5,random_state=2,shuffle=True)
grid_search(kfold,X,y,params={'n_estimators':[2,25,50,75,100]})

#iteration 2
grid_search(kfold,X,y,params={'n_estimators':[2,25,50,75,100], 'max_depth':[1,2,3,4,5,6,7,8]})

#iteration 3
grid_search(kfold,X,y,params={'n_estimators':[2,25,50,75,100], 
                              'max_depth':[1,2,3,4,5,6,7,8],
                              'learning_rate':[0.01,0.05,0.1,0.2,0.3,0.4,0.5]})

#iteration 4
grid_search(kfold,X,y,params={'n_estimators':[2,25,50,75,100], 
                              'max_depth':[1,2,3,4,5,6,7,8],
                              'learning_rate':[0.01,0.05,0.1,0.2,0.3,0.4,0.5],
                              'min_child_weight':[1,2,3,4,5]})

#iteration 5
grid_search(kfold,X,y,params={'n_estimators':[2,25,50,75,100], 
                              'max_depth':[1,2,3,4,5,6,7,8],
                              'learning_rate':[0.01,0.05,0.1,0.2,0.3,0.4,0.5],
                              'min_child_weight':[1,2,3,4,5],
                              'subsample':[0.5,0.6,0.7,0.8,0.9]
                              })

#iteration 6 - Takes a ton of time. 
grid_search(kfold,X,y,params={'n_estimators':[2,25,50,75,100], 
                              'max_depth':[1,2,3,4,5,6,7,8],
                              'learning_rate':[0.01,0.05,0.1,0.2,0.3,0.4,0.5],
                              'min_child_weight':[1,2,3,4,5],
                              'subsample':[0.5,0.6,0.7,0.8,0.9],
                              'colsample_bytree':[0.5,0.6,0.7,0.8,0.9,1],
                              'colsample_bylevel':[0.5,0.6,0.7,0.8,0.9,1],
                              'colsample_bynode':[0.5,0.6,0.7,0.8,0.9,1],
                              'gamma':[0,0.01,0.05,0.1,0.5,1,2,3]
                              })

#iteration 7 : randomizedSearchCV for the parameters
grid_search(kfold,X,y,params={'n_estimators':[2,25,50,75,100], 
                              'max_depth':[1,2,3,4,5,6,7,8],
                              'learning_rate':[0.01,0.05,0.1,0.2,0.3,0.4,0.5],
                              'min_child_weight':[1,2,3,4,5],
                              'subsample':[0.5,0.6,0.7,0.8,0.9],
                              'colsample_bytree':[0.5,0.6,0.7,0.8,0.9,1],
                              'colsample_bylevel':[0.5,0.6,0.7,0.8,0.9,1],
                              'colsample_bynode':[0.5,0.6,0.7,0.8,0.9,1],
                              'gamma':[0,0.01,0.05,0.1,0.5,1,2,3]
                              }, random=True)


