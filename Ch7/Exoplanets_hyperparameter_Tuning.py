# Hypertuning with exoplanets

import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,classification_report
from sklearn.model_selection import train_test_split
import numpy as np

# BASE Model with label encoder
df = pd.read_csv('/Users/shankarmanoharan/VSCode/Handson_XGBoost/Ch7/exoplanets.csv',nrows=400)
X = df.iloc[:,1:]
y = df.iloc[:,0]
# to transform labels from 1,2 to 0,1
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y1 = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X,y1,random_state=2)

model = XGBClassifier(booster='gbtree', objective='binary:logistic',random_state=2)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print('accuracy',accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))

# BASE Model with label encoder
df = pd.read_csv('/Users/shankarmanoharan/VSCode/Handson_XGBoost/Ch7/exoplanets.csv',nrows=400)

# to transform labels from 1,2 to 0,1
df['LABEL'] = df['LABEL'].replace(1,0)
df['LABEL'] = df['LABEL'].replace(2,1)

X = df.iloc[:,1:]
y = df.iloc[:,0]

X_train, X_test, y_train, y_test = train_test_split(X,y1,random_state=2)

model = XGBClassifier(booster='gbtree', objective='binary:logistic',random_state=2)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print('accuracy',accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))

# using scale pos weight - results shd be similar to oversampling
model = XGBClassifier(scale_pos_weight=10,random_state=2)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print('accuracy',accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))
"""
              precision    recall  f1-score   support
           0       0.91      0.95      0.93        88
           1       0.50      0.33      0.40        12
    accuracy                           0.88       100
   macro avg       0.71      0.64      0.67       100
weighted avg       0.86      0.88      0.87       100
"""

# Tuning using Gridsearch & Randomized search - Baseline Model
from sklearn.model_selection import StratifiedKFold, cross_val_score
kfold = StratifiedKFold(n_splits=2,shuffle=True,random_state=2)
model = XGBClassifier(scale_pos_weight=10,random_state=2)

scores = cross_val_score(model,X,y,cv=kfold,scoring='recall')
print('Recall', scores)
print('Recall Mean', scores.mean())

# custom function
from Combining_hyperparameters_v1 import grid_search

#iter 1
grid_search(params={'n_estimators':[50,200,400,800]},kfold=kfold,X=X,y=y)

#iter 2
grid_search(params={'learning_rate':[0.01,0.05,0.2,0.3]}, 
                     kfold=kfold,X=X,y=y)

#iter 3
grid_search(params={'max_depth':[1,2,4,8]}, 
                     kfold=kfold,X=X,y=y)

#iter 4
grid_search(params={'subsample':[0.3,0.4,0.5,0.7,0.9]}, 
                     kfold=kfold,X=X,y=y)

#iter 5
grid_search(params={'gamma':[0.05,0.1,0.5,1]}, 
                     kfold=kfold,X=X,y=y)

#iter 6
grid_search(params={'max_delta_step':[1,3,5,7]}, 
                     kfold=kfold,X=X,y=y)

#iter 7
grid_search(params={'subsample':[0.3,0.4,0.5,0.7],
                    'colsample_bylevel':[0.3,0.5,0.7,0.9,1],
                    'colsample_bynode':[0.3,0.5,0.7,0.9,1],
                    'colsample_bytree':[0.3,0.5,0.7,0.9,1]
                    }, 
                     kfold=kfold,X=X,y=y)

#iter 6
grid_search(params={'max_depth':[1,2],
                    'gamma':[0.025,0.05,0.03,0.1],
                    'subsample':[0.3,0.4,0.5],
                    'learning_rate':[0.001,0.01,0.02,0.03,0.3]}, 
                     kfold=kfold,X=X,y=y)


#iter 7
grid_search(params={'subsample':[0.3,0.5,0.7,0.9,1],
                    'colsample_bylevel':[0.3,0.5,0.7,0.9,1],
                    'colsample_bynode':[0.3,0.5,0.7,0.9,1],
                    'colsample_bytree':[0.3,0.5,0.7,0.9,1]}, 
                     kfold=kfold,X=X,y=y)


#iter 8
grid_search(params={'max_depth':[1,2],
                    'gamma':[0.025,0.03,0.05,0.08],
                    'subsample':[0.3,0.4],
                    'learning_rate':[0.001,0.01,0.02,0.03,0.3],
                    'subsample':[0.3],
                    'colsample_bylevel':[0.5],
                    'colsample_bynode':[1],
                    'colsample_bytree':[0.3]}, 
                     kfold=kfold,X=X,y=y)


# balanced subset 
X_short = X.iloc[:74,:]
y_short = y.iloc[:74]

#grid search with balanced data
grid_search(params={'max_depth':[1,2],
                    'gamma':[0.025,0.03,0.05,0.08],
                    'subsample':[0.3,0.4],
                    'learning_rate':[0.001,0.01,0.02,0.03,0.3],
                    'subsample':[0.3],
                    'colsample_bylevel':[0.4,0.5],
                    'colsample_bynode':[1],
                    'colsample_bytree':[0.3]}, 
                     kfold=kfold,X=X_short,y=y_short)






