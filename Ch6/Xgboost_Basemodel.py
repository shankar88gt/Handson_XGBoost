import pandas as pd
df = pd.read_csv('/Users/shankarmanoharan/VSCode/Handson_XGBoost/Ch6/heart_disease.csv')
print(df.info())

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

model = XGBClassifier(booster='gbtree',objective='binary:logistic',random_state=2)

from sklearn.model_selection import cross_val_score
import numpy as np
scores = cross_val_score(model,X,y,cv=5)
print('Accuracy',np.round(scores,2))
print('Accuracy mean', scores.mean())

#using stratified Kfold
from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=5,random_state=2,shuffle=True)
scores = cross_val_score(model,X,y,cv=kfold)
print('Accuracy',np.round(scores,2))
print('Accuracy mean', scores.mean())

