# Tests with tuned parameters on Data size to check the effects

import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,classification_report
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold 

import numpy as np

# BASE Model with label encoder
df = pd.read_csv('/Users/shankarmanoharan/VSCode/Handson_XGBoost/Ch7/exoplanets.csv')
# to transform labels from 1,2 to 0,1
df['LABEL'] = df['LABEL'].replace(1,0)
df['LABEL'] = df['LABEL'].replace(2,1)

X = df.iloc[:,1:]
y = df.iloc[:,0]

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=2)

model = XGBClassifier(booster='gbtree', objective='binary:logistic',random_state=2)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print('Base Model accuracy',accuracy_score(y_test,y_pred))
print('Base Model Classification report\n', classification_report(y_test,y_pred))
# WITH IMBALANCED DATA, ACCURACY ISNT ENOUGH

"""
Best Params: {'colsample_bylevel': 0.5, 'colsample_bynode': 1, 'colsample_bytree': 0.3, 'subsample': 0.3}
Training Best Score 0.35233918128654973

Best Params: {'gamma': 0.05, 'learning_rate': 0.3, 'max_depth': 2, 'subsample': 0.4}
Training Best Score 0.2733918128654971

Best Params: {'colsample_bylevel': 0.5, 'colsample_bynode': 1, 'colsample_bytree': 0.3, 'subsample': 0.3}
Training Best Score 0.35233918128654973

Best Params: {'colsample_bylevel': 0.5, 'colsample_bynode': 1, 'colsample_bytree': 0.3, 'gamma': 0.08, 'learning_rate': 0.3, 'max_depth': 1, 'subsample': 0.3}
Training Best Score 0.24269005847953215

Best Params: {'colsample_bylevel': 0.4, 'colsample_bynode': 1, 'colsample_bytree': 0.3, 'gamma': 0.025, 'learning_rate': 0.01, 'max_depth': 2, 'subsample': 0.3}
Training Best Score 0.5994152046783625
"""


# Full Dataset accuracy with Scale pos weight
weight = int(5050/37)
kfold = StratifiedKFold(n_splits=2,shuffle=True,random_state=2)
model=XGBClassifier(scale_pos_weight=weight,random_state=2)
scores = cross_val_score(model,X,y,cv=kfold,scoring='recall')
print('Full Model Recall :',scores)
print('Full Model Recall Mean',scores.mean())

# Full Dataset accuracy with Scale pos weight & Params
model=XGBClassifier(scale_pos_weight=weight,learning_rate=0.001,random_state=2)
scores = cross_val_score(model,X,y,cv=kfold,scoring='recall')
print('Full Model with HyperP Recall :',scores)
print('Full Model with HyperP Recall Mean',scores.mean())

# Full Dataset accuracy with Scale pos weight & Params
model=XGBClassifier(scale_pos_weight=weight,max_depth=2,learning_rate=0.001,random_state=2)
scores = cross_val_score(model,X,y,cv=kfold,scoring='recall')
print('Full Model with HyperP1 Recall :',scores)
print('Full Model with HyperP1 Recall Mean',scores.mean())
                    
# 400 Rows accuracy with Scale pos weight & Params
X = df.iloc[:400,1:]
y = df.iloc[:400,0]
model=XGBClassifier(scale_pos_weight=weight,max_depth=2,learning_rate=0.001,random_state=2)
scores = cross_val_score(model,X,y,cv=kfold,scoring='recall')
print('400 rows with HyperP1 Recall :',scores)
print('400 rows with HyperP1 Recall Mean',scores.mean())

# 74 Rows accuracy with Scale pos weight & Params
X = df.iloc[:74,1:]
y = df.iloc[:74,0]
model=XGBClassifier(scale_pos_weight=weight,max_depth=2,learning_rate=0.001,random_state=2)
scores = cross_val_score(model,X,y,cv=kfold,scoring='recall')
print('74 rows with HyperP1 Recall :',scores)
print('74 rows with HyperP1 Recall Mean',scores.mean())


"""
Conclusion
    1) using precision without recall will lead to suboptimal models
    2) Over emphasis on high scores from small datasets is not adviced
    3) Test scores low but training scores high; bias variance issues - overfitting
    4) with high imbalanced data high accuracy is meaningless

    Exoplanets - Model performas with 70% accuracy but 37 exoplanets are not enough to build a robust model

"""
