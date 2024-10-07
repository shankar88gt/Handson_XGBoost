# Ensemble Models

"""
 The winning models of Kaggle Competitions are rarely Individual Models they are almost always Ensembles
 by Ensembles we do not mean by boosting or bagging but pure ensembles that include any distinct models RF, XGboost & others

"""

# Sample Ensemble
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score


X,y = load_breast_cancer(return_X_y=True)
kfold = StratifiedKFold(n_splits=5)

def classification_model(model):
    scores = cross_val_score(model,X,y,cv=kfold)
    return scores.mean()

# Try Different Model which are to be part of Ensemble

print('XGB:',classification_model(XGBClassifier()))
print('XGB_gblin:',classification_model(XGBClassifier(booster='gblinear')))
print('XGB_dart:',classification_model(XGBClassifier(booster='dart',one_drop=True)))
print('RF:',classification_model(RandomForestClassifier(random_state=2)))
print('LogR:',classification_model(LogisticRegression(max_iter=10000)))
print('XGB_Hyp:',classification_model(XGBClassifier(max_depth=2,n_estimators=500,learning_rate=0.1)))

# Corelation
# the purpose of this section is not to select all the model for the ensemble but rather to select the non-corelated models

#Correlation is a statistical measure between -1 & 1 that indicates the strengths of linear relationship between two sets of points
# Correlation basics 
# https://en.wikipedia.org/wiki/Correlation

# Correlation in Ml Models
"""
A high correlation in ML models is undesirable in an ensemble 
Why ?
if all classfifiers make the same predictions then no new information is gained, making the second classifier superfluous
using a majority rules implementation. A prediction is only wrong if the majority of classfiers get it wrong. 
it is desirable to ahve a diversity of models that score well but give different predictions. if most models give same predictions then correlation is high
Finding differences in prediction where a strong model may be wrong gives the ensemble a chance to produce better results; 
"""

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=2)

def y_pred(model):
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test,y_pred)
    print(score)
    return y_pred

y_pred_gbtree = y_pred(XGBClassifier())
y_pred_dart = y_pred(XGBClassifier(booster='dart',one_drop=True))
y_pred_RF = y_pred(RandomForestClassifier())
y_pred_LOGR = y_pred(LogisticRegression(max_iter=10000))
y_pred_XGBHyp = y_pred(XGBClassifier(max_depth=2,n_estimators=500,learning_rate=0.1))

df_pred = pd.DataFrame(data=np.c_[y_pred_gbtree,y_pred_dart,y_pred_RF,y_pred_LOGR,y_pred_XGBHyp],columns = ['gbtree','dart','RF','LOGR','XGBHyp'])

print(df_pred.corr())

#Voting Classifier
# pick best and 2 other from the above
# XGBHyp , LOgR & gbtree are the other 2 with least corelation

# Voting classifier is designed to combine multiple classification models & select the output for each prediction using majority rules
# scikit Learn has votingClassfier & votingRegressor for each prblem


XGB_tuned = XGBClassifier(max_depth=2,n_estimators=500,learning_rate=0.1)
LogR = LogisticRegression(max_iter=10000)
gbtree = XGBClassifier()

estimators = [('XGB_tuned',XGB_tuned),('LogR',LogR),('gbtree',gbtree)]
ensemble = VotingClassifier(estimators)
scores = cross_val_score(ensemble,X,y,cv=kfold)
print('Post Ensemble',scores.mean())

# Results -  1% higher than individual Models







