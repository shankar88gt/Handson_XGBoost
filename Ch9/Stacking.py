# Stacking Models

"""
Stacking combines ML models at two different levels
Base level : Whole models make predictions on all the data
Meta Level : Which takes the predictions of the base models as inputs & uses them to generate final predictions

Note that stacking is distinct from a standard ensemble on account of the meta model
it is generally advised to use a simple meta model 

"""

# Sample Ensemble
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier, StackingClassifier

X,y = load_breast_cancer(return_X_y=True)
kfold = StratifiedKFold(n_splits=5)

#Voting Classifier
XGB_tuned = XGBClassifier(max_depth=2,n_estimators=500,learning_rate=0.1)
LogR = LogisticRegression(max_iter=10000)
gbtree = XGBClassifier()

estimators = [('XGB_tuned',XGB_tuned),('LogR',LogR),('gbtree',gbtree)]
ensemble = VotingClassifier(estimators)
scores = cross_val_score(ensemble,X,y,cv=kfold)
print('Voting Ensemble',scores.mean())

#Stacking
estimators = [('XGB_tuned',XGB_tuned),('LogR',LogR),('gbtree',gbtree)]
meta_model = LogisticRegression()

clf = StackingClassifier(estimators=estimators,final_estimator=meta_model)
scores = cross_val_score(clf,X,y,cv=kfold)
print('Stacking',scores.mean())

#Analysing the Results - Stacking gives the best results so far.  0.5% better than VotingClassifier

