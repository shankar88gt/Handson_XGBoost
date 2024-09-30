# combining GridSearchCV & RandomizedSearchCV  - V1

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
def grid_search(kfold,X,y,params,random=False):
    xgb = XGBClassifier(booster='gbtree',random_state=2)
    if random:
        grid = RandomizedSearchCV(xgb,params,cv=kfold,scoring='recall',n_jobs=-1,n_iter=25)
    else:
        grid = GridSearchCV(xgb,params,cv=kfold,scoring='recall',n_jobs=-1)
    grid.fit(X,y)
    best_params = grid.best_params_
    print('Best Params:',best_params)
    best_score = grid.best_score_
    print('Training Best Score',best_score)