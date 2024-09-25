# combining GridSearchCV & RandomizedSearchCV 

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
def grid_search(kfold,X,y,params,random=False):
    xgb = XGBClassifier(booster='gbtree',objective='binary:logistic',random_state=2)
    if random:
        grid = RandomizedSearchCV(n_iter=20,)
    else:
        grid = GridSearchCV(xgb,params,n_jobs=-1,cv=kfold)
    grid.fit(X,y)
    best_params = grid.best_params_
    print('Best Params:',best_params)
    best_score = grid.best_score_
    print('Training Best Score',best_score)



