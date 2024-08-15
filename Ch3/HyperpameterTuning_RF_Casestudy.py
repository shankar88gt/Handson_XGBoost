import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
import warnings
warnings.filterwarnings('ignore')

#Fetch data
df_census = pd.read_csv('/Users/shankarmanoharan/Stats/Handson_XGBoost/Ch3/bike_rentals_cleaned.csv')
X= df_census.iloc[:,:-1]
y= df_census.iloc[:,-1]
X_train,X_test,y_train,y_test = train_test_split(X,y)

rf = RandomForestRegressor(n_estimators=50,n_jobs=-1,warm_start=True,random_state=2)
scores = cross_val_score(rf,X,y,cv=10,scoring='neg_mean_squared_error')
rmse = np.sqrt(-scores)
print('RMSE', np.round(rmse,3))
print('RMSE mean',rmse.mean())

from sklearn.model_selection import RandomizedSearchCV
def randomized_searchReg(params,runs=16,reg=RandomForestRegressor()):
    rand_reg = RandomizedSearchCV(reg,params,n_iter=runs,scoring='neg_mean_squared_error',cv=10,n_jobs=-1,random_state=2)
    rand_reg.fit(X_train,y_train)
    best_model = rand_reg.best_estimator_
    best_params = rand_reg.best_params_
    print('Best params:',best_params)
    best_score = np.sqrt(-rand_reg.best_score_)
    print('Training Score:',best_score)
    y_pred = best_model.predict(X_test)
    rsme_test = MSE(y_test,y_pred)**0.5
    print('Test Score:', rsme_test)

params = {
            'min_weight_fraction_leaf' : [0.0, 0.0025, 0.005, 0.0075, 0.01, 0.01, 0.05],
            'min_samples_split' : [2,0.01,0.02,0.03,0.04,0.05,0.06,0.08,0.1],
            'min_samples_leaf' : [1,2,4,6,8,10,20,30],
            'min_impurity_decrease' : [0.0,0.01,0.05,0.10,0.15,0.2],
            'max_leaf_nodes' : [10,15,20,25,30,35,40,45,50,None],
            'max_features' : ['auto',0.8,0.7,0.6,0.5,0.4],
            'max_depth' : [None,2,4,6,8,10,20]
}

randomized_searchReg(params=params)

#Narrowing the search space
params1 = {
            
            'min_samples_leaf' : [1,2,4,6,8,10,20,30],
            'min_impurity_decrease' : [0.0,0.01,0.05,0.10,0.15,0.2],            
            'max_features' : ['auto',0.8,0.7,0.6,0.5,0.4],
            'max_depth' : [None,2,4,6,8,10,20]
}

randomized_searchReg(params=params1)

#Further narrowing & increase iterations
params2 = {
            
            'min_samples_leaf' : [1,2,4,6,8,10,20,30],
            'min_impurity_decrease' : [0.0,0.01,0.05,0.10,0.15,0.2],            
            'max_features' : ['auto',0.8,0.7,0.6,0.5,0.4],
            'max_depth' : [None,2,4,6,8,10,12,15,20]
}
randomized_searchReg(params=params2,runs=20)

params3 = {
            
            'min_samples_leaf' : [1,2,3,4,5,6,8,10],
            'min_impurity_decrease' : [0.0,0.01,0.05,0.10,0.12,0.15],            
            'max_features' : ['auto',0.8,0.7,0.6,0.5,0.4],
            'max_depth' : [None,10,12,15,20,25,30]
}
randomized_searchReg(params=params3,runs=20)

#Place the best model int he cross val score 
rf = RandomForestRegressor(n_estimators=100,n_jobs=-1,min_samples_leaf= 1,min_impurity_decrease=0.1,max_features=0.6,max_depth=10,random_state=2)
scores = cross_val_score(rf,X,y,cv=10,scoring='neg_mean_squared_error')
rmse = np.sqrt(-scores)
print('RMSE', np.round(rmse,3))
print('RMSE mean',rmse.mean())

"""RMSE [ 822.389  511.564  527.819  800.491  756.24   734.108  885.499  788.679
  791.347 1597.216]"""
## The last score is bad; shuffling the data if it helps

from sklearn.utils import shuffle
dfs_census = shuffle(df_census,random_state=2)
X= dfs_census.iloc[:,:-1]
y= dfs_census.iloc[:,-1]
rf = RandomForestRegressor(n_estimators=100,n_jobs=-1,min_samples_leaf= 1,min_impurity_decrease=0.1,max_features=0.6,max_depth=10,random_state=2)
scores = cross_val_score(rf,X,y,cv=10,scoring='neg_mean_squared_error')
rmse = np.sqrt(-scores)
print('Post Shuffle RMSE', np.round(rmse,3))
print('Post SHuffle RMSE mean',rmse.mean())