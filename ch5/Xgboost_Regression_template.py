import pandas as pd
import numpy as np
from sklearn import datasets
X,y = datasets.load_diabetes(return_X_y=True)
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
xgb = XGBRegressor(booster='gbtree',objective='reg:squarederror',max_depth=6,learning_rate=0.1,n_estimators=100,random_state=2,n_jobs=-1)
score = cross_val_score(xgb,X,y,scoring='neg_mean_squared_error',cv=5)
rmse = np.sqrt(-score)
print('RMSE:', rmse)
print("RMSE mean:", rmse.mean())

print(pd.DataFrame(y).describe())


