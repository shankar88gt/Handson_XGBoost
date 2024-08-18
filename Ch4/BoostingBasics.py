"""Random Forest are classified as bagging algorithms
    new trees pays no attention to previous trees and built from scratch
Boosting learns from mistakes of individual trees
    each new tree is built from the previous trees; not in isolation and built on top of one another
Adaboost
    each tree adjust its weights based ont he errors of the previous trees
    more attention to predictions that went wrong
    transform weak learners to strong learners
    iterative correction
    if the base model is too strong. learning process is limited
gradient boosting
    also adjust based on incorrect predictions
    builds a new tree entirely based on the errors of the previous trees
    a new tree compltly around the mistakes of the previous trees; doesnt care about the predictions correct
    error shd be comprehensive; residuals are used
    computes the residuals of each tree's prediction and summs all the residuals
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

#Fetch data
df_census = pd.read_csv('/Users/shankarmanoharan/Stats/Handson_XGBoost/Ch3/bike_rentals_cleaned.csv')
X= df_census.iloc[:,:-1]
y= df_census.iloc[:,-1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=2)

from sklearn.tree import DecisionTreeRegressor

#iteration 1
tree_1 = DecisionTreeRegressor(max_depth=2,random_state=2)
tree_1.fit(X_train,y_train)
y_train_pred = tree_1.predict(X_train)
y2_train = y_train - y_train_pred

#Iteration_2
tree_2 = DecisionTreeRegressor(max_depth=2,random_state=2)
tree_2.fit(X_train,y2_train)
y2_train_pred = tree_2.predict(X_train)
y3_train = y2_train - y2_train_pred

#Iteration_3
tree_3 = DecisionTreeRegressor(max_depth=2,random_state=2)
tree_3.fit(X_train,y3_train)

from sklearn.metrics import mean_squared_error as MSE

y1_pred = tree_1.predict(X_test)
print("Y1 pred:", y1_pred.sum())
print(MSE(y_test,y1_pred)**0.5)
y2_pred = tree_2.predict(X_test)
print("Y2 pred:", y2_pred.sum())
print(MSE(y_test,y2_pred)**0.5)
y3_pred = tree_3.predict(X_test)
print("Y3 pred:", y3_pred.sum())
print(MSE(y_test,y3_pred)**0.5)

y_pred = y1_pred + y2_pred + y3_pred
print(MSE(y_test,y_pred)**0.5)

#Building in sklearn
from sklearn.ensemble import GradientBoostingRegressor
gbr = GradientBoostingRegressor(max_depth=2,random_state=2,n_estimators=3,learning_rate=1.0)
gbr.fit(X_train,y_train)
y_pred = gbr.predict(X_test)
print(MSE(y_test,y_pred)**0.5)

#30 estimators
gbr = GradientBoostingRegressor(max_depth=2,random_state=2,n_estimators=30,learning_rate=1.0)
gbr.fit(X_train,y_train)
y_pred = gbr.predict(X_test)
print(MSE(y_test,y_pred)**0.5)

#300 Estimators
gbr = GradientBoostingRegressor(max_depth=2,random_state=2,n_estimators=300,learning_rate=1.0)
gbr.fit(X_train,y_train)
y_pred = gbr.predict(X_test)
print(MSE(y_test,y_pred)**0.5)

#300 Estimators with learning rate set to defaults
gbr = GradientBoostingRegressor(max_depth=2,random_state=2,n_estimators=300)
gbr.fit(X_train,y_train)
y_pred = gbr.predict(X_test)
print(MSE(y_test,y_pred)**0.5)






