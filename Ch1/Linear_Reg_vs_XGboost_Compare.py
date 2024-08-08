import pandas
from ucimlrepo import fetch_ucirepo 
# fetch dataset 
bike_sharing = fetch_ucirepo(id=275) 
# data (as pandas dataframes) 
X = bike_sharing.data.features 
y = bike_sharing.data.targets 

#drop non numerical columns for now
X = X.drop('dteday', axis=1)

# Simple linear Regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import warnings
warnings.filterwarnings('ignore')

#Split the data
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=2)

#############################################################
# Simple linear Reg
#############################################################

# fit & predict
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)
y_pred = lin_reg.predict(X_test)

mse = mean_squared_error(y_test,y_pred)
rmse = np.sqrt(mse)
print('linear Reg error', rmse)

#############################################################
# XGboost
#############################################################

from xgboost import XGBRegressor
xg_reg = XGBRegressor()
xg_reg.fit(X_train,y_train)
y_pred = xg_reg.predict(X_test)

mse = mean_squared_error(y_test,y_pred)
rmse = np.sqrt(mse)
print('XGBReg error', rmse)





