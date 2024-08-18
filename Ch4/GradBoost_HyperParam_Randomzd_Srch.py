import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

#Fetch data
df_census = pd.read_csv('/Users/shankarmanoharan/Stats/Handson_XGBoost/Ch3/bike_rentals_cleaned.csv')
X= df_census.iloc[:,:-1]
y= df_census.iloc[:,-1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=2)

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error as MSE

Params = { 'subsample' : [0.65, 0.7, 0.75],
           'n_estimators' : [300,500, 500],
           'learning_rate' : [0.05,0.075,0.1],
           'max_depth' : [2,3,4]
}

from sklearn.model_selection import RandomizedSearchCV
gbr = GradientBoostingRegressor(random_state=2)
rand = RandomizedSearchCV(gbr,Params,n_iter=15,scoring='neg_mean_squared_error',cv=5,n_jobs=-1,random_state=2)

rand.fit(X_train,y_train)
best_model = rand.best_estimator_
best_params= rand.best_params_
print('best param :',best_params)
best_score = np.sqrt(-rand.best_score_)
print('best Training Score :', best_score)
y_pred = best_model.predict(X_test)
rmse_test = MSE(y_test,y_pred)**0.5
print('Test Score:',rmse_test)

"""try around this
best param : {'subsample': 0.75, 'n_estimators': 1600, 'max_depth': 3, 'learning_rate': 0.02}
RMSE - around 596"""

### Sammple XgBoost
from xgboost import XGBRegressor
xg_reg = XGBRegressor(max_depth=3,n_estimators=1600,eta=0.02,subsample =0.75,random_state=2)
xg_reg.fit(X_train,y_train)
y_pred=xg_reg.predict(X_test)
print("XGB Sample Error", MSE(y_test,y_pred)**0.5)
