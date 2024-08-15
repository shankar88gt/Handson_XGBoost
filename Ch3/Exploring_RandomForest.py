import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

#Fetch data
df_census = pd.read_csv('/Users/shankarmanoharan/Stats/Handson_XGBoost/Ch3/census_cleaned.csv')
X= df_census.iloc[:,:-1]
y= df_census.iloc[:,-1]

rf = RandomForestClassifier(n_estimators=10,random_state=2,n_jobs=-1)
scores = cross_val_score(rf,X,y,cv=5)
print('accuracy', np.round(scores,3))
print('accuracy mean',scores.mean())


### Random Forest regression
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

#Fetch data
df_census = pd.read_csv('/Users/shankarmanoharan/Stats/Handson_XGBoost/Ch3/bike_rentals_cleaned.csv')
X= df_census.iloc[:,:-1]
y= df_census.iloc[:,-1]

rf = RandomForestRegressor(n_estimators=10,random_state=2,n_jobs=-1)
scores = cross_val_score(rf,X,y,cv=10,scoring='neg_mean_squared_error')
rmse = np.sqrt(-scores)
print('RMSE', np.round(rmse,3))
print('RMSE mean',rmse.mean())