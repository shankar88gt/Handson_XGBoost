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

depths = [None, 1,2,3,4]
for depth in depths:
    gbr = GradientBoostingRegressor(max_depth=depth,random_state=2,n_estimators=300)
    gbr.fit(X_train,y_train)
    y_pred = gbr.predict(X_test)
    print('depth:', depth , 'Error:', MSE(y_test,y_pred)**0.5)
