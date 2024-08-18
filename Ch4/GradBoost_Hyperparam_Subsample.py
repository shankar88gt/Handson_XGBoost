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

samples = [1, 0.9, 0.8,0.7,0.6,0.5]
for sample in samples:
    gbr = GradientBoostingRegressor(max_depth=3,random_state=2,n_estimators=300,subsample=sample)
    gbr.fit(X_train,y_train)
    y_pred = gbr.predict(X_test)
    print('sample:', sample , 'Error:', MSE(y_test,y_pred)**0.5)

# When subsample is not equal to 1.0, the model is classfied as Stochastic gradient decent. 
# where stochastic indicates some randomness is inhererent in the model

