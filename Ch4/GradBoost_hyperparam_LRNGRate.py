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
import matplotlib.pyplot as plt
import seaborn as sns

learning_rate = [0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]
RMSE300 = []
RMSE30 = []
RMSE3000= []


for value in learning_rate:
    gbr = GradientBoostingRegressor(max_depth=2,random_state=2,n_estimators=300,learning_rate=value)
    gbr.fit(X_train,y_train)
    y_pred = gbr.predict(X_test)
    RMSE300.append(MSE(y_test,y_pred)**0.5)
    print('learning rate:', value, 'Error:', MSE(y_test,y_pred)**0.5)


sns.set_theme()
plt.figure(figsize=(15,7))
plt.plot(learning_rate,RMSE300)
plt.xlabel('Learning Rate with 300 Trees')
plt.ylabel('RMSE')
plt.title("RMSE vs Learning Rate")
plt.show()

for value in learning_rate:
    gbr = GradientBoostingRegressor(max_depth=2,random_state=2,n_estimators=30,learning_rate=value)
    gbr.fit(X_train,y_train)
    y_pred = gbr.predict(X_test)
    RMSE30.append(MSE(y_test,y_pred)**0.5)
    print('learning rate:', value, 'Error:', MSE(y_test,y_pred)**0.5)

sns.set_theme()
plt.figure(figsize=(15,7))
plt.plot(learning_rate,RMSE30)
plt.xlabel('Learning Rate with 30 Trees')
plt.ylabel('RMSE')
plt.title("RMSE vs Learning Rate")
plt.show()

for value in learning_rate:
    gbr = GradientBoostingRegressor(max_depth=2,random_state=2,n_estimators=3000,learning_rate=value)
    gbr.fit(X_train,y_train)
    y_pred = gbr.predict(X_test)
    RMSE3000.append(MSE(y_test,y_pred)**0.5)
    print('learning rate:', value, 'Error:', MSE(y_test,y_pred)**0.5)

sns.set_theme()
plt.figure(figsize=(15,7))
plt.plot(learning_rate,RMSE3000)
plt.xlabel('Learning Rate with 3000 Trees')
plt.ylabel('RMSE')
plt.title("RMSE vs Learning Rate")
plt.show()