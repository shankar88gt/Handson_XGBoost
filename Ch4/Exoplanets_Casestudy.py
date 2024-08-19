import pandas as pd
import warnings
warnings.filterwarnings('ignore')

#Fetch data
df_planets = pd.read_csv('/Users/shankarmanoharan/Stats/Handson_XGBoost/Ch4/exoplanets.csv')
X = df_planets.iloc[:,1:]
y = df_planets.iloc[:,0]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=2)

from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import time

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.fit_transform(y_test)

start = time.time()
df_planets.info()
end=time.time()
elasped = end - start
print("run time:" + str(elasped) + 'seconds')

start = time.time()
gbr = GradientBoostingClassifier(n_estimators=100,max_depth=2,random_state=2)
gbr.fit(X_train,y_train)
y_pred = gbr.predict(X_test)
score = accuracy_score(y_pred,y_test)
print('Score:' + str(score))
end = time.time()
elasped = end - start
print("run time:" + str(elasped) + 'seconds')


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.fit_transform(y_test)

start = time.time()
xg_reg = XGBClassifier(n_estimator=100,max_depth=2,random_state=2)
xg_reg.fit(X_train,y_train)
y_pred = xg_reg.predict(X_test)
score = accuracy_score(y_pred,y_test)
print('Score:' + str(score))
end = time.time()
elasped = end - start
print("run time:" + str(elasped) + 'seconds')




